import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.nn.functional as F


class LithographyDatasetMulti(Dataset):
    def __init__(self, data_dir, split='train'):
        base = Path(data_dir) / split
        self.input_dir = base / 'inputs'
        self.int_dir = base / 'intensities'
        self.resist_dir = base / 'resists'

        self.inputs = sorted(self.input_dir.glob('*.png'))
        self.ints = sorted(self.int_dir.glob('*.png'))
        self.resists = sorted(self.resist_dir.glob('*.png'))

        assert len(self.inputs) == len(self.ints) == len(self.resists)

    def __len__(self):
        return len(self.inputs)

    def _load_img(self, path):
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        x = self._load_img(self.inputs[idx])
        y_int = self._load_img(self.ints[idx])
        y_res = self._load_img(self.resists[idx])
        return x, y_int, y_res


class ConvBlockGN(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class LithoSurrogateNetMultiHead(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlockGN(1, base_ch)              # 64
        self.enc2 = ConvBlockGN(base_ch, base_ch*2)      # 128
        self.enc3 = ConvBlockGN(base_ch*2, base_ch*4)    # 256

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlockGN(base_ch*4, base_ch*8)  # 512

        # Decoder - matching encoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)  # 512 to 256
        self.dec3 = ConvBlockGN(base_ch*4 + base_ch*4, base_ch*4)         # 256+256=512 to 56

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)  # 256 to 128
        self.dec2 = ConvBlockGN(base_ch*2 + base_ch*2, base_ch*2)         # 128+128=256 to 128

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)    # 128 to 64
        self.dec1 = ConvBlockGN(base_ch + base_ch, base_ch)               # 64+64=128 to 64

        # Output heads
        self.head_intensity = nn.Conv2d(base_ch, 1, 1)
        self.head_resist = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # base_ch
        e2 = self.enc2(self.pool(e1))  # base_ch*2
        e3 = self.enc3(self.pool(e2))  # base_ch*4

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # base_ch*8

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # base_ch*4
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # base_ch*2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # base_ch

        # Output with clamp instead of sigmoid for better OPC gradients
        intensity = torch.clamp(self.head_intensity(d1), 0, 1)
        resist = torch.clamp(self.head_resist(d1), 0, 1)
        
        return intensity, resist


class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def get_edges(self, img):
        # Ensure sobel filters are on same device as input
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        edge_x = F.conv2d(img, sobel_x, padding=1)
        edge_y = F.conv2d(img, sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edges
    
    def forward(self, pred, target):
        # Standard MSE
        mse = torch.mean((pred - target) ** 2)
        
        # Edge-aware MSE
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        edge_mse = torch.mean((pred_edges - target_edges) ** 2)
        
        return mse + 2.0 * edge_mse  # Weight edge loss higher


class MultiHeadLoss(nn.Module):
    def __init__(self, w_resist=1.0, w_intensity=0.3):
        super().__init__()
        self.wr = w_resist
        self.wi = w_intensity
        self.edge_loss = EdgeAwareLoss()

    def forward(self, pred_int, pred_resist, gt_int, gt_resist):
        l_res = self.edge_loss(pred_resist, gt_resist)
        l_int = self.edge_loss(pred_int, gt_int)
        return self.wr * l_res + self.wi * l_int


class LithoSurrogateTrainerMulti:
    def __init__(self, data_dir, batch_size=16, learning_rate=1e-4,
                 device='cuda', seed=42):

        self._set_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.train_ds = LithographyDatasetMulti(data_dir, 'train')
        self.test_ds = LithographyDatasetMulti(data_dir, 'test')

        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size,
            shuffle=True, num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=batch_size,
            shuffle=False, num_workers=4
        )

        self.model = LithoSurrogateNetMultiHead().to(self.device)
        self.criterion = MultiHeadLoss()

        # Print model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params/1e6:.2f}M parameters")

        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, factor=0.5, patience=5, min_lr=1e-6
        )

        self.max_grad_norm = 1.0
        self.train_losses = []
        self.test_losses = []

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_epoch(self):
        self.model.train()
        total = 0.0

        for x, y_int, y_res in tqdm(self.train_loader, desc='Training'):
            x = x.to(self.device)
            y_int = y_int.to(self.device)
            y_res = y_res.to(self.device)

            self.opt.zero_grad()
            p_int, p_res = self.model(x)
            loss = self.criterion(p_int, p_res, y_int, y_res)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()

            total += loss.item()

        return total / len(self.train_loader)

    def eval_epoch(self):
        self.model.eval()
        total = 0.0

        with torch.no_grad():
            for x, y_int, y_res in tqdm(self.test_loader, desc='Evaluating'):
                x = x.to(self.device)
                y_int = y_int.to(self.device)
                y_res = y_res.to(self.device)
                p_int, p_res = self.model(x)
                total += self.criterion(p_int, p_res, y_int, y_res).item()

        return total / len(self.test_loader)

    def train(self, num_epochs=20, save_path='litho_surrogate_multi.pth'):
        best = float('inf')
        patience = 0

        for epoch in range(num_epochs):
            tl = self.train_epoch()
            vl = self.eval_epoch()

            self.train_losses.append(tl)
            self.test_losses.append(vl)
            self.sched.step(vl)

            print(f"Epoch {epoch+1}/{num_epochs} | Train {tl:.6f} | Val {vl:.6f}")

            if vl < best:
                best = vl
                patience = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience += 1
                if patience >= 15:
                    break

    def visualize_predictions(self, num_samples=10, show=False):
        self.model.eval()

        with torch.no_grad():
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                x, _, y_res = self.test_ds[i]
                x = x.unsqueeze(0).to(self.device)
                _, p_res = self.model(x)

                x_img = x.cpu().squeeze().numpy() * 255
                y_img = y_res.squeeze().numpy() * 255
                p_img = p_res.cpu().squeeze().numpy() * 255
                diff = np.abs(y_img - p_img)

                axes[i, 0].imshow(x_img, cmap='gray', vmin=0, vmax=255)
                axes[i, 1].imshow(p_img, cmap='gray', vmin=0, vmax=255)
                axes[i, 2].imshow(y_img, cmap='gray', vmin=0, vmax=255)
                im = axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=50)

                for j in range(4):
                    axes[i, j].axis('off')

                axes[i, 0].set_title('Input')
                axes[i, 1].set_title('Prediction')
                axes[i, 2].set_title('Ground Truth')
                axes[i, 3].set_title(f'Error {diff.mean():.2f}')

                plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig('predictions.png', dpi=150)
            if show:
                plt.show()
            plt.close()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))