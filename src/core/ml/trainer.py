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


class LithographyDataset(Dataset):
    def __init__(self, data_dir, split='train', target_type='intensities'):
        self.input_dir = Path(data_dir) / split / 'inputs'
        self.target_dir = Path(data_dir) / split / target_type
        
        self.input_files = sorted(list(self.input_dir.glob('*.png')))
        self.target_files = sorted(list(self.target_dir.glob('*.png')))
        
        assert len(self.input_files) == len(self.target_files), "Mismatch in input/target counts"
        print(f"Loaded {len(self.input_files)} {split} samples (target: {target_type})")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx]).convert('L')
        target_img = Image.open(self.target_files[idx]).convert('L')
        
        input_array = np.array(input_img, dtype=np.float32) / 255.0
        target_array = np.array(target_img, dtype=np.float32) / 255.0
        
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)
        target_tensor = torch.from_numpy(target_array).unsqueeze(0)
        
        return input_tensor, target_tensor


class SimpleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class LithoSurrogateNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=512):
        super().__init__()
        
        self.img_size = img_size
        
        # Encoder - 3 levels
        self.enc1 = SimpleConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = SimpleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = SimpleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = SimpleConvBlock(256, 512)
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = SimpleConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = SimpleConvBlock(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = SimpleConvBlock(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d1 = self.dec1(torch.cat([self.up1(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))
        
        return torch.sigmoid(self.out(d3))


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.8, edge_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.edge_weight = edge_weight
        
    def forward(self, pred, target):
        # MSE loss
        mse_loss = torch.mean((pred - target) ** 2)
        
        # Edge loss (gradient-based)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        edge_loss = torch.mean((pred_dx - target_dx) ** 2) + \
                    torch.mean((pred_dy - target_dy) ** 2)
        
        return self.mse_weight * mse_loss + self.edge_weight * edge_loss


class LithoSurrogateTrainer:
    def __init__(self, data_dir, target_type='intensities', batch_size=8, 
                 learning_rate=1e-3, device='cuda', img_size=512, seed=42):
        # Set all random seeds for reproducibility
        self._set_seed(seed)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.target_type = target_type
        print(f"Using device: {self.device}")
        print(f"Training mode: mask → {target_type}")
        print(f"Random seed: {seed}")
        
        self.train_dataset = LithographyDataset(data_dir, split='train', target_type=target_type)
        self.test_dataset = LithographyDataset(data_dir, split='test', target_type=target_type)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                      shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, 
                                     shuffle=False, num_workers=4)
        
        self.model = LithoSurrogateNet(img_size=img_size).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {num_params:,} parameters ({num_params/1e6:.2f}M)")
        
        self.criterion = CombinedLoss(mse_weight=0.8, edge_weight=0.2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, 
            min_lr=1e-6
        )
        
        # Gradient clipping threshold
        self.max_grad_norm = 1.0
        
        self.train_losses = []
        self.test_losses = []
        self.learning_rates = []
        
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'EPE': f'{loss.item():.6f}'})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def visualize_predictions(self, num_samples=4, show=False):
        self.model.eval()
        
        samples_per_page = 2
        num_pages = (num_samples + samples_per_page - 1) // samples_per_page
        
        all_targets = []
        all_outputs = []
        all_errors = []
        
        with torch.no_grad():
            for page in range(num_pages):
                fig, axes = plt.subplots(samples_per_page, 4, figsize=(16, 4 * samples_per_page))
                if samples_per_page == 1:
                    axes = axes.reshape(1, -1)
                
                fig.suptitle(f'Lithography Predictions - Page {page + 1}/{num_pages}', 
                            fontsize=16, fontweight='bold', y=0.98)
                
                for row in range(samples_per_page):
                    sample_idx = page * samples_per_page + row
                    
                    if sample_idx >= num_samples:
                        for col in range(4):
                            axes[row, col].axis('off')
                        continue
                    
                    inputs, targets = self.test_dataset[sample_idx]
                    inputs = inputs.unsqueeze(0).to(self.device)
                    outputs = self.model(inputs)
                    
                    input_img = inputs.cpu().squeeze().numpy() * 255
                    target_img = targets.squeeze().numpy() * 255
                    output_img = outputs.cpu().squeeze().numpy() * 255
                    diff_img = np.abs(target_img - output_img)
                    
                    all_targets.append(target_img.flatten())
                    all_outputs.append(output_img.flatten())
                    all_errors.append(diff_img.mean())
                    
                    # Input mask
                    axes[row, 0].imshow(input_img, cmap='gray', vmin=0, vmax=255)
                    axes[row, 0].set_title('Input Mask', fontsize=12, fontweight='bold')
                    axes[row, 0].axis('off')
                    
                    # Prediction
                    axes[row, 1].imshow(output_img, cmap='gray', vmin=0, vmax=255)
                    axes[row, 1].set_title('Prediction', fontsize=12, fontweight='bold')
                    axes[row, 1].axis('off')
                    
                    # Ground truth
                    axes[row, 2].imshow(target_img, cmap='gray', vmin=0, vmax=255)
                    axes[row, 2].set_title(f'Ground Truth', fontsize=12, fontweight='bold')
                    axes[row, 2].axis('off')
                    
                    # Error map
                    im = axes[row, 3].imshow(diff_img, cmap='hot', vmin=0, vmax=50)
                    axes[row, 3].set_title(f'Error (Mean: {diff_img.mean():.2f})', 
                                        fontsize=12, fontweight='bold')
                    axes[row, 3].axis('off')
                    
                    plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                filename = f'predictions_page_{page + 1}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved {filename}")
                
                if show:
                    plt.show()
                plt.close()
        
        # Summary statistics
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        mean_error = np.mean(all_errors)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY ({self.target_type})")
        print(f"{'='*60}")
        print(f"Samples analyzed: {num_samples}")
        print(f"Mean absolute error: {np.abs(all_targets - all_outputs).mean():.2f}")
        print(f"Average per-image error: {mean_error:.2f}")
        print(f"\nTarget - Range: [{all_targets.min():.1f}, {all_targets.max():.1f}], "
            f"Mean: {all_targets.mean():.1f}, Std: {all_targets.std():.1f}")
        print(f"Output - Range: [{all_outputs.min():.1f}, {all_outputs.max():.1f}], "
            f"Mean: {all_outputs.mean():.1f}, Std: {all_outputs.std():.1f}")
        print(f"{'='*60}\n")
        
    def plot_losses(self, show=False):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train EPE', linewidth=2)
        axes[0].plot(epochs, self.test_losses, 'r-', label='Test EPE', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('EPE Loss (L1)', fontsize=12)
        axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        if len(self.learning_rates) > 0:
            axes[1].plot(epochs, self.learning_rates, 'g-', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Learning Rate', fontsize=12)
            axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def train(self, num_epochs=50, save_path='litho_surrogate.pth'):
        print(f"\nStarting training for {num_epochs} epochs...")
        
        print("\nGenerating initial predictions (before training)...")
        self.visualize_predictions(num_samples=6, show=False)
        print("Initial predictions saved")
        
        best_test_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            print(f"Train EPE: {train_loss:.6f}, Test EPE: {test_loss:.6f}, LR: {current_lr:.2e}")
            
            # Learning rate scheduling
            self.scheduler.step(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping after {max_patience} epochs without improvement")
                    break
            
            if (epoch + 1) % 10 == 0:
                self.plot_losses(show=False)
        
        print(f"\nTraining complete! Best test EPE: {best_test_loss:.6f}")
        
        self.plot_losses(show=True)
        self.visualize_predictions(num_samples=6, show=True)
    
    def load_model(self, path='litho_surrogate.pth'):
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}")


