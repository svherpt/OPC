import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class LithographyDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.input_dir = Path(data_dir) / split / 'inputs'
        self.target_dir = Path(data_dir) / split / 'targets'
        
        self.input_files = sorted(list(self.input_dir.glob('*.png')))
        self.target_files = sorted(list(self.target_dir.glob('*.png')))
        
        assert len(self.input_files) == len(self.target_files), "Mismatch in input/target counts"
        print(f"Loaded {len(self.input_files)} {split} samples")
    
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


class LithoSurrogateNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(LithoSurrogateNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LithoSurrogateTrainer:
    def __init__(self, data_dir, batch_size=8, learning_rate=1e-3, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.train_dataset = LithographyDataset(data_dir, split='train')
        self.test_dataset = LithographyDataset(data_dir, split='test')
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                      shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, 
                                     shuffle=False, num_workers=4)
        
        self.model = LithoSurrogateNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.test_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
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
    
    def visualize_predictions(self, num_samples=4):
        self.model.eval()
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        with torch.no_grad():
            for i in range(num_samples):
                inputs, targets = self.test_dataset[i]
                inputs = inputs.unsqueeze(0).to(self.device)
                
                outputs = self.model(inputs)
                
                input_img = inputs.cpu().squeeze().numpy()
                target_img = targets.squeeze().numpy()
                output_img = outputs.cpu().squeeze().numpy()
                
                axes[i, 0].imshow(input_img, cmap='gray')
                axes[i, 0].set_title('Input Mask')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(target_img, cmap='gray')
                axes[i, 1].set_title('Ground Truth Intensity')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(output_img, cmap='gray')
                axes[i, 2].set_title('Predicted Intensity')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def train(self, num_epochs=50, save_path='litho_surrogate.pth'):
        print(f"\nStarting training for {num_epochs} epochs...")
        best_test_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
            
            if (epoch + 1) % 10 == 0:
                self.plot_losses()
                self.visualize_predictions()
        
        print(f"\nTraining complete! Best test loss: {best_test_loss:.6f}")
        self.plot_losses()
        self.visualize_predictions()
    
    def load_model(self, path='litho_surrogate.pth'):
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}")


if __name__ == "__main__":
    data_dir = "./augmented_dataset/"
    
    trainer = LithoSurrogateTrainer(
        data_dir=data_dir,
        batch_size=16,
        learning_rate=1e-3,
        device='cuda'
    )
    
    trainer.train(num_epochs=10, save_path='litho_surrogate.pth')
    
    trainer.visualize_predictions(num_samples=6)