import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class MultiTargetDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        base = Path(data_dir) / split
        self.input_dir = base / 'inputs'
        self.intensity_dir = base / 'intensities'
        self.resist_dir = base / 'resists'

        self.inputs = sorted(self.input_dir.glob('*.png'))
        self.intensities = sorted(self.intensity_dir.glob('*.png'))
        self.resists = sorted(self.resist_dir.glob('*.png'))

        assert len(self.inputs) == len(self.intensities) == len(self.resists)
        print(f"Loaded {len(self.inputs)} {split} samples (multi-target)")

    def __len__(self):
        return len(self.inputs)

    def _load_image(self, path):
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        x = self._load_image(self.inputs[idx])
        y_intensity = self._load_image(self.intensities[idx])
        y_resist = self._load_image(self.resists[idx])
        return x, y_intensity, y_resist
    

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


class MultiTargetUNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        
        self.enc1 = ConvBlockGN(1, base_ch)
        self.enc2 = ConvBlockGN(base_ch, base_ch * 2)
        self.enc3 = ConvBlockGN(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlockGN(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlockGN(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlockGN(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlockGN(base_ch * 2, base_ch)

        self.head_intensity = nn.Conv2d(base_ch, 1, 1)
        self.head_resist = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        intensity = torch.clamp(self.head_intensity(d1), 0, 1)
        resist = torch.clamp(self.head_resist(d1), 0, 1)
        
        return intensity, resist
    
    def predict(self, mask_np):
        self.eval()
        device = next(self.parameters()).device
        
        mask = np.array(mask_np, dtype=np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            intensity, resist = self(mask)
        
        intensity = intensity.cpu().squeeze().numpy()
        resist = resist.cpu().squeeze().numpy()
        
        return intensity, resist