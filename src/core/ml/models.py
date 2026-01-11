from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F


class LithographyDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        base = Path('./data') / data_dir / split

        self.mask_dir = base / "masks"
        self.illum_dir = base / "illuminations"
        self.intensity_dir = base / "intensities"
        self.resist_dir = base / "resists"

        self.masks = sorted(self.mask_dir.glob("*.png"))
        self.illums = sorted(self.illum_dir.glob("*.png"))
        self.intensities = sorted(self.intensity_dir.glob("*.png"))
        self.resists = sorted(self.resist_dir.glob("*.png"))

        assert len(self.masks) == len(self.illums) == len(self.intensities) == len(self.resists)

    def __len__(self):
        return len(self.masks)

    def _load(self, path):
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        mask = self._load(self.masks[idx])              # 1×256×256
        illum_full = self._load(self.illums[idx])       # 1×64×64

        H, W = illum_full.shape[-2:]
        illum_q = illum_full[:, H // 2 :, W // 2 :]      # 1×32×32

        intensity = self._load(self.intensities[idx])
        resist = self._load(self.resists[idx])

        return mask, illum_q, intensity, resist

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

class LithographyUNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()

        # Mask encoder
        self.enc1 = ConvBlockGN(1, base_ch)
        self.enc2 = ConvBlockGN(base_ch, base_ch*2)
        self.enc3 = ConvBlockGN(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)

        # Illumination encoder
        self.illum_enc1 = ConvBlockGN(1, base_ch)
        self.illum_enc2 = ConvBlockGN(base_ch, base_ch*2)
        self.illum_enc3 = ConvBlockGN(base_ch*2, base_ch*4)
        self.illum_proj = nn.Conv2d(base_ch*4, base_ch*4, 1)  # project to same channels as mask

        # Bottleneck
        self.bottleneck = ConvBlockGN(base_ch*4, base_ch*8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlockGN(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlockGN(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlockGN(base_ch*2, base_ch)

        self.head_intensity = nn.Conv2d(base_ch, 1, 1)
        self.head_resist = nn.Conv2d(base_ch, 1, 1)
        
        # Initialize output heads with better initialization
        nn.init.xavier_uniform_(self.head_intensity.weight, gain=1.0)
        nn.init.constant_(self.head_intensity.bias, 0.1)  # Small positive bias for intensity
        nn.init.xavier_uniform_(self.head_resist.weight, gain=1.0)
        nn.init.constant_(self.head_resist.bias, 0.0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total_params:,} parameters")

    def forward(self, mask, illum_q):
        # Mask encoder
        e1 = self.enc1(mask)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Illumination encoder
        i1 = self.illum_enc1(illum_q)
        i2 = self.illum_enc2(self.pool(i1))
        i3 = self.illum_enc3(self.pool(i2))
        i_proj = self.illum_proj(i3)

        # Pool mask features once
        m3_pooled = self.pool(e3)
        
        # Upsample illumination spatially to match pooled mask features
        i_proj_up = F.interpolate(i_proj, size=m3_pooled.shape[-2:], mode='bilinear', align_corners=False)

        # Combine features at bottleneck
        b = self.bottleneck(m3_pooled + i_proj_up)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Apply heads - use sigmoid instead of clamp for better gradient flow
        intensity = torch.sigmoid(self.head_intensity(d1))
        resist = torch.sigmoid(self.head_resist(d1))
        return intensity, resist

    def predict(self, mask_np, illum_np):
        self.eval()
        device = next(self.parameters()).device

        mask = torch.from_numpy(mask_np.astype('float32')).unsqueeze(0).unsqueeze(0)
        illum = torch.from_numpy(illum_np.astype('float32')).unsqueeze(0).unsqueeze(0)
        mask, illum = mask.to(device), illum.to(device)

        with torch.no_grad():
            intensity, resist = self(mask, illum)

        return intensity.squeeze().cpu().numpy(), resist.squeeze().cpu().numpy()