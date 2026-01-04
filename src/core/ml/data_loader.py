import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image


class SingleTargetDataset(Dataset):
    def __init__(self, data_dir, split='train', target_type='resists'):
        self.input_dir = Path(data_dir) / split / 'inputs'
        self.target_dir = Path(data_dir) / split / target_type
        
        self.input_files = sorted(list(self.input_dir.glob('*.png')))
        self.target_files = sorted(list(self.target_dir.glob('*.png')))
        
        assert len(self.input_files) == len(self.target_files)
        print(f"Loaded {len(self.input_files)} {split} samples (target: {target_type})")
    
    def __len__(self):
        return len(self.input_files)
    
    def _load_image(self, path):
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    
    def __getitem__(self, idx):
        input_tensor = self._load_image(self.input_files[idx])
        target_tensor = self._load_image(self.target_files[idx])
        return input_tensor, target_tensor


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