# src/core/ml/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


class LithographyDataset(Dataset):
    """Dataset of (mask, illumination quadrant, wafer intensity, resist) tuples loaded per sample from disk."""

    def __init__(self, data_dir, split="train", max_samples=None):
        """Scan and validate PNG files under ./data/<data_dir>/<split>/{masks, illuminations, intensities, resists}."""
        base = Path("./data") / data_dir / split

        self.mask_paths      = sorted((base / "masks").glob("*.png"))
        self.illum_paths     = sorted((base / "illuminations").glob("*.png"))
        self.intensity_paths = sorted((base / "intensities").glob("*.png"))
        self.resist_paths    = sorted((base / "resists").glob("*.png"))

        assert len(self.mask_paths) == len(self.illum_paths) == \
               len(self.intensity_paths) == len(self.resist_paths), \
               "Mismatch in number of files across subdirectories"

        if max_samples is not None:
            self.mask_paths      = self.mask_paths[:max_samples]
            self.illum_paths     = self.illum_paths[:max_samples]
            self.intensity_paths = self.intensity_paths[:max_samples]
            self.resist_paths    = self.resist_paths[:max_samples]

        print(f"Found {len(self.mask_paths)} samples ({split})")

    def _load(self, path):
        """Load a grayscale PNG as a [1, H, W] float32 tensor normalised to [0, 1]."""
        return torch.from_numpy(
            np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        ).unsqueeze(0)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.mask_paths)

    def __getitem__(self, idx):
        """Return (mask, illum_q, intensity, resist) where illum_q is the bottom-right quadrant of the full illumination."""
        mask      = self._load(self.mask_paths[idx])
        intensity = self._load(self.intensity_paths[idx])
        resist    = self._load(self.resist_paths[idx])

        illum   = self._load(self.illum_paths[idx])
        _, H, W = illum.shape
        illum_q = illum[:, H // 2:, W // 2:]

        return mask, illum_q, intensity, resist


def build_dataloaders(config):
    """Build train and test DataLoaders from config['data'] with keys: data_dir, batch_size, num_workers, max_samples."""
    data_cfg    = config["data"]
    max_samples = data_cfg.get("max_samples", None)

    train_dataset = LithographyDataset(data_cfg["data_dir"], split="train", max_samples=max_samples)
    test_dataset  = LithographyDataset(data_cfg["data_dir"], split="test",  max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    config = {
        "data": {
            "data_dir": "augmented_medium",
            "batch_size": 4,
            "num_workers": 2,
            "max_samples": 400,
        }
    }

    train_loader, test_loader = build_dataloaders(config)

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Test batches  : {len(test_loader)}")
    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Test samples  : {len(test_loader.dataset)}")

    mask, illum_q, intensity, resist = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"  mask      : {tuple(mask.shape)}      min={mask.min():.3f}  max={mask.max():.3f}")
    print(f"  illum_q   : {tuple(illum_q.shape)}   min={illum_q.min():.3f}  max={illum_q.max():.3f}")
    print(f"  intensity : {tuple(intensity.shape)}  min={intensity.min():.3f}  max={intensity.max():.3f}")
    print(f"  resist    : {tuple(resist.shape)}     min={resist.min():.3f}  max={resist.max():.3f}")

    print("\nAll checks passed.")