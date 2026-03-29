# src/core/augmenters/mask_augmenter.py
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
import skimage.morphology as morphology
import random


class MaskAugmenter:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._operation_registry = [
            (2, lambda m: self.add_random_squares(m, num_squares=np.random.randint(10, 25), max_size=np.random.randint(5, 25))),
            (1, lambda m: self.dilate_uniform(m, size=np.random.randint(1, 5))),
            (1, lambda m: self.erode_uniform(m, size=np.random.randint(5))),
            (2, lambda m: self.add_corner_rounding(m, radius=np.random.randint(3, 10))),
            (1, lambda m: self.add_noise(m, noise_prob=np.random.uniform(0.01, 0.1))),
            (2, lambda m: self.add_expansion_boundary_noise(m, dilate_size=np.random.randint(1, 4), noise_density=np.random.uniform(0.2, 0.5))),
            (2, lambda m: self.add_erosion_boundary_noise(m, erode_size=np.random.randint(1, 4), noise_density=np.random.uniform(0.15, 0.25))),
            (2, lambda m: self.add_bidirectional_boundary_noise(m, morph_size=np.random.randint(1, 4), expansion_density=np.random.uniform(0.2, 0.4), erosion_density=np.random.uniform(0.2, 0.4))),
        ]

        self._weighted_ops = [
            fn for weight, fn in self._operation_registry for _ in range(weight)
        ]

        # Softening methods with equal weight
        self._softening_ops = [
            self._soften_blur,
            self._soften_noise,
            self._soften_sigmoid,
        ]


    # ── Existing augmentation methods (unchanged) ─────────────────────────

    def add_random_squares(self, mask, num_squares=5, max_size=10):
        result = mask.copy().astype(float)
        offsetFactor = 5
        mask_coords = np.where(mask > 0)
        if len(mask_coords[0]) == 0:
            return result.astype(mask.dtype)
        for _ in range(num_squares):
            size = np.random.randint(1, max_size)
            idx = np.random.randint(0, len(mask_coords[0]))
            center_y, center_x = mask_coords[0][idx], mask_coords[1][idx]
            offset = np.random.randint(-offsetFactor*size, offsetFactor*size, size=2)
            y = np.clip(center_y + offset[0], 0, mask.shape[0] - size)
            x = np.clip(center_x + offset[1], 0, mask.shape[1] - size)
            result[y:y+size, x:x+size] = 1
        return result.astype(mask.dtype)

    def dilate_uniform(self, mask, size=3):
        structure = morphology.disk(size)
        return ndimage.binary_dilation(mask, structure=structure).astype(mask.dtype)

    def erode_uniform(self, mask, size=3):
        structure = morphology.disk(size)
        return ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)

    def add_corner_rounding(self, mask, radius=2):
        structure = morphology.disk(radius)
        return ndimage.binary_closing(mask, structure=structure).astype(mask.dtype)

    def add_noise(self, mask, noise_prob=0.01):
        result = mask.copy()
        noise_mask = np.random.random(mask.shape) < noise_prob
        result[noise_mask] = 1 - result[noise_mask]
        return result

    def add_expansion_boundary_noise(self, mask, dilate_size=3, noise_density=0.3):
        result      = mask.copy().astype(float)
        dilated     = self.dilate_uniform(mask, size=dilate_size)
        expansion_region = (dilated > 0) & (mask == 0)
        noise_mask  = np.random.random(mask.shape) < noise_density
        result[expansion_region & noise_mask] = 1
        return result.astype(mask.dtype)

    def add_erosion_boundary_noise(self, mask, erode_size=3, noise_density=0.3):
        result      = mask.copy().astype(float)
        eroded      = self.erode_uniform(mask, size=erode_size)
        erosion_region = (mask > 0) & (eroded == 0)
        noise_mask  = np.random.random(mask.shape) < noise_density
        result[erosion_region & noise_mask] = 0
        return result.astype(mask.dtype)

    def add_bidirectional_boundary_noise(self, mask, morph_size=3, expansion_density=0.3, erosion_density=0.3):
        result = self.add_expansion_boundary_noise(mask, morph_size, expansion_density)
        return self.add_erosion_boundary_noise(result, morph_size, erosion_density)


    # ── Softening methods ─────────────────────────────────────────────────

    def _soften_blur(self, mask, sigma_range=(0.5, 2.0)):
        """Soften mask edges with Gaussian blur."""
        sigma = np.random.uniform(*sigma_range)
        return gaussian_filter(mask.astype(np.float32), sigma=sigma)

    def _soften_noise(self, mask, noise_scale=0.15):
        """Add per-pixel continuous noise to binary mask."""
        noise = np.random.uniform(-noise_scale, noise_scale, mask.shape)
        return np.clip(mask.astype(np.float32) + noise, 0.0, 1.0)

    def _soften_sigmoid(self, mask, sharpness_range=(2.0, 10.0)):
        """Apply sigmoid with random sharpness — low sharpness = soft edges."""
        k = np.random.uniform(*sharpness_range)
        return (1.0 / (1.0 + np.exp(-k * (mask.astype(np.float32) - 0.5)))).astype(np.float32)

    def _apply_random_softening(self, mask):
        """Apply one randomly chosen softening method."""
        fn = random.choice(self._softening_ops)
        return fn(mask)


    # ── Core augmentation ─────────────────────────────────────────────────

    def batch_augment(self, masks, augmentations_per_mask=5):
        masks     = np.asarray(masks)
        originals = np.repeat(masks, augmentations_per_mask, axis=0)
        augmented = np.array([self.random_augmentation(m) for m in originals])
        return augmented, originals

    def random_augmentation(self, mask):
        """Apply random structural augmentations, with 50% chance of softening."""
        result = mask.copy()
        n_ops  = np.random.randint(1, 4)
        chosen = np.random.choice(len(self._weighted_ops), size=n_ops, replace=True)
        for idx in chosen:
            result = self._weighted_ops[idx](result)

        # 50% chance of softening to simulate continuous mask values during optimisation
        if np.random.random() < 0.5:
            result = self._apply_random_softening(result)

        return result