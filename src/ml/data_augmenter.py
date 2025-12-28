import numpy as np
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import skimage.draw as draw
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from PIL import Image
import src.simulator.masks as masks
from src.simulator import lithography_simulator as simulator

class MaskAugmenter:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def dilate_isotropic(self, mask, size=3):
        structure = morphology.disk(size)
        return ndimage.binary_dilation(mask, structure=structure).astype(mask.dtype)
    
    def erode_isotropic(self, mask, size=3):
        structure = morphology.disk(size)
        return ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)
    
    def dilate_anisotropic(self, mask, x_size=3, y_size=1):
        structure = morphology.footprint_rectangle((2*y_size + 1, 2*x_size + 1))
        return ndimage.binary_dilation(mask, structure=structure).astype(mask.dtype)
    
    def erode_anisotropic(self, mask, x_size=3, y_size=1):
        structure = morphology.footprint_rectangle((2*y_size + 1, 2*x_size + 1))
        return ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)
    
    def add_random_blobs(self, mask, n_blobs=5, size_range=(3, 10), shape='circle'):
        result = mask.copy()
        h, w = mask.shape
        
        for _ in range(n_blobs):
            size = np.random.randint(size_range[0], size_range[1] + 1)
            cx = np.random.randint(size, w - size)
            cy = np.random.randint(size, h - size)
            
            blob_shape = shape if shape != 'random' else np.random.choice(['circle', 'square'])
            
            if blob_shape == 'circle':
                rr, cc = draw.disk((cy, cx), size, shape=(h, w))
                result[rr, cc] = 1
            else:
                start = (max(0, cy - size), max(0, cx - size))
                end = (min(h, cy + size), min(w, cx + size))
                rr, cc = draw.rectangle(start, end=end, shape=(h, w))
                result[rr, cc] = 1
        
        return result
    
    def remove_random_blobs(self, mask, removal_prob=0.3, min_size=5):
        labeled, n_features = ndimage.label(mask)
        result = mask.copy()
        
        for label_id in range(1, n_features + 1):
            blob = labeled == label_id
            blob_size = np.sum(blob)
            
            if blob_size >= min_size and np.random.random() < removal_prob:
                result[blob] = 0
        
        return result
    
    def add_random_lines(self, mask, n_lines=3, thickness_range=(1, 3), length_range=(5, 20)):
        result = mask.copy()
        h, w = mask.shape
        
        for _ in range(n_lines):
            thickness = np.random.randint(thickness_range[0], thickness_range[1] + 1)
            length = np.random.randint(length_range[0], length_range[1] + 1)
            
            orientation = np.random.choice(['horizontal', 'vertical'])
            
            if orientation == 'horizontal':
                y = np.random.randint(0, h - thickness)
                x = np.random.randint(0, w - length)
                result[y:y+thickness, x:x+length] = 1
            else:
                y = np.random.randint(0, h - length)
                x = np.random.randint(0, w - thickness)
                result[y:y+length, x:x+thickness] = 1
        
        return result
    
    def remove_white_blocks(self, mask, n_blocks=2, size_range=(3, 8)):
        result = mask.copy()
        h, w = mask.shape
        
        for _ in range(n_blocks):
            size_y = np.random.randint(size_range[0], size_range[1] + 1)
            size_x = np.random.randint(size_range[0], size_range[1] + 1)
            
            y = np.random.randint(0, max(1, h - size_y))
            x = np.random.randint(0, max(1, w - size_x))
            
            result[y:y+size_y, x:x+size_x] = 0
        
        return result
    
    def add_corner_rounding(self, mask, radius=2):
        structure = morphology.disk(radius)
        return ndimage.binary_closing(mask, structure=structure).astype(mask.dtype)
    
    def add_noise(self, mask, noise_prob=0.01):
        result = mask.copy()
        noise_mask = np.random.random(mask.shape) < noise_prob
        result[noise_mask] = 1 - result[noise_mask]
        return result
    
    def random_augmentation(self, mask, operations=None):
        if operations is None:
            operations = ['dilate', 'erode', 'dilate_aniso', 'erode_aniso', 
                         'add_blobs', 'remove_blobs', 'add_lines', 'remove_blocks',
                         'round_corners', 'noise']
        
        result = mask.copy()
        n_ops = np.random.randint(0, 5)
        
        for _ in range(n_ops):
            op = np.random.choice(operations)
            
            if op == 'dilate':
                size = np.random.randint(1, 10)
                result = self.dilate_isotropic(result, size)
            elif op == 'erode':
                size = np.random.randint(1, 10)
                result = self.erode_isotropic(result, size)
            elif op == 'dilate_aniso':
                x_size = np.random.randint(1, 25)
                y_size = np.random.randint(1, 25)
                result = self.dilate_anisotropic(result, x_size, y_size)
            elif op == 'erode_aniso':
                x_size = np.random.randint(1, 25)
                y_size = np.random.randint(1, 25)
                result = self.erode_anisotropic(result, x_size, y_size)
            elif op == 'add_blobs':
                n_blobs = np.random.randint(1, 20)
                result = self.add_random_blobs(result, n_blobs, size_range=(5, 25))
            elif op == 'remove_blobs':
                result = self.remove_random_blobs(result, removal_prob=0.2)
            elif op == 'add_lines':
                n_lines = np.random.randint(1, 10)
                result = self.add_random_lines(result, n_lines=n_lines, 
                                              thickness_range=(1, 3), length_range=(5, 15))
            elif op == 'remove_blocks':
                n_blocks = np.random.randint(1, 10)
                result = self.remove_white_blocks(result, n_blocks=n_blocks, size_range=(3, 8))
            elif op == 'round_corners':
                radius = np.random.randint(1, 10)
                result = self.add_corner_rounding(result, radius)
            elif op == 'noise':
                result = self.add_noise(result, noise_prob=0.005)
        
        return result
    
    def batch_augment(self, masks, augmentations_per_mask=5):
        if isinstance(masks, list):
            masks = np.array(masks)
        
        n_masks = len(masks)
        augmented = []
        originals = []
        
        for i in range(n_masks):
            for _ in range(augmentations_per_mask):
                aug_mask = self.random_augmentation(masks[i])
                augmented.append(aug_mask)
                originals.append(masks[i])
        
        return np.array(augmented), np.array(originals)
    
    def visualize_augmentations(self, mask, n_augmentations=5):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        axes[0].imshow(mask, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original Mask', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        for i in range(n_augmentations):
            aug_mask = self.random_augmentation(mask)
            axes[i+1].imshow(aug_mask, cmap='gray', interpolation='nearest')
            axes[i+1].set_title(f'Augmentation {i+1}', fontsize=12)
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_dataset(self, masks_list, output_dir, sim_config, augmentations_per_mask=5, train_split=0.8):
        output_dir = Path(output_dir)
        train_dir = output_dir / 'train'
        test_dir = output_dir / 'test'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        (train_dir / 'inputs').mkdir(exist_ok=True)
        (train_dir / 'targets').mkdir(exist_ok=True)
        (test_dir / 'inputs').mkdir(exist_ok=True)
        (test_dir / 'targets').mkdir(exist_ok=True)
        
        augmented_batch, original_batch = self.batch_augment(masks_list, augmentations_per_mask)
        
        n_total = len(augmented_batch)
        n_train = int(n_total * train_split)
        
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        print("Generating train dataset...")
        for idx, data_idx in enumerate(train_indices):
            if idx % 100 == 0:
                print(f"Processing train sample {idx}/{len(train_indices)}")
            
            input_mask = augmented_batch[data_idx]
            input_img = Image.fromarray((input_mask * 255).astype(np.uint8))
            input_img.save(train_dir / 'inputs' / f'{idx:06d}.png')
            
            litho_sim = simulator.LithographySimulator(sim_config)
            sim_results = litho_sim.simulate(input_mask)
            wafer_intensity = sim_results["wafer_intensity"]
            
            intensity_min = wafer_intensity.min()
            intensity_max = wafer_intensity.max()

            if intensity_max == 0:
                intensity_max = 1

            intensity_normalized = ((wafer_intensity - intensity_min) / 
                                   (intensity_max - intensity_min) * 255).astype(np.uint8)
            target_img = Image.fromarray(intensity_normalized)
            target_img.save(train_dir / 'targets' / f'{idx:06d}.png')
        
        print("Generating test dataset...")
        for idx, data_idx in enumerate(test_indices):
            if idx % 100 == 0:
                print(f"Processing test sample {idx}/{len(test_indices)}")
            
            input_mask = augmented_batch[data_idx]
            input_img = Image.fromarray((input_mask * 255).astype(np.uint8))
            input_img.save(test_dir / 'inputs' / f'{idx:06d}.png')
            
            litho_sim = simulator.LithographySimulator(sim_config)
            sim_results = litho_sim.simulate(input_mask)
            wafer_intensity = sim_results["wafer_intensity"]

            intensity_min = wafer_intensity.min()
            intensity_max = wafer_intensity.max()

            if intensity_max == 0:
                intensity_max = 1

            intensity_normalized = ((wafer_intensity - intensity_min) / 
                                   (intensity_max - intensity_min) * 255).astype(np.uint8)
            
            target_img = Image.fromarray(intensity_normalized)
            target_img.save(test_dir / 'targets' / f'{idx:06d}.png')
        
        print(f"Dataset saved to {output_dir}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Test samples: {len(test_indices)}")
        print(f"Train/Test split: {train_split:.0%}/{1-train_split:.0%}")


if __name__ == "__main__":
    mask = masks.read_mask_from_img(mask_id='21', mask_grid_size=512)
    
    augmenter = MaskAugmenter()
    
    augmenter.visualize_augmentations(mask, n_augmentations=5)

    