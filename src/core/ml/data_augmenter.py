import numpy as np
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import skimage.draw as draw
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
import json
import src.core.simulator.masks as masks
import src.core.simulator.lithography_simulator as simulator


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
    
    def add_expansion_boundary_noise(self, mask, dilate_size=3, noise_density=0.3):
        result = mask.copy().astype(float)
        
        # Dilate the original mask
        dilated = self.dilate_isotropic(mask, size=dilate_size)
        
        # Find the expansion region (dilated - original)
        expansion_region = (dilated > 0) & (mask == 0)
        
        # Create noise mask for expansion region
        noise_mask = np.random.random(mask.shape) < noise_density
        
        # Apply noise only in expansion region
        add_noise = expansion_region & noise_mask
        result[add_noise] = 1
        
        return result.astype(mask.dtype)
    
    def add_erosion_boundary_noise(self, mask, erode_size=3, noise_density=0.3):
        result = mask.copy().astype(float)
        
        # Erode the original mask
        eroded = self.erode_isotropic(mask, size=erode_size)
        
        # Find the erosion region (original - eroded)
        erosion_region = (mask > 0) & (eroded == 0)
        
        # Create noise mask for erosion region
        noise_mask = np.random.random(mask.shape) < noise_density
        
        # Apply noise only in erosion region (remove pixels)
        remove_noise = erosion_region & noise_mask
        result[remove_noise] = 0
        
        return result.astype(mask.dtype)
    
    def add_bidirectional_boundary_noise(self, mask, morph_size=3, expansion_density=0.3, erosion_density=0.3):
        # Apply expansion noise
        result = self.add_expansion_boundary_noise(mask, morph_size, expansion_density)
        
        # Apply erosion noise
        result = self.add_erosion_boundary_noise(result, morph_size, erosion_density)
        
        return result
    
    def add_structured_noise(self, mask, noise_density=0.05, cluster_size=3):
        result = mask.copy().astype(float)
        h, w = mask.shape
        
        n_clusters = int(h * w * noise_density / (cluster_size ** 2))
        
        for _ in range(n_clusters):
            cy = np.random.randint(cluster_size, h - cluster_size)
            cx = np.random.randint(cluster_size, w - cluster_size)
            
            # Create small cluster
            y_range = slice(cy - cluster_size // 2, cy + cluster_size // 2 + 1)
            x_range = slice(cx - cluster_size // 2, cx + cluster_size // 2 + 1)
            
            # Flip the cluster
            result[y_range, x_range] = 1 - result[y_range, x_range]
        
        return (result > 0.5).astype(mask.dtype)
    
    def add_global_salt_pepper_noise(self, mask, noise_prob=0.05):
        result = mask.copy()
        noise_mask = np.random.random(mask.shape) < noise_prob
        result[noise_mask] = 1 - result[noise_mask]
        return result
    
    def add_local_dense_noise(self, mask, n_regions=5, region_size=50, noise_prob=0.3):
        result = mask.copy()
        h, w = mask.shape
        
        # Find locations near existing features
        mask_locations = np.argwhere(mask > 0)
        
        if len(mask_locations) == 0:
            # If no features, just add random regions
            mask_locations = np.random.randint(0, [h, w], size=(n_regions * 10, 2))
        
        for _ in range(n_regions):
            if len(mask_locations) > 0:
                # Pick a random location near an existing feature
                center_idx = np.random.randint(0, len(mask_locations))
                cy, cx = mask_locations[center_idx]
                
                # Define region around this location
                y_start = max(0, cy - region_size // 2)
                y_end = min(h, cy + region_size // 2)
                x_start = max(0, cx - region_size // 2)
                x_end = min(w, cx + region_size // 2)
                
                # Add dense noise in this region
                region = result[y_start:y_end, x_start:x_end]
                noise_mask = np.random.random(region.shape) < noise_prob
                region[noise_mask] = 1 - region[noise_mask]
                result[y_start:y_end, x_start:x_end] = region
        
        return result
    
    def add_edge_noise(self, mask, noise_prob=0.1, thickness=2):
        result = mask.copy()
        
        # Find edges using gradient
        gy, gx = np.gradient(mask.astype(float))
        edges = (np.abs(gx) + np.abs(gy)) > 0.1
        
        # Dilate edges to get a band
        edge_band = ndimage.binary_dilation(edges, iterations=thickness)
        
        # Add noise only in edge regions
        noise_mask = (np.random.random(mask.shape) < noise_prob) & edge_band
        result[noise_mask] = 1 - result[noise_mask]
        
        return result
    
    def add_sub_resolution_features(self, mask, feature_size=1, density=0.02):
        result = mask.copy()
        h, w = mask.shape
        
        n_features = int(h * w * density)
        
        for _ in range(n_features):
            y = np.random.randint(feature_size, h - feature_size)
            x = np.random.randint(feature_size, w - feature_size)
            
            # Add small pixel or line
            if np.random.random() < 0.5:
                # Single pixel
                result[y, x] = 1
            else:
                # Small line (horizontal or vertical)
                if np.random.random() < 0.5:
                    result[y, x:x+feature_size+1] = 1
                else:
                    result[y:y+feature_size+1, x] = 1
        
        return result
    
    def add_assist_features(self, mask, n_features=10, size_range=(2, 5), min_distance_from_main=5):

        result = mask.copy()
        h, w = mask.shape
        
        # Find main feature locations
        labeled, _ = ndimage.label(mask)
        
        for _ in range(n_features):
            size = np.random.randint(size_range[0], size_range[1] + 1)
            
            # Try to place near existing features
            attempts = 0
            while attempts < 10:
                cy = np.random.randint(size + min_distance_from_main, 
                                      h - size - min_distance_from_main)
                cx = np.random.randint(size + min_distance_from_main, 
                                      w - size - min_distance_from_main)
                
                # Check if near (but not on) existing features
                local_region = mask[cy-min_distance_from_main:cy+min_distance_from_main,
                                   cx-min_distance_from_main:cx+min_distance_from_main]
                
                if np.any(local_region) and not mask[cy, cx]:
                    # Add small feature
                    rr, cc = draw.disk((cy, cx), size, shape=(h, w))
                    result[rr, cc] = 1
                    break
                
                attempts += 1
        
        return result
    
    def add_serifs(self, mask, serif_size=3, probability=0.3):
        result = mask.copy()
        h, w = mask.shape
        
        # Find corners using convolution
        corners = np.zeros_like(mask, dtype=bool)
        
        # Simple corner detection
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mask[i, j]:
                    # Check for corner patterns
                    neighbors = mask[i-1:i+2, j-1:j+2]
                    if np.sum(neighbors) <= 5:  # Rough corner detection
                        corners[i, j] = True
        
        # Add serifs at corners
        corner_coords = np.argwhere(corners)
        for coord in corner_coords:
            if np.random.random() < probability:
                y, x = coord
                # Add small extension
                direction = np.random.choice(['h', 'v'])
                if direction == 'h':
                    x_end = min(w, x + serif_size)
                    result[y, x:x_end] = 1
                else:
                    y_end = min(h, y + serif_size)
                    result[y:y_end, x] = 1
        
        return result
    
    def add_checkerboard_noise(self, mask, checker_size=2, density=0.05):
        result = mask.copy()
        h, w = mask.shape
        
        n_checkers = int(h * w * density / (checker_size * 2) ** 2)
        
        for _ in range(n_checkers):
            cy = np.random.randint(checker_size * 2, h - checker_size * 2)
            cx = np.random.randint(checker_size * 2, w - checker_size * 2)
            
            # Create checkerboard pattern
            for i in range(2):
                for j in range(2):
                    if (i + j) % 2 == 0:
                        y_slice = slice(cy + i*checker_size, cy + (i+1)*checker_size)
                        x_slice = slice(cx + j*checker_size, cx + (j+1)*checker_size)
                        result[y_slice, x_slice] = 1 - result[y_slice, x_slice]
        
        return result
    
    def random_augmentation(self, mask, operations=None):
        if operations is None:
            base_ops = ['dilate', 'erode', 'dilate_aniso', 'erode_aniso', 
                       'add_blobs', 'remove_blobs', 'add_lines', 'remove_blocks',
                       'round_corners', 'noise']
            
            # Add new operations with higher probability
            optimizer_ops = ['structured_noise', 'edge_noise', 'sub_res_features',
                           'assist_features', 'serifs', 'checkerboard',
                           'global_noise', 'local_dense_noise',
                           'expansion_boundary', 'erosion_boundary', 'bidirectional_boundary']
            operations = base_ops + optimizer_ops * 2  # Double weight for new ops
        
        result = mask.copy()
        n_ops = np.random.randint(1, 6)  # At least 1 operation
        
        for _ in range(n_ops):
            op = np.random.choice(operations)
            
            # Original operations
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
            
            # New optimizer-focused operations
            elif op == 'structured_noise':
                density = np.random.uniform(0.01, 0.1)
                cluster_size = np.random.randint(2, 5)
                result = self.add_structured_noise(result, density, cluster_size)
            elif op == 'edge_noise':
                noise_prob = np.random.uniform(0.05, 0.2)
                thickness = np.random.randint(1, 4)
                result = self.add_edge_noise(result, noise_prob, thickness)
            elif op == 'sub_res_features':
                density = np.random.uniform(0.01, 0.05)
                result = self.add_sub_resolution_features(result, feature_size=1, density=density)
            elif op == 'assist_features':
                n_features = np.random.randint(5, 20)
                result = self.add_assist_features(result, n_features=n_features)
            elif op == 'serifs':
                serif_size = np.random.randint(2, 6)
                probability = np.random.uniform(0.2, 0.5)
                result = self.add_serifs(result, serif_size, probability)
            elif op == 'checkerboard':
                checker_size = np.random.randint(2, 4)
                density = np.random.uniform(0.02, 0.08)
                result = self.add_checkerboard_noise(result, checker_size, density)
            elif op == 'global_noise':
                noise_prob = np.random.uniform(0.02, 0.15)
                result = self.add_global_salt_pepper_noise(result, noise_prob)
            elif op == 'local_dense_noise':
                n_regions = np.random.randint(3, 10)
                region_size = np.random.randint(30, 100)
                noise_prob = np.random.uniform(0.2, 0.5)
                result = self.add_local_dense_noise(result, n_regions, region_size, noise_prob)
            
            # New boundary noise operations
            elif op == 'expansion_boundary':
                dilate_size = np.random.randint(2, 6)
                noise_density = np.random.uniform(0.2, 0.5)
                result = self.add_expansion_boundary_noise(result, dilate_size, noise_density)
            elif op == 'erosion_boundary':
                erode_size = np.random.randint(2, 6)
                noise_density = np.random.uniform(0.2, 0.5)
                result = self.add_erosion_boundary_noise(result, erode_size, noise_density)
            elif op == 'bidirectional_boundary':
                morph_size = np.random.randint(2, 5)
                exp_density = np.random.uniform(0.2, 0.4)
                ero_density = np.random.uniform(0.2, 0.4)
                result = self.add_bidirectional_boundary_noise(result, morph_size, exp_density, ero_density)
        
        return result
    
    def visualize_specific_augmentation(self, mask, augmentation_name, **kwargs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original
        axes[0].imshow(mask, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Augmented
        try:
            method = getattr(self, augmentation_name)
            augmented = method(mask, **kwargs)
            axes[1].imshow(augmented, cmap='gray', interpolation='nearest')
            
            params_str = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
            axes[1].set_title(f'{augmentation_name}\n({params_str})', fontsize=12)
            axes[1].axis('off')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{augmentation_name}_example.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return augmented
    
    def visualize_augmentations(self, mask, n_augmentations=5):
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.flatten()
        
        axes[0].imshow(mask, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        new_ops = [
            ('expansion_boundary', lambda m: self.add_expansion_boundary_noise(m, 3, 0.3)),
            ('erosion_boundary', lambda m: self.add_erosion_boundary_noise(m, 3, 0.3)),
            ('bidirectional_boundary', lambda m: self.add_bidirectional_boundary_noise(m, 3, 0.3, 0.3)),
            ('structured_noise', lambda m: self.add_structured_noise(m, 0.05, 3)),
            ('edge_noise', lambda m: self.add_edge_noise(m, 0.15, 2)),
            ('sub_res_features', lambda m: self.add_sub_resolution_features(m, 1, 0.03)),
            ('assist_features', lambda m: self.add_assist_features(m, 15)),
            ('global_noise', lambda m: self.add_global_salt_pepper_noise(m, 0.08)),
            ('local_dense_noise', lambda m: self.add_local_dense_noise(m, 5, 50, 0.3)),
        ]
        
        for i, (name, func) in enumerate(new_ops, 1):
            try:
                aug_mask = func(mask)
                axes[i].imshow(aug_mask, cmap='gray', interpolation='nearest')
                axes[i].set_title(name.replace('_', ' ').title(), fontsize=11)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {name}', ha='center', va='center')
                axes[i].axis('off')
        
        # Last two: combined augmentations
        for i in range(2):
            aug_mask = self.random_augmentation(mask)
            axes[10 + i].imshow(aug_mask, cmap='gray', interpolation='nearest')
            axes[10 + i].set_title(f'Combined Random {i+1}', fontsize=11)
            axes[10 + i].axis('off')
        
        plt.tight_layout()
        plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
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
    
    def save_dataset(self, masks_list, output_dir, sim_config, augmentations_per_mask=5, train_split=0.8):
        
        output_dir = Path(output_dir)
        
        # Create all directories
        splits = ['train', 'test']
        subdirs = ['inputs', 'intensities', 'resists']
        
        for split in splits:
            for subdir in subdirs:
                (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Generate augmented data
        augmented_batch, original_batch = self.batch_augment(
            masks_list, augmentations_per_mask
        )
        
        # Split into train/test
        n_total = len(augmented_batch)
        n_train = int(n_total * train_split)
        
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        split_indices = {
            'train': indices[:n_train],
            'test': indices[n_train:]
        }
        
        # Process each split
        for split_name, split_idx in split_indices.items():
            split_dir = output_dir / split_name
            
            # Get base ID from existing files
            base_id = len(list((split_dir / 'inputs').glob("*.png")))
            
            print(f"\nGenerating {split_name} dataset...")
            for idx, data_idx in enumerate(split_idx):
                if idx % 100 == 0:
                    print(f"Processing {split_name} sample {idx}/{len(split_idx)}")

                file_id = base_id + idx

                # Save input mask
                input_mask = augmented_batch[data_idx]
                input_img = Image.fromarray((input_mask * 255).astype(np.uint8))
                input_img.save(split_dir / 'inputs' / f"{file_id:06d}.png")
                
                # Run simulation
                litho_sim = simulator.LithographySimulator(sim_config)
                sim_results = litho_sim.simulate(input_mask)
                
                # Save wafer intensity
                wafer_intensity = sim_results["wafer_intensity"]
                intensity_normalized = (wafer_intensity * 255).astype(np.uint8)
                intensity_img = Image.fromarray(intensity_normalized)
                intensity_img.save(split_dir / 'intensities' / f"{file_id:06d}.png")
                
                # Save resist profile
                resist_profile = sim_results["resist_profile"]
                resist_normalized = (resist_profile * 255).astype(np.uint8)
                resist_img = Image.fromarray(resist_normalized)
                resist_img.save(split_dir / 'resists' / f"{file_id:06d}.png")

        print(f"\nDataset saved to {output_dir}")
        print(f"Train samples: {len(split_indices['train'])}")
        print(f"Test samples: {len(split_indices['test'])}")


if __name__ == "__main__":
    # Example usage with a simple test mask
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    num_masks = 4
    
    random_masks = masks.get_dataset_masks('./data/ganopc-data/artitgt', num_masks, **sim_config)
    
    augmenter = MaskAugmenter(seed=42)
    
    # Visualize all augmentations
    augmenter.visualize_augmentations(random_masks[0])
    
    # Visualize specific boundary noise augmentations
    print("\nExpansion boundary noise:")