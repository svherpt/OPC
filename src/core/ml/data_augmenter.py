import numpy as np
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import skimage.draw as draw
import random
from pathlib import Path
from PIL import Image
import json
import src.core.simulator.masks as masks
import src.core.simulator.lithography_simulator as simulator


# Contains a bunch of different mask augmentation methods in order to generate 
class MaskAugmenter:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    

    def dilate_uniform(self, mask, size=3):
        structure = morphology.disk(size)
        return ndimage.binary_dilation(mask, structure=structure).astype(mask.dtype)
    

    def erode_uniform(self, mask, size=3):
        structure = morphology.disk(size)
        return ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)
    

    def dilate_directional(self, mask, x_size=3, y_size=1):
        structure = morphology.footprint_rectangle((2*y_size + 1, 2*x_size + 1))
        return ndimage.binary_dilation(mask, structure=structure).astype(mask.dtype)
    

    def erode_directional(self, mask, x_size=3, y_size=1):
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
        dilated = self.dilate_uniform(mask, size=dilate_size)
        
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
        eroded = self.erode_uniform(mask, size=erode_size)
        
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
    

    def random_augmentation(self, mask, operations=None):
        if operations is None:
            base_ops = ['dilate_uniform', 'erode_uniform', 'dilate_directional', 'erode_directional', 
                        'add_blobs', 'remove_blobs', 'add_lines', 'remove_blocks',
                        'round_corners', 'noise']
            
            # Add new operations with higher probability
            optimizer_ops = ['structured_noise', 'edge_noise',
                            'global_noise', 'local_dense_noise',
                            'expansion_boundary', 'erosion_boundary', 'bidirectional_boundary']
            operations = base_ops + optimizer_ops * 2
        
        result = mask.copy()
        n_ops = np.random.randint(1, 6)  # At least 1 operation
        
        for _ in range(n_ops):
            op = np.random.choice(operations)
            
            if op == 'dilate_uniform':
                size = np.random.randint(1, 10)
                result = self.dilate_uniform(result, size)
            elif op == 'erode_uniform':
                size = np.random.randint(1, 10)
                result = self.erode_uniform(result, size)
            elif op == 'dilate_directional':
                x_size = np.random.randint(1, 25)
                y_size = np.random.randint(1, 25)
                result = self.dilate_directional(result, x_size, y_size)
            elif op == 'erode_directional':
                x_size = np.random.randint(1, 25)
                y_size = np.random.randint(1, 25)
                result = self.erode_directional(result, x_size, y_size)
            elif op == 'add_blobs':
                n_blobs = np.random.randint(1, 20)
                result = self.add_random_blobs(result, n_blobs, size_range=(5, 25))
            elif op == 'remove_blobs':
                result = self.remove_random_blobs(result, removal_prob=0.2)
            elif op == 'add_lines':
                n_lines = np.random.randint(1, 10)
                result = self.add_random_lines(result, n_lines=n_lines, thickness_range=(1, 3), length_range=(5, 15))
            elif op == 'remove_blocks':
                n_blocks = np.random.randint(1, 10)
                result = self.remove_white_blocks(result, n_blocks=n_blocks, size_range=(3, 8))
            elif op == 'round_corners':
                radius = np.random.randint(1, 10)
                result = self.add_corner_rounding(result, radius)
            elif op == 'noise':
                result = self.add_noise(result, noise_prob=0.005)
            elif op == 'structured_noise':
                density = np.random.uniform(0.01, 0.1)
                cluster_size = np.random.randint(2, 5)
                result = self.add_structured_noise(result, density, cluster_size)
            elif op == 'edge_noise':
                noise_prob = np.random.uniform(0.05, 0.2)
                thickness = np.random.randint(1, 4)
                result = self.add_edge_noise(result, noise_prob, thickness)
            elif op == 'global_noise':
                noise_prob = np.random.uniform(0.02, 0.15)
                result = self.add_global_salt_pepper_noise(result, noise_prob)
            elif op == 'local_dense_noise':
                n_regions = np.random.randint(3, 10)
                region_size = np.random.randint(30, 100)
                noise_prob = np.random.uniform(0.2, 0.5)
                result = self.add_local_dense_noise(result, n_regions, region_size, noise_prob)
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

    def save_dataset(self, masks_list, output_dir, sim_config, augmentations_per_mask=5, train_split=0.8):
        output_dir = Path('./data/' + output_dir)
        
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


def generate_n_augmentations(num_masks, n_augmentations=5, output_dir='augmented_medium'):
    augmenter = MaskAugmenter()

    random_masks = masks.get_dataset_masks('ganopc-data/artitgt', num_masks, **sim_config)
    augmenter.save_dataset(random_masks, output_dir=output_dir, sim_config=sim_config, augmentations_per_mask=n_augmentations, train_split=0.8)


if __name__ == "__main__":
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)

    generate_n_augmentations(10, 1, output_dir='test')

    # for i in range(5):
    #     generate_n_augmentations(1000, 10, output_dir='./augmented_massive')