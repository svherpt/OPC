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
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.ndimage import rotate
from scipy.ndimage import rotate, binary_dilation, binary_erosion
from skimage.morphology import disk

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

import numpy as np
import random
from scipy.ndimage import rotate, binary_dilation, binary_erosion
from skimage.morphology import disk

class LightSourceAugmenter:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _loguniform(self, low, high, size=None):
        # Sample from log-uniform distribution for wide parameter ranges
        log_low = np.log10(low)
        log_high = np.log10(high)
        if size is None:
            return 10 ** np.random.uniform(log_low, log_high)
        return 10 ** np.random.uniform(log_low, log_high, size=size)
    
    def generate_annular(self, grid_size, max_spatial_frequency, n_modes=None):
        # Ring pattern - common in real lithography
        if n_modes is None:
            n_modes = np.random.randint(4, 12)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        ring_radius = np.random.uniform(0.4, 0.8) * max_spatial_frequency
        ring_width = self._loguniform(0.05, 0.2) * max_spatial_frequency
        
        for i in range(n_modes):
            angle = 2 * np.pi * i / n_modes + np.random.uniform(-0.3, 0.3)
            xi = ring_radius * np.cos(angle)
            yi = ring_radius * np.sin(angle)
            sigma = ring_width * np.random.uniform(0.5, 1.5)
            intensity = self._loguniform(0.1, 3.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_dipole(self, grid_size, max_spatial_frequency):
        # 2 opposite modes
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        angle = np.random.uniform(0, np.pi)
        distance = np.random.uniform(0.3, 0.9) * max_spatial_frequency
        
        for sign in [1, -1]:
            xi = sign * distance * np.cos(angle)
            yi = sign * distance * np.sin(angle)
            sigma = self._loguniform(0.01, 0.15) * max_spatial_frequency
            intensity = self._loguniform(0.5, 3.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_quadrupole(self, grid_size, max_spatial_frequency):
        # 4-way symmetric pattern
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        distance = np.random.uniform(0.3, 0.9) * max_spatial_frequency
        angle_offset = np.random.uniform(0, np.pi/2)
        
        for i in range(4):
            angle = np.pi/2 * i + angle_offset
            xi = distance * np.cos(angle)
            yi = distance * np.sin(angle)
            sigma = self._loguniform(0.01, 0.15) * max_spatial_frequency
            intensity = self._loguniform(0.5, 3.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_off_axis(self, grid_size, max_spatial_frequency, n_modes=None):
        # Modes pushed toward edges
        if n_modes is None:
            n_modes = np.random.randint(2, 8)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        for _ in range(n_modes):
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(0.7, 1.0) * max_spatial_frequency
            xi = distance * np.cos(angle)
            yi = distance * np.sin(angle)
            sigma = self._loguniform(0.005, 0.1) * max_spatial_frequency
            intensity = self._loguniform(0.1, 3.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_clustered(self, grid_size, max_spatial_frequency, n_modes=None):
        # Many modes concentrated in one region
        if n_modes is None:
            n_modes = np.random.randint(5, 15)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Define cluster center and spread
        cluster_xi = np.random.uniform(0.2, 0.8) * max_spatial_frequency
        cluster_yi = np.random.uniform(0.2, 0.8) * max_spatial_frequency
        cluster_spread = np.random.uniform(0.1, 0.3) * max_spatial_frequency
        
        for _ in range(n_modes):
            xi = cluster_xi + np.random.uniform(-cluster_spread, cluster_spread)
            yi = cluster_yi + np.random.uniform(-cluster_spread, cluster_spread)
            sigma = self._loguniform(0.005, 0.1) * max_spatial_frequency
            intensity = self._loguniform(0.1, 2.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_uniform_random(self, grid_size, max_spatial_frequency, n_modes=None):
        # Fully random scattered modes
        if n_modes is None:
            n_modes = np.random.randint(3, 20)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        for _ in range(n_modes):
            # Full coverage including edges
            xi = np.random.uniform(0.0, 1.0) * max_spatial_frequency
            yi = np.random.uniform(0.0, 1.0) * max_spatial_frequency
            sigma = self._loguniform(0.003, 0.25) * max_spatial_frequency
            intensity = self._loguniform(0.05, 5.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_sparse_strong(self, grid_size, max_spatial_frequency):
        # 1-3 very strong modes
        n_modes = np.random.randint(1, 4)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        for _ in range(n_modes):
            xi = np.random.uniform(0.1, 0.9) * max_spatial_frequency
            yi = np.random.uniform(0.1, 0.9) * max_spatial_frequency
            sigma = self._loguniform(0.02, 0.15) * max_spatial_frequency
            intensity = self._loguniform(1.0, 10.0)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_dense_weak(self, grid_size, max_spatial_frequency):
        # Many weak modes
        n_modes = np.random.randint(10, 30)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        for _ in range(n_modes):
            xi = np.random.uniform(0.0, 1.0) * max_spatial_frequency
            yi = np.random.uniform(0.0, 1.0) * max_spatial_frequency
            sigma = self._loguniform(0.005, 0.08) * max_spatial_frequency
            intensity = self._loguniform(0.05, 0.5)
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def generate_center_strong(self, grid_size, max_spatial_frequency):
        # Strong modes concentrated at center (low spatial frequencies)
        n_modes = np.random.randint(3, 8)
        
        illumination = np.zeros((grid_size, grid_size), dtype=float)
        x = np.linspace(0, max_spatial_frequency, grid_size)
        y = np.linspace(0, max_spatial_frequency, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Center region (low spatial frequency)
        center_radius = np.random.uniform(0.1, 0.3) * max_spatial_frequency
        
        for _ in range(n_modes):
            # Sample within center circle
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(0, center_radius)
            xi = distance * np.cos(angle)
            yi = distance * np.sin(angle)
            sigma = self._loguniform(0.03, 0.2) * max_spatial_frequency
            intensity = self._loguniform(1.0, 5.0)  # Strong
            
            illumination += intensity * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
        
        return illumination
    
    def apply_morphological_ops(self, illumination):
        # Apply random morphological operations to modify illumination shape
        ops = ['dilate', 'erode', 'none']
        op = np.random.choice(ops, p=[0.3, 0.3, 0.4])
        
        if op == 'none':
            return illumination
        
        # Threshold to create binary mask of "bright" regions
        threshold = np.random.uniform(0.1, 0.3) * np.max(illumination)
        binary_mask = illumination > threshold
        
        # Apply morphological operation
        size = np.random.randint(1, 4)
        structure = disk(size)
        
        if op == 'dilate':
            modified_mask = binary_dilation(binary_mask, structure)
        else:  # erode
            modified_mask = binary_erosion(binary_mask, structure)
        
        # Blend back with original - fade regions that are removed
        result = illumination.copy()
        fade_factor = np.random.uniform(0.3, 0.7)
        result[~modified_mask] *= fade_factor
        
        return result
    
    def quadrant_to_full(self, quadrant_source):
        top_half = np.concatenate([quadrant_source[:, ::-1], quadrant_source], axis=1)
        full_source = np.concatenate([top_half[::-1, :], top_half], axis=0)
        return full_source
    
    def augment_illumination(self, quadrant_illum_grid_size, numerical_aperture, 
                            wavelength_nm, **kwargs):
        max_spatial_frequency = numerical_aperture / wavelength_nm
        
        # Randomly select 1-6 families to combine
        n_families = np.random.randint(1, 7)
        
        family_generators = {
            'annular': self.generate_annular,
            'dipole': self.generate_dipole,
            'quadrupole': self.generate_quadrupole,
            'off_axis': self.generate_off_axis,
            'clustered': self.generate_clustered,
            'uniform_random': self.generate_uniform_random,
            'sparse_strong': self.generate_sparse_strong,
            'dense_weak': self.generate_dense_weak,
            'center_strong': self.generate_center_strong,
        }
        
        selected_families = random.sample(list(family_generators.keys()), n_families)
        
        # Dirichlet distribution ensures weights sum to 1
        weights = np.random.dirichlet(np.ones(n_families))
        
        # Generate each family component
        components = []
        for family_name in selected_families:
            generator = family_generators[family_name]
            component = generator(quadrant_illum_grid_size, max_spatial_frequency)
            components.append(component)
        
        # Weighted combination
        illum_quadrant = sum(w * comp for w, comp in zip(weights, components))
        
        # Apply morphological operations (like mask augmentation)
        if np.random.random() < 0.4:
            illum_quadrant = self.apply_morphological_ops(illum_quadrant)
        
        # Optional rotation BEFORE mirroring (preserves symmetry after mirroring)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-45, 45)
            illum_quadrant = rotate(illum_quadrant, angle=angle, reshape=False, order=1, mode='nearest')
        
        # Return quadrant (32x32) - will be mirrored to full only for saving
        return np.clip(illum_quadrant, 0, 1)


def save_dataset(mask_illumination_simtriplets, output_dir, train_split=0.8):
    output_dir = Path("./data/" + output_dir)
    splits = ['train', 'test']
    subdirs = ['masks', 'illuminations', 'intensities', 'resists']

    for split in splits:
        for subdir in subdirs:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    n_total = len(mask_illumination_simtriplets)
    n_train = int(n_total * train_split)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    split_indices = {
        'train': indices[:n_train],
        'test': indices[n_train:]
    }

    for split_name, split_idx in split_indices.items():
        split_dir = output_dir / split_name

        # Get num files already in masks to continue numbering
        existing_files = list((split_dir / 'masks').glob('*.png'))
        start_id = len(existing_files)

        for idx, data_idx in enumerate(split_idx):
            mask, illumination, sim_results = mask_illumination_simtriplets[data_idx]

            file_id = start_id + idx

            # Save input mask
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(split_dir / 'masks' / f"{file_id:06d}.png")

            # Save illumination
            illum_img = Image.fromarray((illumination * 255).astype(np.uint8))
            illum_img.save(split_dir / 'illuminations' / f"{file_id:06d}.png")

            # Save wafer intensity
            wafer_intensity = sim_results["wafer_intensity"]
            intensity_img = Image.fromarray((wafer_intensity * 255).astype(np.uint8))
            intensity_img.save(split_dir / 'intensities' / f"{file_id:06d}.png")

            # Save resist profile
            resist_profile = sim_results["resist_profile"]
            resist_img = Image.fromarray((resist_profile * 255).astype(np.uint8))
            resist_img.save(split_dir / 'resists' / f"{file_id:06d}.png")


def generate_n_augmentations(num_masks, num_illuminations, augmentations_per_mask, output_dir, sim_config):
    mask_augmenter = MaskAugmenter()
    light_source_augmenter = LightSourceAugmenter()

    # Load base masks from dataset
    base_masks = masks.get_dataset_masks('ganopc-data/artitgt', num_masks, **sim_config)
    
    # Augment each base mask
    augmented_masks = []
    for mask in base_masks:
        for _ in range(augmentations_per_mask):
            augmented_masks.append(mask_augmenter.random_augmentation(mask))

    # Generate illumination quadrants (32x32)
    illumination_quadrants = [light_source_augmenter.augment_illumination(
        quadrant_illum_grid_size=32,
        numerical_aperture=sim_config["numerical_aperture"],
        wavelength_nm=sim_config["wavelength_nm"]
    ) for _ in range(num_illuminations)]

    # Pair each mask with random illuminations
    mask_illumination_pairs = []
    for mask in augmented_masks:
        selected_quadrants = random.sample(illumination_quadrants, num_illuminations)
        for illum_quadrant in selected_quadrants:
            # Simulator gets quadrant (32x32)
            sim_results = simulator.LithographySimulator(sim_config).simulate(mask, illum_quadrant)
            
            # Save full (64x64) for visualization - will extract quadrant again when loading
            illum_full = light_source_augmenter.quadrant_to_full(illum_quadrant)
            
            mask_illumination_pairs.append((mask, illum_full, sim_results))

    save_dataset(mask_illumination_pairs, output_dir)

def main():
    with open("sim_config.json", "r") as f:
            sim_config = json.load(f)

    for i in tqdm(range(12500), desc="Generating batches"):
        generate_n_augmentations(num_masks=5, num_illuminations=1, 
                                augmentations_per_mask=1, output_dir='augmented_medium', 
                                sim_config=sim_config)

if __name__ == "__main__":
   
    main()
    # Main generation: 4:1 mask:illumination ratio
    # Each batch: 4 base masks × 1 augmentation × 1 illumination = 4 pairs
    # For 10k pairs total: 10000 / 4 = 2500 batches
    # for i in tqdm(range(2500), desc="Generating batches"):
    #     generate_n_augmentations(
    #         num_masks=4,
    #         num_illuminations=1, 
    #         augmentations_per_mask=1, 
    #         output_dir='augmented_massive', 
    #         sim_config=sim_config
    #     )