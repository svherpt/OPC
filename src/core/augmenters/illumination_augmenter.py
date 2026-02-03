
import numpy as np
import random
from scipy.ndimage import rotate, binary_dilation, binary_erosion
from skimage.morphology import disk

class IlluminationAugmenter:
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
        # Ring pattern
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