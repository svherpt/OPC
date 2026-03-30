import numpy as np
import random
from scipy.ndimage import rotate


class IlluminationAugmenter:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Random params are bound in the registry, base functions take explicit params
        # r_range is expressed as a fraction of max_spatial_frequency (0.0 to 1.0)
        self._sampler_registry = [
            (1, lambda max_spatial_freq: self._sample_uniform(max_spatial_freq)),
            (1, lambda max_spatial_freq: self._sample_radial(max_spatial_freq, r_range=(0.4, 0.8))),  # annular
            (1, lambda max_spatial_freq: self._sample_radial(max_spatial_freq, r_range=(0.7, 1.0))),  # off-axis
            (1, lambda max_spatial_freq: self._sample_radial(max_spatial_freq, r_range=(0.0, 0.3))),  # center
            (1, lambda max_spatial_freq: self._sample_clustered(max_spatial_freq, center_range=(0.2, 0.8), spread=0.2)),
        ]
        self._weighted_samplers = [fn for w, fn in self._sampler_registry for _ in range(w)]

    def _sample_uniform(self, max_spatial_freq):
        # Returns a random (kx, ky) position anywhere in frequency space
        kx = np.random.uniform(0, max_spatial_freq)
        ky = np.random.uniform(0, max_spatial_freq)
        return kx, ky

    def _sample_radial(self, max_spatial_freq, r_range):
        # Returns a (kx, ky) position at a given radial distance from the origin
        # r_range controls how far from center (0=DC, 1=edge of aperture)
        radius = np.random.uniform(*r_range) * max_spatial_freq
        angle = np.random.uniform(0, 2 * np.pi)
        return radius * np.cos(angle), radius * np.sin(angle)

    def _sample_clustered(self, max_spatial_freq, center_range, spread):
        # Returns a (kx, ky) position near a randomly chosen cluster center
        # spread controls how tightly modes are grouped (as fraction of max_spatial_freq)
        cx = np.random.uniform(*center_range) * max_spatial_freq
        cy = np.random.uniform(*center_range) * max_spatial_freq
        kx = cx + np.random.uniform(-spread, spread) * max_spatial_freq
        ky = cy + np.random.uniform(-spread, spread) * max_spatial_freq
        return kx, ky

    def _loguniform(self, low, high):
        return 10 ** np.random.uniform(np.log10(low), np.log10(high))

    def _generate(self, quadrant_illum_grid_size, max_spatial_freq, sampler, n_modes, sigma_range=(0.005, 0.2), intensity_range=(0.05, 5.0)):
        # Build a frequency-space grid from 0 to max_spatial_freq
        kx_axis = np.linspace(0, max_spatial_freq, quadrant_illum_grid_size)
        KX, KY = np.meshgrid(kx_axis, kx_axis)
        illum = np.zeros((quadrant_illum_grid_size, quadrant_illum_grid_size))

        for _ in range(n_modes):
            # Sample a mode center (kx, ky) in frequency space
            kx, ky = sampler(max_spatial_freq)
            sigma = self._loguniform(*sigma_range) * max_spatial_freq
            intensity = self._loguniform(*intensity_range)
            illum += intensity * np.exp(-((KX - kx)**2 + (KY - ky)**2) / (2 * sigma**2))

        return illum

    def augment_illumination(self, quadrant_illum_grid_size, numerical_aperture, wavelength_nm, **kwargs):
        """Generate a random augmented illumination quadrant, normalised so intensities sum to 1."""
        max_spatial_freq = numerical_aperture / wavelength_nm
        samplers  = random.sample(self._weighted_samplers, k=np.random.randint(1, 4))
        weights   = np.random.dirichlet(np.ones(len(samplers)))
        components = [
            self._generate(quadrant_illum_grid_size, max_spatial_freq, sampler, n_modes=np.random.randint(1, 15))
            for sampler in samplers
        ]
        normalized = [c / c.max() if c.max() > 0 else c for c in components]
        illum      = sum(w * c for w, c in zip(weights, normalized))
        if np.random.random() < 0.3:
            illum = rotate(illum, angle=np.random.uniform(-45, 45), reshape=False, order=1, mode='nearest')

        #Clip and normalise to sum=1 for consistent exposure     
        illum = np.clip(illum, 0, 1)
        illum /= (illum.sum() + 1e-8)
        return illum
    
    def get_batch(self, batch_size, sim_config):
        """Generate a batch of random augmented illumination quadrants."""
        return np.stack([
            self.augment_illumination(**sim_config).astype(np.float32)
            for _ in range(batch_size)
        ])