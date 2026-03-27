import torch 
import numpy as np
import src.core.simulator.masks as masks
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
import src.core.simulator.illuminator as illuminator
import src.core.misc as misc
from scipy.special import expit
from scipy.ndimage import gaussian_filter


class LithographySimulator:
    """Simulates the lithography process given a mask and a quadrant of the source illumination."""
    def __init__(self, config, chunk_size=128):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size

        self.cache_frequency_grids()

        #print(f"LithographySimulator initialized on device: {self.device}")


    def cache_frequency_grids(self):
        """Pre-cache frequency grids that are reused across simulations."""
        mask_size = self.config.get("mask_grid_size", 512)
        mask_width_nm = self.config.get("mask_width_nm", 1000)
        dx = mask_width_nm / mask_size
        freq = torch.fft.fftshift(torch.fft.fftfreq(mask_size, d=dx)).to(self.device)
        self.freq_x, self.freq_y = torch.meshgrid(freq, freq, indexing='ij')
        
        # Cache pupil parameters
        self.wavelength_nm = self.config.get("wavelength_nm", 193)
        self.numerical_aperture = self.config.get("numerical_aperture", 1.35)
        self.cutoff_frequency = self.numerical_aperture / self.wavelength_nm
        self.defocus_nm = self.config.get("defocus_nm", 0.0)
        self.pupil_eps = self.config.get("pupil_eps", 1e-3)


    def simulate(self, mask, source_illum_quadrant):
        """Performs the lithography simulation using the Fourier optics model."""
        # Data preparation
        mask = torch.from_numpy(mask).to(self.device, dtype=torch.float32)
        source_illumination = illuminator.quadrant_to_full(source_illum_quadrant)
        source_illumination = torch.from_numpy(source_illumination).to(self.device, dtype=torch.float32)
        
        # Compute frequency grids
        pupil_freq_x_grid, pupil_freq_y_grid = self.compute_pupil_frequency_grids(source_illumination.shape[0])
        mask_ft = torch.fft.fftshift(torch.fft.fft2(mask))
        
        # Core simulation: propagate through pupil
        total_filtered_ft, total_intensity = self.propagate_through_pupil(mask_ft, source_illumination, pupil_freq_x_grid, pupil_freq_y_grid)
        
        # Apply physical effects
        wafer_intensity_clipped = self.apply_flare_and_normalize(total_intensity)
        
        # Compute resist response
        wafer_intensity_np = wafer_intensity_clipped.cpu().numpy()
        resist_profile = self.get_resist_profile(wafer_intensity_np)
        
        return {
            "mask_ft": mask_ft.cpu().numpy(),
            "filtered_ft": total_filtered_ft.cpu().numpy(),
            "wafer_intensity": wafer_intensity_clipped.cpu().numpy(),
            "resist_profile": resist_profile
        }
    

    def compute_pupil_frequency_grids(self, pupil_size):
        """Computes the pupil frequency grids based on NA and wavelength."""
        NA = self.config.get("numerical_aperture", 1.35)
        wavelength_nm = self.config.get("wavelength_nm", 193)
        max_spatial_frequency = NA / wavelength_nm
        pupil_spatial_frequencies_1d = torch.linspace(-max_spatial_frequency, max_spatial_frequency, pupil_size, device=self.device)
        return torch.meshgrid(pupil_spatial_frequencies_1d, pupil_spatial_frequencies_1d, indexing='ij')


    def propagate_through_pupil(self, mask_ft, source_illumination, pupil_freq_x_grid, pupil_freq_y_grid):
        """Propagates the mask through all pupil points, accumulating intensity and field."""
        total_filtered_ft = torch.zeros_like(mask_ft, dtype=torch.complex64, device=self.device)
        total_intensity = torch.zeros(mask_ft.shape, dtype=torch.float32, device=self.device)
        
        # Get non-zero pupil points
        threshold = 1e-6
        pupil_indices = torch.nonzero(source_illumination > threshold, as_tuple=False)
        weights = source_illumination[pupil_indices[:,0], pupil_indices[:,1]]
        
        num_points = len(weights)
        for start in range(0, num_points, self.chunk_size):
            end = min(start + self.chunk_size, num_points)
            chunk_indices = pupil_indices[start:end]
            chunk_weights = weights[start:end].to(self.device, dtype=torch.float32)
            
            # Compute filtered fields for this chunk
            filters = torch.stack([
                self.get_pupil_filter(self.freq_x - pupil_freq_x_grid[i, j], self.freq_y - pupil_freq_y_grid[i, j])
                for i,j in chunk_indices.tolist()
            ])
            
            filtered_fts_chunk = mask_ft[None, :, :] * filters
            weighted_fts = filtered_fts_chunk * chunk_weights[:,None,None]
            total_filtered_ft += torch.sum(weighted_fts, dim=0)
            
            # Accumulate intensity
            wafer_fields = torch.fft.ifft2(torch.fft.ifftshift(filtered_fts_chunk, dim=(-2,-1)), dim=(-2,-1))
            total_intensity += torch.sum(chunk_weights[:,None,None] * torch.abs(wafer_fields)**2, dim=0)
        
        # Normalize by total source power
        total_weight = torch.sum(weights)
        total_filtered_ft /= total_weight
        total_intensity /= total_weight
        
        return total_filtered_ft, total_intensity


    def apply_flare_and_normalize(self, total_intensity):
        """Applies flare effect and normalizes intensity to [0, 1] range."""
        flare_fraction = self.config.get("flare_fraction", 0.0)
        max_intensity = self.config.get("max_intensity", 4.0)
        
        mean_intensity = torch.mean(total_intensity)
        wafer_intensity = (1.0 - flare_fraction) * total_intensity + flare_fraction * mean_intensity
        wafer_intensity_clipped = torch.clamp(wafer_intensity / max_intensity, 0.0, 1.0)
        
        return wafer_intensity_clipped


    def get_resist_profile(self, intensity):
        """Converts normalized intensity to resist profile using a sigmoid function."""
        threshold = self.config.get("resist_threshold", 0.5)
        sigma = self.config.get("resist_blur_sigma", 1.0)
        eps = self.config.get("resist_eps", 1e-3)

        blurred = gaussian_filter(intensity, sigma=sigma)

        max_blurred = np.max(blurred)
        if max_blurred == 0:
            max_blurred = 1.0

        normalized = blurred / max_blurred

        # Use the expit function for numerical stability for the formula: resist_profile = 1 / (1 + np.exp(-(normalized_intensity - threshold)/eps))
        resist_profile = expit((normalized - threshold) / eps)
        return resist_profile


    def get_pupil_filter(self, freq_x, freq_y):
        """Computes the pupil filter with aperture cutoff and defocus phase."""
        freq_radial = torch.sqrt(freq_x**2 + freq_y**2)
        pupil = 1 / (1 + torch.exp((freq_radial - self.cutoff_frequency) / self.pupil_eps))

        rho = freq_radial / self.cutoff_frequency
        phase = self.defocus_nm / self.wavelength_nm * 2 * np.pi * (2 * rho**2 - 1)

        return pupil * torch.exp(1j * phase)


if __name__ == "__main__":
    sim_config = misc.get_simulation_config()

    # Random mask and illumination example
    random_mask = masks.get_random_dataset_mask(**sim_config)
    source_illumination = illuminator.create_quadrant_source(sim_config)

    simulator = LithographySimulator(sim_config)
    out = simulator.simulate(random_mask, source_illumination)

    simulation_visualizer.visualize_simulation_results(out, mask=random_mask, illumination=source_illumination, config=sim_config)