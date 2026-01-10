import torch 
import numpy as np
import matplotlib.pyplot as plt
import src.core.simulator.masks as masks
import json
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
import src.core.simulator.light_sources as light_sources
from scipy.special import expit
from scipy.ndimage import gaussian_filter


class LithographySimulator:
    def __init__(self, config, chunk_size=128):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size

        print(f"LithographySimulatorGPU initialized on device: {self.device}")

    def simulate(self, mask, source_illumination):
        mask = torch.from_numpy(mask).to(self.device, dtype=torch.float32)
        source_illumination = torch.from_numpy(source_illumination).to(self.device, dtype=torch.float32)
        mask_size = mask.shape[0]
        pupil_size = source_illumination.shape[0]

        flare_fraction = self.config.get("flare_fraction", 0.0)
        max_intensity = self.config.get("max_intensity", 4.0)

        mask_x, mask_y = self.get_initial_grid()
        dx = mask_x[0,1] - mask_x[0,0]
        freq = torch.fft.fftshift(torch.fft.fftfreq(mask_size, d=dx)).to(self.device)
        freq_x, freq_y = torch.meshgrid(freq, freq, indexing='ij')

        NA = self.config.get("numerical_aperture", 1.35)
        wavelength_nm = self.config.get("wavelength_nm", 193)
        max_spatial_frequency = NA / wavelength_nm
        pupil_spatial_frequencies_1d = torch.linspace(-max_spatial_frequency, max_spatial_frequency, pupil_size, device=self.device)
        pupil_freq_x_grid, pupil_freq_y_grid = torch.meshgrid(pupil_spatial_frequencies_1d, pupil_spatial_frequencies_1d, indexing='ij')

        # FFT of mask
        mask_ft = torch.fft.fftshift(torch.fft.fft2(mask))

        total_filtered_ft = torch.zeros_like(mask_ft, dtype=torch.complex64, device=self.device)
        total_intensity = torch.zeros_like(mask, dtype=torch.float32, device=self.device)

        # Get non-zero pupil points
        pupil_indices = torch.nonzero(source_illumination, as_tuple=False)
        weights = source_illumination[pupil_indices[:,0], pupil_indices[:,1]]

        num_points = len(weights)
        for start in range(0, num_points, self.chunk_size):
            end = min(start + self.chunk_size, num_points)
            chunk_indices = pupil_indices[start:end]
            chunk_weights = weights[start:end].to(self.device, dtype=torch.float32)

            # Create filters for this chunk
            filters = torch.stack([
                self.get_pupil_filter(freq_x - pupil_freq_x_grid[i,j], freq_y - pupil_freq_y_grid [i,j])
                for i,j in chunk_indices
            ])  # shape: (chunk_size, mask_size, mask_size)

            # Multiply mask FFT by filters
            filtered_fts_chunk = mask_ft[None, :, :] * filters

            # Sum weighted filtered FT
            weighted_fts = filtered_fts_chunk * chunk_weights[:,None,None]
            total_filtered_ft += torch.sum(weighted_fts, dim=0)

            # Compute wafer fields and intensities
            wafer_fields = torch.fft.ifft2(torch.fft.ifftshift(filtered_fts_chunk, dim=(-2,-1)), dim=(-2,-1))
            total_intensity += torch.sum(chunk_weights[:,None,None] * torch.abs(wafer_fields)**2, dim=0)

        # Normalize by total source power
        total_weight = torch.sum(weights)
        total_filtered_ft /= total_weight
        total_intensity /= total_weight

        # Apply flare
        mean_intensity = torch.mean(total_intensity)
        wafer_intensity = (1.0 - flare_fraction) * total_intensity + flare_fraction * mean_intensity
        wafer_intensity_clipped = torch.clamp(wafer_intensity / max_intensity, 0.0, 1.0)

        # Compute resist profile on CPU (identical to CPU code)
        wafer_intensity_np = wafer_intensity_clipped.cpu().numpy()
        resist_profile = self.get_resist_profile(wafer_intensity_np)

        return {
            "mask_ft": mask_ft.cpu().numpy(),
            "filtered_ft": total_filtered_ft.cpu().numpy(),
            "wafer_intensity": wafer_intensity_clipped.cpu().numpy(),
            "resist_profile": resist_profile
        }

    def get_resist_profile(self, intensity):
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
        wavelength_nm = self.config.get("wavelength_nm", 193)
        numerical_aperture = self.config.get("numerical_aperture", 1.35)
        defocus_nm = self.config.get("defocus_nm", 0.0)
        eps = self.config.get("pupil_eps", 1e-3)

        cutoff_frequency = numerical_aperture / wavelength_nm
        freq_radial = torch.sqrt(freq_x**2 + freq_y**2)
        pupil = 1 / (1 + torch.exp((freq_radial - cutoff_frequency)/eps))

        rho = freq_radial / cutoff_frequency
        phase = defocus_nm / wavelength_nm * 2 * np.pi * (2 * rho**2 - 1)

        return pupil * torch.exp(1j * phase)

    def get_initial_grid(self):
        mask_grid_size = self.config.get("mask_grid_size", 512)
        mask_width_nm = self.config.get("mask_width_nm", 1000)

        x = np.linspace(-mask_width_nm/2, mask_width_nm/2, mask_grid_size)
        y = np.linspace(-mask_width_nm/2, mask_width_nm/2, mask_grid_size)
        return np.meshgrid(x, y)

if __name__ == "__main__":

    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)

    # Random mask example
    random_mask = masks.read_mask_from_img('ganopc-data/artitgt/1.glp.png', **sim_config)
    illumination = light_sources.get_source_grid(sim_config)
    masks.visualise_mask(random_mask)
    simulator = LithographySimulatorGPU(sim_config, device='cuda')
    out = simulator.simulate(random_mask, illumination)

    simulation_visualizer.visualize_simulation_results(out, mask=random_mask, illumination=illumination, config=sim_config)