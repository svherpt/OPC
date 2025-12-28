import numpy as np
from scipy.ndimage import gaussian_filter
import src.simulator.light_sources as light_sources

class LithographySimulator:
    def __init__(self, config):
        self.config = config

    def simulate(self, mask):
        flare_fraction = self.config.get("flare_fraction", 0.0)

        return_obj = {}

        mask_x_nm, mask_y_nm = self.get_initial_grid()
        grid_size = mask_x_nm[0,1] - mask_x_nm[0,0]

        freq = np.fft.fftshift(np.fft.fftfreq(mask_x_nm.shape[0], d=grid_size))
        freq_x, freq_y = np.meshgrid(freq, freq)

        source_points = light_sources.get_source_points(self.config)

        total_intensity = np.zeros_like(mask, dtype=np.float64)
        mask_ft = np.fft.fftshift(np.fft.fft2(mask))

        total_filtered_ft = np.zeros_like(mask, dtype=np.complex128)

        for index, (source_x, source_y, source_intensity) in enumerate(source_points):
            pupil_filter = self.get_pupil_filter(freq_x - source_x, freq_y - source_y)
            filtered_ft = mask_ft * pupil_filter

            total_filtered_ft += filtered_ft * source_intensity

            wafer_field = np.fft.ifft2(np.fft.ifftshift(filtered_ft))
            total_intensity += np.abs(wafer_field)**2

            # Store wafer field for central source for phase analysis
            if index == len(source_points) // 2:
                return_obj["wafer_field"] = wafer_field

        #Normalise for number of sources
        total_weight = sum(weight for _, _, weight in source_points)
        total_intensity /= total_weight
        total_filtered_ft /= total_weight

        #Apply flare
        mean_intensity = np.mean(total_intensity)
        wafer_intensity = (1.0 - flare_fraction) * total_intensity + flare_fraction * mean_intensity
        resist_profile = self.get_resist_profile(wafer_intensity)

        #Return all relevant data so it can be visualised later
        return_obj["mask_ft"] = mask_ft
        return_obj["filtered_ft"] = total_filtered_ft
        return_obj["wafer_intensity"] = wafer_intensity
        return_obj["resist_profile"] = resist_profile
        return return_obj

    def get_resist_profile(self, intensity):
        threshold = self.config.get("resist_threshold", 0.5)
        sigma = self.config.get("resist_blur_sigma", 1.0)
        eps = self.config.get("resist_eps", 1e-3)

        #Perform Gaussian blur to simulate resist response
        blurred_intensity = gaussian_filter(intensity, sigma=sigma)

        max_blurred_intensity = np.max(blurred_intensity)
        if max_blurred_intensity == 0:
            max_blurred_intensity = 1.0

        normalized_intensity = blurred_intensity / max_blurred_intensity

        #Sigmoid thresholding for differentiability
        resist_profile = 1 / (1 + np.exp(-(normalized_intensity - threshold)/eps))


        return resist_profile


    def get_pupil_filter(self, freq_x, freq_y):
        wavelength_nm = self.config.get("wavelength_nm", 193)
        numerical_aperture = self.config.get("numerical_aperture", 1.35)
        defocus_nm = self.config.get("defocus_nm", 0.0)
        eps = self.config.get("pupil_eps", 1e-3)

        cutoff_frequency = numerical_aperture / wavelength_nm

        freq_radial = np.sqrt(freq_x**2 + freq_y**2)


        #Sigmoid function so it is differentiable
        pupil = 1 / (1 + np.exp((freq_radial - cutoff_frequency)/eps))

        #Defocus abberation
        rho = freq_radial / cutoff_frequency
        phase = defocus_nm / wavelength_nm * 2 * np.pi * (2 * rho**2 - 1)

        phase = phase * pupil
        return pupil * np.exp(1j * phase)

    def get_initial_grid(self):
        mask_grid_size = self.config.get("mask_grid_size", 512)
        mask_width_nm = self.config.get("mask_width_nm", 1000)

        x = np.linspace(-mask_width_nm/2, mask_width_nm/2, mask_grid_size)
        y = np.linspace(-mask_width_nm/2, mask_width_nm/2, mask_grid_size)
        
        return np.meshgrid(x, y)