import numpy as np
import matplotlib.pyplot as plt
import src.core.misc as misc
from scipy.ndimage import zoom
import os


def create_quadrant_source(config):
    """Creates a quadrant illumination pattern based on the specified configuration."""
    quadrant_illum_grid_size = config.get("quadrant_illum_grid_size", 64)
    numerical_aperture = config.get("numerical_aperture", 1.35)
    wavelength_nm = config.get("wavelength_nm", 193)
    illumination_type = config.get("illumination_type", "conventional")

    # Function mapper for illumination types
    ILLUMINATION_TYPES = {
        "conventional": conventional_illumination,
        "dipole_x": dipole_x_illumination,
        "quadruple": quadruple_illumination,
        "annular": annular_illumination,
    }

    if illumination_type not in ILLUMINATION_TYPES:
        raise ValueError(f"Unknown illumination_type: {illumination_type}")
    
    max_spatial_frequency = numerical_aperture / wavelength_nm
    spatial_frequency_x, spatial_frequency_y = np.meshgrid(
        np.linspace(0, max_spatial_frequency, quadrant_illum_grid_size),
        np.linspace(0, max_spatial_frequency, quadrant_illum_grid_size)
    )
    
    illumination_func = ILLUMINATION_TYPES[illumination_type]
    source_illumination = illumination_func(spatial_frequency_x, spatial_frequency_y, max_spatial_frequency)
    
    return source_illumination


def conventional_illumination(spatial_frequency_x, spatial_frequency_y, max_spatial_frequency):
    """Conventional circular illumination pattern."""
    source_illumination = np.zeros_like(spatial_frequency_x)
    spot_mask = (spatial_frequency_x**2 + spatial_frequency_y**2) < (0.05 * max_spatial_frequency)**2
    source_illumination[spot_mask] = 1.0
    return source_illumination


def dipole_x_illumination(spatial_frequency_x, spatial_frequency_y, max_spatial_frequency):
    """Dipole X-oriented illumination pattern."""
    source_illumination = np.zeros_like(spatial_frequency_x)
    spot_distance = 0.6 * max_spatial_frequency
    spot_sigma = 0.05 * max_spatial_frequency
    source_illumination += np.exp(-((spatial_frequency_x - spot_distance)**2 + spatial_frequency_y**2) / (2 * spot_sigma**2))
    return source_illumination


def quadruple_illumination(spatial_frequency_x, spatial_frequency_y, max_spatial_frequency):
    """Quadruple illumination pattern."""
    source_illumination = np.zeros_like(spatial_frequency_x)
    spot_distance = 0.6 * max_spatial_frequency
    spot_sigma = 0.05 * max_spatial_frequency
    source_illumination += np.exp(-((spatial_frequency_x - spot_distance)**2 + spatial_frequency_y**2) / (2 * spot_sigma**2))
    source_illumination += np.exp(-((spatial_frequency_x)**2 + (spatial_frequency_y - spot_distance)**2) / (2 * spot_sigma**2))
    return source_illumination


def annular_illumination(spatial_frequency_x, spatial_frequency_y, max_spatial_frequency):
    """Annular illumination pattern."""
    source_illumination = np.zeros_like(spatial_frequency_x)
    inner_radius = 0.4 * max_spatial_frequency
    outer_radius = 0.8 * max_spatial_frequency
    radial_distance = np.sqrt(spatial_frequency_x**2 + spatial_frequency_y**2)
    annular_mask = (radial_distance > inner_radius) & (radial_distance < outer_radius)
    source_illumination[annular_mask] = 1.0
    return source_illumination


def quadrant_to_full(quadrant_illumination):
    """Converts a quadrant illumination pattern to a full illumination pattern by mirroring across both axes."""
    top_half = np.concatenate([quadrant_illumination[:, ::-1], quadrant_illumination], axis=1)
    full_pupil = np.concatenate([top_half[::-1, :], top_half], axis=0)
    return full_pupil


def read_illumination_quarter_from_file(file_path, **kwargs):
    """Reads a quadrant illumination pattern from an image file."""
    illumination = plt.imread(os.path.join('./data/', file_path))

    #Return just a single quadrant
    illumination = illumination[:illumination.shape[0]//2, :illumination.shape[1]//2]

    return illumination


def read_random_illumination_quarter(dir_path="example_masks", **kwargs):
    """Reads a random quadrant illumination pattern from an image file in the specified directory."""
    all_files = [f for f in os.listdir(os.path.join('./data/', dir_path)) if f.endswith('.png')]
    random_file = np.random.choice(all_files)

    return read_illumination_quarter_from_file(dir_path + "/" + random_file, **kwargs)


def get_full_illumination(config):
    """Generates a full illumination pattern by creating a quadrant source and converting it to full."""
    quadrant_source = create_quadrant_source(config)
    full_illumination = quadrant_to_full(quadrant_source)
    return full_illumination


def upsample_illumination(lowres_illumination, target_size):
    """Upsamples a low-resolution illumination pattern to the target size using cubic interpolation."""
    current_size = lowres_illumination.shape[0]
    zoom_factor = target_size / current_size
    upsampled = zoom(lowres_illumination, zoom=zoom_factor, order=3)
    return upsampled


def visualise_illumination(illumination):
    """Visualizes the given illumination pattern."""
    plt.imshow(illumination, cmap='hot')
    plt.title("Generated Illumination")
    plt.show()


#Show an example illumination as well as upscaled
if __name__ == "__main__":
    sim_config = misc.get_simulation_config()

    lowres_source_illumination = get_full_illumination(sim_config)
    true_highres_source = get_full_illumination({**sim_config, "quadrant_illum_grid_size": 256})
    upsampled_illumination = upsample_illumination(lowres_source_illumination, target_size=true_highres_source.shape[0])

    visualise_illumination(lowres_source_illumination)
    visualise_illumination(upsampled_illumination)
    