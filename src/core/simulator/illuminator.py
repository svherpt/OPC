import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import json
import os


def create_quadrant_source(config):
    quadrant_illum_grid_size = config.get("quadrant_illum_grid_size", 64)
    numerical_aperture = config.get("numerical_aperture", 1.35)
    wavelength_nm = config.get("wavelength_nm", 193)
    illumination_type = config.get("illumination_type", "conventional")
    
    max_spatial_frequency = numerical_aperture / wavelength_nm
    spatial_frequency_x, spatial_frequency_y = np.meshgrid(
        np.linspace(0, max_spatial_frequency, quadrant_illum_grid_size),
        np.linspace(0, max_spatial_frequency, quadrant_illum_grid_size)
    )
    source_illumination = np.zeros_like(spatial_frequency_x)

    if illumination_type == "conventional":
        spot_mask = (spatial_frequency_x**2 + spatial_frequency_y**2) < (0.05 * max_spatial_frequency)**2
        source_illumination[spot_mask] = 1.0
    elif illumination_type == "dipole_x":
        spot_distance = 0.6 * max_spatial_frequency
        spot_sigma = 0.05 * max_spatial_frequency
        source_illumination += np.exp(-((spatial_frequency_x - spot_distance)**2 + spatial_frequency_y**2) / (2 * spot_sigma**2))
    elif illumination_type == "quadruple":
        spot_distance = 0.6 * max_spatial_frequency
        spot_sigma = 0.05 * max_spatial_frequency
        source_illumination += np.exp(-((spatial_frequency_x - spot_distance)**2 + spatial_frequency_y**2) / (2 * spot_sigma**2))
        source_illumination += np.exp(-((spatial_frequency_x)**2 + (spatial_frequency_y - spot_distance)**2) / (2 * spot_sigma**2))
    elif illumination_type == "annular":
        inner_radius = 0.4 * max_spatial_frequency
        outer_radius = 0.8 * max_spatial_frequency
        radial_distance = np.sqrt(spatial_frequency_x**2 + spatial_frequency_y**2)
        annular_mask = (radial_distance > inner_radius) & (radial_distance < outer_radius)
        source_illumination[annular_mask] = 1.0 
    else:
        raise ValueError(f"Unknown illumination_type: {illumination_type}")

    return source_illumination


def quadrant_to_full(quadrant_illumination):
    top_half = np.concatenate([quadrant_illumination[:, ::-1], quadrant_illumination], axis=1)
    full_pupil = np.concatenate([top_half[::-1, :], top_half], axis=0)
    return full_pupil


def read_illumination_quarter_from_file(file_path, **kwargs):
    # illumination_size = kwargs.get("quadrant_illum_grid_size", 32)
    illumination = plt.imread('./data/' + file_path)

    #Return just a single quadrant
    illumination = illumination[:illumination.shape[0]//2, :illumination.shape[1]//2]

    return illumination


def read_random_illumination_quarter(dir_path="example_masks", **kwargs):
    all_files = [f for f in os.listdir('./data/' + dir_path) if f.endswith('.png')]
    random_file = np.random.choice(all_files)

    return read_illumination_quarter_from_file(dir_path + "/" + random_file, **kwargs)


def get_full_illumination(config):
    quadrant_source = create_quadrant_source(config)
    full_illumination = quadrant_to_full(quadrant_source)
    return full_illumination


def upsample_illumination(lowres_illumination, target_size):
    current_size = lowres_illumination.shape[0]
    zoom_factor = target_size / current_size
    upsampled = zoom(lowres_illumination, zoom=zoom_factor, order=3)
    return upsampled


def visualise_illumination(illumination):
    plt.imshow(illumination, cmap='gray')
    plt.title("Generated Illumination")
    plt.show()


if __name__ == "__main__":
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)

    lowres_source_illumination = get_full_illumination(sim_config)
    true_highres_source = get_full_illumination({**sim_config, "quadrant_illum_grid_size": 256})
    upsampled_illumination = upsample_illumination(lowres_source_illumination, target_size=true_highres_source.shape[0])

    visualise_illumination(lowres_source_illumination)
    visualise_illumination(upsampled_illumination)
    