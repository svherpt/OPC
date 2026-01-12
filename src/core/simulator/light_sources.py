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
        source_illumination += np.exp(
            -((spatial_frequency_x - spot_distance)**2 + spatial_frequency_y**2) / (2 * spot_sigma**2)
        )
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

def read_random_illumination_quarter(dir_path="ganopc-data/artitgt", **kwargs):
    all_files = [f for f in os.listdir('./data/' + dir_path) if f.endswith('.png')]
    random_file = np.random.choice(all_files)

    return read_illumination_quarter_from_file(dir_path + "/" + random_file, **kwargs)

def visualize_pupil(lowres_illumination, target_size=256, highres_illumination=None):
    """
    Show low-res, upsampled, and optionally true high-res illumination side by side.
    """
    # Upsample low-res to target size
    zoom_factor = target_size / lowres_illumination.shape[0]
    upsampled_illumination = zoom(lowres_illumination, zoom=zoom_factor, order=3)

    # If no true high-res provided, just use upsampled as placeholder
    if highres_illumination is None:
        highres_illumination = upsampled_illumination

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Low-res
    im0 = axes[0].imshow(lowres_illumination, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    axes[0].set_title(f"Low-res ({lowres_illumination.shape[0]}x{lowres_illumination.shape[1]})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Relative intensity")

    # Upsampled
    im1 = axes[1].imshow(upsampled_illumination, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    axes[1].set_title(f"Upsampled ({target_size}x{target_size})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Relative intensity")

    # True high-res
    im2 = axes[2].imshow(highres_illumination, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    axes[2].set_title(f"High-res ({highres_illumination.shape[0]}x{highres_illumination.shape[1]})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Relative intensity")

    plt.tight_layout()
    plt.show()



def get_full_illumination(config):
    quadrant_source = create_quadrant_source(config)
    full_illumination = quadrant_to_full(quadrant_source)
    return full_illumination


def upsample_illumination(lowres_illumination, target_size):
    current_size = lowres_illumination.shape[0]
    zoom_factor = target_size / current_size
    upsampled = zoom(lowres_illumination, zoom=zoom_factor, order=3)
    return upsampled


if __name__ == "__main__":
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)

    lowres_source_illumination = get_full_illumination(sim_config)
    true_highres_source = get_full_illumination({**sim_config, "quadrant_illum_grid_size": 256})  # or already generated
    visualize_pupil(lowres_source_illumination, highres_illumination=true_highres_source)