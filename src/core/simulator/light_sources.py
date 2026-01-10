import numpy as np
import matplotlib.pyplot as plt

def create_quadrant_source(quadrant_grid_size, numerical_aperture, wavelength_nm, illumination_type):
    max_spatial_frequency = numerical_aperture / wavelength_nm
    k_values = np.linspace(0, max_spatial_frequency, quadrant_grid_size)
    kx_grid, ky_grid = np.meshgrid(k_values, k_values)
    source_illumination = np.zeros_like(kx_grid)

    if illumination_type == "conventional":
        # small spot near origin
        spot_mask = (kx_grid**2 + ky_grid**2) < (0.05 * max_spatial_frequency)**2
        source_illumination[spot_mask] = 1.0

    elif illumination_type == "dipole_x":
        # two spots along kx axis
        spot_distance = 0.6 * max_spatial_frequency
        spot_sigma = 0.05 * max_spatial_frequency
        source_illumination += np.exp(
            -((kx_grid - spot_distance)**2 + ky_grid**2) / (2 * spot_sigma**2)
        )

    else:
        raise ValueError(f"Unknown illumination_type: {illumination_type}")

    return source_illumination, kx_grid, ky_grid


def quadrant_to_full(quadrant_illumination):
    # Mirror quadrant to full 4-fold symmetric pupil
    top_half = np.concatenate([quadrant_illumination[:, ::-1], quadrant_illumination], axis=1)
    full_pupil = np.concatenate([top_half[::-1, :], top_half], axis=0)
    return full_pupil


def visualize_pupil(full_illumination):
    plt.imshow(full_illumination, extent=(-1, 1, -1, 1), origin='lower')
    plt.title("Pupil intensity")
    plt.colorbar(label="Relative intensity")
    plt.show()


def get_source_grid(config):
    illumination_grid_size = config.get("illumination_grid_size", 33)
    numerical_aperture = config.get("numerical_aperture", 1.35)
    wavelength_nm = config.get("wavelength_nm", 193)
    illumination_type = config.get("illumination_type", "conventional")

    quadrant_grid_size = illumination_grid_size // 2
    quadrant_source, kx_quadrant, ky_quadrant = create_quadrant_source(
        quadrant_grid_size, numerical_aperture, wavelength_nm, illumination_type
    )

    full_illumination = quadrant_to_full(quadrant_source)
    return full_illumination


if __name__ == "__main__":
    config = {
        "illumination_grid_size": 65,
        "numerical_aperture": 1.35,
        "wavelength_nm": 193,
        "illumination_type": "dipole_x"
    }

    full_source_illumination = get_source_grid(config)
    visualize_pupil(full_source_illumination)
