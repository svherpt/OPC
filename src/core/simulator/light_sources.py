import numpy as np
import matplotlib.pyplot as plt

def create_quadrant_source(grid_size, NA, wavelength_nm, illumination_type):
    """
    Create a single quadrant pupil intensity array.
    """
    k_max = NA / wavelength_nm
    k = np.linspace(0, k_max, grid_size)
    kx, ky = np.meshgrid(k, k)
    source_illumination = np.zeros_like(kx)

    if illumination_type == "conventional":
        # small spot near origin
        mask = (kx**2 + ky**2) < (0.05*k_max)**2
        source_illumination[mask] = 1.0

    elif illumination_type == "dipole_x":
        r = 0.6 * k_max
        sigma = 0.05 * k_max
        source_illumination += np.exp(-((kx - r)**2 + ky**2)/(2*sigma**2))

    else:
        raise ValueError(f"Unknown illumination_type: {illumination_type}")

    return source_illumination, kx, ky


def quadrant_to_full(source_illumination_quadrant):
    """
    Mirror a quadrant to full 4-fold symmetric pupil.
    """
    top = np.concatenate([source_illumination_quadrant[:, ::-1], source_illumination_quadrant], axis=1)  # left + right
    full = np.concatenate([top[::-1, :], top], axis=0)  # bottom + top
    return full


def visualize_pupil(source_illumination_full):
    plt.imshow(source_illumination_full, extent=(-1,1,-1,1), origin='lower')
    plt.title("Pupil intensity")
    plt.colorbar(label="Relative intensity")
    plt.show()


def get_source_grid(config):
    """
    Returns the full pupil intensity array and optional coordinates.
    """
    grid_size = config.get("pupil_grid_size", 33)
    NA = config.get("numerical_aperture", 1.35)
    wavelength_nm = config.get("wavelength_nm", 193)
    illumination_type = config.get("illumination_type", "conventional")

    source_illumination_quadrant, kx_q, ky_q = create_quadrant_source(grid_size, NA, wavelength_nm, illumination_type)
    source_illumination_full = quadrant_to_full(source_illumination_quadrant)

    return source_illumination_full


if __name__ == "__main__":
    config = {
        "pupil_grid_size": 65,
        "numerical_aperture": 1.35,
        "wavelength_nm": 193,
        "illumination_type": "dipole_x"
    }
    source_illumination_full = get_source_grid(config)
    visualize_pupil(source_illumination_full)