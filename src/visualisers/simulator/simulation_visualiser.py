import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import src.core.simulator.lithography_simulator as simulator
import src.core.simulator.masks as masks

# Visual configuration per simulation field.
FIELD_CONFIG = {
    "mask": {
        "cmap": "gray",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Transmission",
        "domain": "spatial",
        "log": False,
        "black_bg": False,
    },
    "mask_ft": {
        "cmap": "viridis",
        "xlabel": "Frequency X (1/nm)",
        "ylabel": "Frequency Y (1/nm)",
        "cbar": "Magnitude (log)",
        "domain": "frequency",
        "log": True,
        "black_bg": False,
    },
    "filtered_ft": {
        "cmap": "viridis",
        "xlabel": "Frequency X (1/nm)",
        "ylabel": "Frequency Y (1/nm)",
        "cbar": "Magnitude (log)",
        "domain": "frequency",
        "log": True,
        "black_bg": True,
    },
    "wafer_field": {
        "cmap": "plasma",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Field magnitude",
        "domain": "spatial",
        "log": False,
        "black_bg": False,
    },
    "wafer_intensity": {
        "cmap": "hot",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Intensity",
        "domain": "spatial",
        "log": False,
        "black_bg": False,
    },
    "resist_profile": {
        "cmap": "gray",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": " Unexposed (0) / Exposed (1)",
        "domain": "spatial",
        "log": False,
        "black_bg": False,
    },
}


def _plot_image(ax, data, cfg, title, extent):
    if cfg["log"]:
        valid = data[np.isfinite(data) & (data > 0)]
        norm = LogNorm(vmin=valid.min(), vmax=valid.max()) if valid.size else None
    else:
        norm = None

    im = ax.imshow(
        data,
        cmap=cfg["cmap"],
        origin="lower",
        extent=extent,
        norm=norm,
    )

    if cfg["black_bg"]:
        ax.set_facecolor("black")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=cfg["cbar"])


def visualize_simulation_results(return_obj, mask=None, config=None):
    freq_extent, spatial_extent = _get_extents(config)

    # Collect fields in display order
    fields = []
    if mask is not None:
        fields.append(("mask", mask))

    for key in FIELD_CONFIG:
        if key in return_obj:
            data = return_obj[key]
            if np.iscomplexobj(data):
                data = np.abs(data)
            fields.append((key, data))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (key, data) in zip(axes, fields):
        cfg = FIELD_CONFIG[key]
        extent = freq_extent if cfg["domain"] == "frequency" else spatial_extent
        title = key.replace("_", " ").title()
        _plot_image(ax, data, cfg, title, extent)

    # Hide unused subplots
    for ax in axes[len(fields):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def visualize_comparison_multi(return_objs, parameter_values, parameter, visualise_parameter="resist_profile", titles=None, config=None):
    cfg = FIELD_CONFIG[visualise_parameter]
    freq_extent, spatial_extent = _get_extents(config)
    extent = freq_extent if cfg["domain"] == "frequency" else spatial_extent

    n = len(return_objs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    if titles is None:
        titles = [f"{parameter}: {parameter_values[i]}" for i in range(n)]

    for ax, obj, title in zip(axes, return_objs, titles):
        if visualise_parameter not in obj:
            ax.set_visible(False)
            continue

        data = obj[visualise_parameter]
        if np.iscomplexobj(data):
            data = np.abs(data)

        _plot_image(ax, data, cfg, title, extent)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def _get_extents(config):
    if config is None:
        return None, None

    grid_size = config.get("mask_grid_size", 512)
    mask_width_nm = config.get("mask_width_nm", 1000)
    spacing = mask_width_nm / grid_size

    freq = np.fft.fftshift(np.fft.fftfreq(grid_size, d=spacing))
    freq_extent = [freq[0], freq[-1], freq[0], freq[-1]]
    spatial_extent = [
        -mask_width_nm / 2,
        mask_width_nm / 2,
        -mask_width_nm / 2,
        mask_width_nm / 2,
    ]

    return freq_extent, spatial_extent


def compare_results(initial_config, initial_mask, param_name, param_values, visualise_parameter):
    #Simulate for each parameter value
    return_objs = []
    for i in range(6):
        sim_config_copy = initial_config.copy()
        sim_config_copy[param_name] = param_values[i]

        litho_sim = simulator.LithographySimulator(sim_config_copy)
        sim_results = litho_sim.simulate(initial_mask)
        return_objs.append(sim_results)

    #Visualise comparison
    visualize_comparison_multi(return_objs, param_values, parameter=param_name, visualise_parameter=visualise_parameter, config=initial_config)




def show_n_masks(dir_path, num_masks=5):
    

if __name__ == "__main__":
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)

    # Example usage: compare resist profiles for different resist blur values
    param_name = "resist_blur_sigma"
    sigma_values = np.linspace(0.5, 3.0, 6)
    mask_size = sim_config.get("mask_grid_size", 512)

    initial_mask = masks.read_mask_from_img("./data/ganopc-data/artitgt/1.glp.png", mask_grid_size=mask_size)
    compare_results(sim_config, initial_mask, param_name, sigma_values, 'resist_profile')

    random_masks = masks.get_dataset_masks(dir_path, num_masks, **sim_config)

    for i, mask in enumerate(random_masks):
        litho_sim = simulator.LithographySimulator(sim_config)
        sim_results = litho_sim.simulate(mask)
        simulation_visualiser.visualize_simulation_results(sim_results, mask=mask, config=sim_config)