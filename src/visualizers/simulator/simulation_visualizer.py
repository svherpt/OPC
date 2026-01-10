import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import src.core.simulator.masks as masks

FIELD_CONFIG = {
    "illumination": {
        "cmap": "viridis",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Intensity",
        "log": False,
        "black_bg": False,
    },
    "mask": {
        "cmap": "gray",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Transmission",
        "log": False,
        "black_bg": False,
    },
    "wafer_intensity": {
        "cmap": "hot",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Intensity",
        "log": False,
        "black_bg": False,
    },
    "resist_profile": {
        "cmap": "gray",
        "xlabel": "X position (nm)",
        "ylabel": "Y position (nm)",
        "cbar": "Resist",
        "log": False,
        "black_bg": False,
    },
}


def _plot_field(ax, data, cfg, title, extent):
    if cfg["log"]:
        valid = data[np.isfinite(data) & (data > 0)]
        norm = LogNorm(vmin=valid.min(), vmax=valid.max()) if valid.size else None
    else:
        norm = None

    im = ax.imshow(data, cmap=cfg["cmap"], origin="lower", extent=extent, norm=norm)

    if cfg["black_bg"]:
        ax.set_facecolor("black")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=cfg["cbar"])


def visualize_simulation_results(sim_results, mask=None, illumination=None, config=None):
    mask_size = config.get("mask_grid_size", 512) if config else sim_results["wafer_intensity"].shape[0]

    print(f"Visualizing simulation results with mask size: {mask.shape if mask is not None else 'N/A'} and illumination size: {illumination.shape if illumination is not None else 'N/A'}")

    mask_width_nm = config.get("mask_width_nm", 1000) if config else 1.0
    spatial_extent = [-mask_width_nm/2, mask_width_nm/2, -mask_width_nm/2, mask_width_nm/2]

    fields_to_plot = []

    if illumination is not None:
        fields_to_plot.append(("illumination", illumination))
    if mask is not None:
        fields_to_plot.append(("mask", mask))
    fields_to_plot.append(("wafer_intensity", sim_results["wafer_intensity"]))
    fields_to_plot.append(("resist_profile", sim_results["resist_profile"]))

    ncols = len(fields_to_plot)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    if ncols == 1:
        axes = [axes]

    for ax, (field_name, data) in zip(axes, fields_to_plot):
        cfg = FIELD_CONFIG[field_name]
        if np.iscomplexobj(data):
            data = np.abs(data)
        _plot_field(ax, data, cfg, title=field_name.replace("_", " ").title(), extent=spatial_extent)

    plt.tight_layout()
    plt.show()
