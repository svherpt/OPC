# src/core/ml/optimisation_visualiser.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from tqdm import tqdm
from pathlib import Path

import src.core.simulator.illuminator as illuminator
from src.core.ml.predict import predict_single


def _mirror_quadrant_to_full(quadrant):
    """Mirror a bottom-right quadrant into a full symmetric illumination pattern."""
    top_half = np.concatenate([quadrant[:, ::-1], quadrant], axis=1)
    return np.concatenate([top_half[::-1, :], top_half], axis=0)


def _upsample_illumination(illum_quadrant, target_size):
    """Mirror quadrant to full and upsample to target size."""
    full_illum = _mirror_quadrant_to_full(illum_quadrant)
    return illuminator.upsample_illumination(full_illum, target_size)


def _plot_field(ax, data, title, cmap, vmin=0, vmax=1):
    """Plot a single field with colorbar using simulator-style layout."""
    im      = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return im


def _get_nn_prediction(model, mask, illum_q, device):
    """Run surrogate model inference on numpy inputs, returning intensity and resist."""
    mask_tensor  = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    illum_tensor = torch.from_numpy(illum_q.astype(np.float32)).unsqueeze(0)
    return predict_single(model, mask_tensor, illum_tensor, device)


def _get_litho_prediction(litho_sim, mask, illum_q):
    """Run ground truth lithography simulator on mask and illumination quadrant."""
    results = litho_sim.simulate(mask, illum_q)
    return results["wafer_intensity"], results["resist_profile"]


def _get_conventional_baseline(litho_sim, target_resist, sim_config):
    """Simulate resist using target pattern as mask with conventional illumination."""
    conventional_illum = illuminator.create_quadrant_source(sim_config)
    _, baseline_resist = _get_litho_prediction(litho_sim, target_resist, conventional_illum)
    return baseline_resist


def show_optimisation_results(target_resist, optimised_mask, optimised_illum_q,
                               model, litho_sim, sim_config, device,
                               save_dir="results", show=True):
    """Show final optimisation results comparing NN and simulator predictions against target.

    Layout (2x4):
        Row 0: Target resist | Conventional baseline | Optimised mask | Optimised illumination
        Row 1: NN intensity  | Litho intensity       | NN resist      | Litho resist
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    baseline_resist                  = _get_conventional_baseline(litho_sim, target_resist, sim_config)
    nn_intensity,    nn_resist       = _get_nn_prediction(model, optimised_mask, optimised_illum_q, device)
    litho_intensity, litho_resist    = _get_litho_prediction(litho_sim, optimised_mask, optimised_illum_q)
    illum_display                    = _upsample_illumination(optimised_illum_q, target_size=optimised_mask.shape[0])
    max_intensity                    = max(nn_intensity.max(), litho_intensity.max())

    nn_mae    = np.abs(target_resist - nn_resist).mean()
    litho_mae = np.abs(target_resist - litho_resist).mean()

    fig, axes = plt.subplots(2, 4, figsize=(4.5 * 4, 4.5 * 2))

    _plot_field(axes[0, 0], target_resist,    "Target resist",              cmap="gray")
    _plot_field(axes[0, 1], baseline_resist,  "Conventional baseline",      cmap="gray")
    _plot_field(axes[0, 2], optimised_mask,   "Optimised mask",             cmap="gray")
    _plot_field(axes[0, 3], illum_display,    "Optimised illumination",     cmap="hot")
    _plot_field(axes[1, 0], nn_intensity,     "NN wafer intensity",         cmap="magma", vmin=0, vmax=max_intensity)
    _plot_field(axes[1, 1], litho_intensity,  "Litho wafer intensity",      cmap="magma", vmin=0, vmax=max_intensity)
    _plot_field(axes[1, 2], nn_resist,        f"NN resist (MAE {nn_mae:.4f})",    cmap="gray")
    _plot_field(axes[1, 3], litho_resist,     f"Litho resist (MAE {litho_mae:.4f})", cmap="gray")

    plt.tight_layout()
    save_path = Path(save_dir) / "optimised_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def create_optimisation_animation(target_resist, history, model, litho_sim, sim_config,
                                   device, output_path="results/optimisation.mp4",
                                   fps=10, figsize=(20, 10)):
    """Create an animation showing mask and illumination evolving over optimisation iterations."""
    mask_snapshots  = history["mask_snapshots"]
    illum_snapshots = history["illum_snapshots"]

    if not mask_snapshots:
        print("No snapshots in history.")
        return

    baseline_resist = _get_conventional_baseline(litho_sim, target_resist, sim_config)

    print("Processing snapshots...")
    nn_intensities, nn_resists         = [], []
    litho_intensities, litho_resists   = [], []

    for mask, illum_q in tqdm(zip(mask_snapshots, illum_snapshots), total=len(mask_snapshots)):
        nn_i,    nn_r    = _get_nn_prediction(model, mask, illum_q, device)
        litho_i, litho_r = _get_litho_prediction(litho_sim, mask, illum_q)
        nn_intensities.append(nn_i)
        nn_resists.append(nn_r)
        litho_intensities.append(litho_i)
        litho_resists.append(litho_r)

    max_intensity = max(max(i.max() for i in nn_intensities), max(i.max() for i in litho_intensities))

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    _plot_field(axes[0, 0], target_resist,   "Target resist",                  cmap="gray")
    _plot_field(axes[0, 1], baseline_resist, "Conventional baseline",           cmap="hot")

    im2    = axes[0, 2].imshow(mask_snapshots[0],  cmap="gray",  vmin=0, vmax=1, origin="lower")
    title2 = axes[0, 2].set_title("Optimised mask (iter 0)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    im3    = axes[0, 3].imshow(_upsample_illumination(illum_snapshots[0], mask_snapshots[0].shape[0]), cmap="hot", vmin=0, vmax=1, origin="lower")
    title3 = axes[0, 3].set_title("Optimised illumination (iter 0)", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")

    im4    = axes[1, 0].imshow(nn_intensities[0],    cmap="magma", vmin=0, vmax=max_intensity, origin="lower")
    axes[1, 0].set_title("NN wafer intensity",       fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    im5    = axes[1, 1].imshow(litho_intensities[0], cmap="magma", vmin=0, vmax=max_intensity, origin="lower")
    axes[1, 1].set_title("Litho wafer intensity",    fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    im6    = axes[1, 2].imshow(nn_resists[0],    cmap="gray", vmin=0, vmax=1, origin="lower")
    title6 = axes[1, 2].set_title(f"NN resist (MAE {np.abs(target_resist - nn_resists[0]).mean():.4f})", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    im7    = axes[1, 3].imshow(litho_resists[0], cmap="gray", vmin=0, vmax=1, origin="lower")
    title7 = axes[1, 3].set_title(f"Litho resist (MAE {np.abs(target_resist - litho_resists[0]).mean():.4f})", fontsize=11, fontweight="bold")
    axes[1, 3].axis("off")

    plt.tight_layout()

    def update(frame):
        """Update animation frame."""
        iteration = frame * 10
        im2.set_data(mask_snapshots[frame])
        title2.set_text(f"Optimised mask (iter {iteration})")
        im3.set_data(_upsample_illumination(illum_snapshots[frame], mask_snapshots[frame].shape[0]))
        title3.set_text(f"Optimised illumination (iter {iteration})")
        im4.set_data(nn_intensities[frame])
        im5.set_data(litho_intensities[frame])
        im6.set_data(nn_resists[frame])
        title6.set_text(f"NN resist (MAE {np.abs(target_resist - nn_resists[frame]).mean():.4f})")
        im7.set_data(litho_resists[frame])
        title7.set_text(f"Litho resist (MAE {np.abs(target_resist - litho_resists[frame]).mean():.4f})")
        return [im2, im3, im4, im5, im6, im7, title2, title3, title6, title7]

    anim   = FuncAnimation(fig, update, frames=len(mask_snapshots), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving animation to {output_path}...")
    with tqdm(total=len(mask_snapshots), desc="Rendering") as pbar:
        anim.save(output_path, writer=writer, dpi=150, progress_callback=lambda i, n: pbar.update(1))
    plt.close()
    print(f"Animation saved to {output_path}")