# src/visualizers/ml/optimisation_visualiser.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import argparse
from pathlib import Path

import src.core.simulator.illuminator as illuminator
import src.core.simulator.lithography_simulator as simulator
import src.core.simulator.masks as masks_module
import src.core.misc as misc
from src.core.data.illumination_augmenter import IlluminationAugmenter
from src.core.ml.predict import predict_single
from src.core.optimizing.optimizer import SourceMaskOptimiser


def _plot_field(ax, data, title, cmap, vmin=0, vmax=1):
    """Plot a single field with colorbar."""
    im      = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return im


def _nn_prediction(model, mask, illum_q, device):
    """Run surrogate model inference on numpy inputs, returning intensity and resist."""
    mask_tensor  = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    illum_tensor = torch.from_numpy(illum_q.astype(np.float32)).unsqueeze(0)
    return predict_single(model, mask_tensor, illum_tensor, device)


def _litho_prediction(litho_sim, mask, illum_q):
    """Run ground truth lithography simulator, returning intensity and resist."""
    results = litho_sim.simulate(mask, illum_q)
    return results["wafer_intensity"], results["resist_profile"]


def _illum_display(illum_q, target_size):
    """Mirror quadrant to full and upsample to target size for display."""
    full = illuminator.quadrant_to_full(illum_q)
    return illuminator.upsample_illumination(full, target_size)


def show_optimisation_results(target_resist, optimised_mask, optimised_illum_q,
                               model, litho_sim, sim_config, device,
                               save_dir="results", show=True, save=True, sample_idx=0):
    """Show final optimisation results comparing NN and simulator predictions against target."""
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    conv_illum                    = illuminator.create_quadrant_source(sim_config)
    _, baseline_resist            = _litho_prediction(litho_sim, target_resist, conv_illum)
    nn_intensity, nn_resist       = _nn_prediction(model, optimised_mask, optimised_illum_q, device)
    litho_intensity, litho_resist = _litho_prediction(litho_sim, optimised_mask, optimised_illum_q)
    illum_disp                    = _illum_display(optimised_illum_q, target_resist.shape[0])
    max_intensity                 = max(nn_intensity.max(), litho_intensity.max())
    nn_mae                        = np.abs(target_resist - nn_resist).mean()
    litho_mae                     = np.abs(target_resist - litho_resist).mean()
    baseline_mae                  = np.abs(target_resist - baseline_resist).mean()

    fig, axes = plt.subplots(3, 4, figsize=(4.5 * 4, 4.5 * 3))

    _plot_field(axes[0, 0], target_resist,   "Target resist",                      cmap="gray")
    _plot_field(axes[0, 1], baseline_resist, f"Baseline (MAE {baseline_mae:.4f})", cmap="gray")
    _plot_field(axes[0, 2], optimised_mask,  "Optimised mask",                     cmap="gray")
    _plot_field(axes[0, 3], illum_disp,      "Optimised illumination",             cmap="hot")
    _plot_field(axes[1, 0], nn_intensity,    "NN wafer intensity",                 cmap="magma", vmin=0, vmax=max_intensity)
    _plot_field(axes[1, 1], nn_resist,       f"NN resist (MAE {nn_mae:.4f})",      cmap="gray")
    _plot_field(axes[1, 2], litho_intensity, "Litho wafer intensity",              cmap="magma", vmin=0, vmax=max_intensity)
    _plot_field(axes[1, 3], litho_resist,    f"Litho resist (MAE {litho_mae:.4f})", cmap="gray")

    nn_error    = np.abs(target_resist - nn_resist)
    litho_error = np.abs(target_resist - litho_resist)
    _plot_field(axes[2, 0], nn_error,    "NN error",    cmap="hot", vmin=0, vmax=1)
    _plot_field(axes[2, 1], litho_error, "Litho error", cmap="hot", vmin=0, vmax=1)
    axes[2, 2].axis("off")
    axes[2, 3].axis("off")

    plt.tight_layout()
    if save:
        save_path = Path(save_dir) / f"optimised_results_{sample_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def show_snapshot_evolution(history, sample_idx=0, save_dir="results", show=True, save=True):
    """Show mask and illumination evolution across snapshots for a single sample."""
    mask_snapshots  = history["mask_snapshots"]
    illum_snapshots = history.get("illum_snapshots", [])

    if not mask_snapshots:
        print("No snapshots in history.")
        return

    N                = len(mask_snapshots)
    total_iterations = len(history["loss"])
    snapshot_every   = total_iterations // N if N > 0 else 1
    has_illum        = len(illum_snapshots) == N

    n_rows = 2 if has_illum else 1
    fig, axes = plt.subplots(n_rows, N, figsize=(3 * N, 3 * n_rows))
    if N == 1:
        axes = axes.reshape(n_rows, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for col, snapshot in enumerate(mask_snapshots):
        mask      = snapshot[sample_idx] if snapshot.ndim == 3 else snapshot
        iteration = col * snapshot_every
        axes[0, col].imshow(mask, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[0, col].set_title(f"iter {iteration}", fontsize=9)
        axes[0, col].axis("off")

    if has_illum:
        for col, snapshot in enumerate(illum_snapshots):
            illum = snapshot[sample_idx] if snapshot.ndim == 3 else snapshot
            axes[1, col].imshow(illum, cmap="hot", vmin=0, vmax=illum.max() + 1e-5, origin="lower")
            axes[1, col].axis("off")
        axes[1, 0].set_ylabel("Illumination", fontsize=9)

    axes[0, 0].set_ylabel("Mask", fontsize=9)
    plt.suptitle(f"Evolution — sample {sample_idx}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f"snapshot_evolution_{sample_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def parse_args():
    """Parse CLI arguments for visualiser."""
    parser = argparse.ArgumentParser(description="Run and visualise OPC optimisation results")
    parser.add_argument("--checkpoint",        type=str,   required=True)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--base_iterations",   type=int,   default=350)
    parser.add_argument("--binary_iterations", type=int,   default=150)
    parser.add_argument("--snapshot_every",    type=int,   default=50)
    parser.add_argument("--coverage_weight",   type=float, default=0.05)
    parser.add_argument("--no_compile",        action="store_true")
    parser.add_argument("--fix_illum",         action="store_true")
    parser.add_argument("--no_save",           action="store_true")
    parser.add_argument("--save_dir",          type=str,   default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    sim_config = misc.get_simulation_config()
    litho_sim  = simulator.LithographySimulator(sim_config)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    optimiser = SourceMaskOptimiser(
        f"checkpoints/{args.checkpoint}",
        compile_model=not args.no_compile
    )

    illum_augmenter = IlluminationAugmenter()
    target_resists  = masks_module.get_batch('example_masks', args.batch_size, **sim_config)
    illum_quadrants = illum_augmenter.get_batch(args.batch_size, sim_config)

    mask_results, illum_results, history = optimiser.optimise_batch(
        target_resists    = target_resists,
        illum_quadrants   = illum_quadrants,
        num_iterations    = args.base_iterations,
        binary_iterations = args.binary_iterations,
        snapshot_every    = args.snapshot_every,
        coverage_weight   = args.coverage_weight,
        optimise_illum    = not args.fix_illum,
    )

    save = not args.no_save
    for i in range(args.batch_size):
        show_optimisation_results(
            target_resist     = target_resists[i],
            optimised_mask    = mask_results[i],
            optimised_illum_q = illum_results[i],
            model             = optimiser.model,
            litho_sim         = litho_sim,
            sim_config        = sim_config,
            device            = device,
            save_dir          = args.save_dir,
            show              = True,
            save              = save,
            sample_idx        = i,
        )
        show_snapshot_evolution(
            history    = history,
            sample_idx = i,
            save_dir   = args.save_dir,
            show       = True,
            save       = save,
        )