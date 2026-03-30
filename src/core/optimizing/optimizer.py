# src/core/ml/optimiser.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

import src.core.misc as misc
import src.core.simulator.masks as masks_module
import src.core.simulator.lithography_simulator as simulator
from src.core.augmenters.illumination_augmenter import IlluminationAugmenter
from src.core.ml.predict import load_model_from_checkpoint
from src.visualizers.ml.optimisation_visualiser import show_optimisation_results, show_snapshot_evolution


class SourceMaskOptimiser:
    """Gradient-based optimiser for jointly optimising batches of masks given fixed illuminations."""

    def __init__(self, checkpoint_path, compile_model=True):
        """Load surrogate model from checkpoint, freeze parameters, and optionally compile."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = load_model_from_checkpoint(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if compile_model and self.device.type == "cuda":
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
            print("Done.")

        self.use_amp = self.device.type == "cuda"
        print(f"Loaded model from epoch {self.checkpoint['epoch']} (val_loss {self.checkpoint['val_loss']:.4f})")
        print(f"Optimising on: {self.device} | AMP: {self.use_amp}")


    def _gaussian_blur(self, x, sigma):
        """Apply separable 2D Gaussian blur to tensor x with given sigma."""
        if sigma <= 0:
            return x
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords   = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        kernel   = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel   = kernel / kernel.sum()
        kernel_x = kernel.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1)
        x = F.conv2d(x, kernel_x, padding=(0, kernel_size // 2), groups=x.shape[1])
        x = F.conv2d(x, kernel_y, padding=(kernel_size // 2, 0), groups=x.shape[1])
        return x


    def _apply_blur(self, mask_param, blur_sigma):
        """Apply blur and clamp, skipping blur when sigma is negligible."""
        if blur_sigma < 0.1:
            return torch.clamp(mask_param, 0.0, 1.0)
        return torch.clamp(self._gaussian_blur(mask_param, blur_sigma), 0.0, 1.0)


    def _tv_loss(self, x):
        """Compute mean total variation loss across batch."""
        return torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])) + \
               torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))


    def optimise_batch(self, target_resists, illum_quadrants,
                       num_iterations=500, lr_mask=0.05,
                       initial_blur_mask=4.0, final_blur_mask=0.01,
                       tv_weight=0.01, binarize_final=True,
                       binary_iterations=200, binary_weight_max=0.3,
                       snapshot_every=50):
        """Optimise a batch of masks in parallel to match target resist patterns.

        Args:
            target_resists:    numpy array [N, H, W] of target resist patterns
            illum_quadrants:   numpy array [N, H, W] of illumination quadrants
            num_iterations:    number of continuous phase iterations
            lr_mask:           mask learning rate
            initial_blur_mask: starting blur sigma
            final_blur_mask:   ending blur sigma
            tv_weight:         total variation weight
            binarize_final:    whether to run binary phase
            binary_iterations: number of binary phase iterations
            binary_weight_max: maximum binary penalty weight
            snapshot_every:    save snapshot every N iterations

        Returns:
            mask_results:  numpy array [N, H, W] of optimised masks
            history:       dict of loss curves and snapshots
        """
        N = len(target_resists)

        target = torch.from_numpy(
            target_resists.astype(np.float32)
        ).unsqueeze(1).to(self.device)                           # [N, 1, H, W]

        illum = torch.from_numpy(
            illum_quadrants.astype(np.float32)
        ).unsqueeze(1).to(self.device)                           # [N, 1, H, W]

        mask_param = nn.Parameter(
            torch.from_numpy(target_resists.copy().astype(np.float32)).unsqueeze(1).to(self.device)
        )                                                        # [N, 1, H, W]

        optimizer = torch.optim.Adam([{"params": [mask_param], "lr": lr_mask}])

        history = {
            "loss": [], "resist_loss": [], "tv_loss": [],
            "binary_penalty": [], "mask_snapshots": [],
        }

        total_iterations = num_iterations + (binary_iterations if binarize_final else 0)
        pbar = tqdm(range(total_iterations), desc=f"Optimising batch of {N}")

        for i in pbar:
            optimizer.zero_grad()

            in_binary_phase = binarize_final and i >= num_iterations

            if not in_binary_phase:
                progress   = min(i / (0.8 * num_iterations), 1.0)
                blur_sigma = initial_blur_mask * (final_blur_mask / initial_blur_mask) ** progress
            else:
                binary_progress = (i - num_iterations) / binary_iterations
                blur_sigma      = final_blur_mask * (0.1 / final_blur_mask) ** binary_progress

            mask = self._apply_blur(mask_param, blur_sigma)

            # Mixed precision forward pass
            if self.use_amp:
                with torch.autocast(device_type="cuda"):
                    _, pred_resist = self.model(mask, illum)
            else:
                _, pred_resist = self.model(mask, illum)

            pred_resist = pred_resist.float()

            resist_loss = F.mse_loss(pred_resist, target)
            tv_loss     = self._tv_loss(mask)

            if not in_binary_phase:
                loss               = resist_loss + tv_weight * tv_loss
                binary_penalty_val = 0.0
            else:
                binary_progress    = (i - num_iterations) / binary_iterations
                binary_weight      = binary_weight_max * (binary_progress ** 2)
                binary_penalty     = torch.mean(4 * mask * (1 - mask))
                binary_penalty_val = binary_penalty.item()
                loss               = resist_loss + tv_weight * tv_loss + binary_weight * binary_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_param], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                history["loss"].append(loss.item())
                history["resist_loss"].append(resist_loss.item())
                history["tv_loss"].append(tv_loss.item())
                history["binary_penalty"].append(binary_penalty_val)
                if i % snapshot_every == 0:
                    history["mask_snapshots"].append(
                        mask.squeeze(1).cpu().numpy().copy()     # [N, H, W]
                    )

            if i % 20 == 0:
                phase = "BINARY" if in_binary_phase else "CONT"
                pbar.set_postfix({
                    "phase":  phase,
                    "loss":   f"{loss.item():.6f}",
                    "resist": f"{resist_loss.item():.6f}",
                    "tv":     f"{tv_loss.item():.4f}",
                    "blur":   f"{blur_sigma:.2f}",
                    **({"bin": f"{binary_penalty_val:.4f}"} if in_binary_phase else {}),
                })

        with torch.no_grad():
            raw          = self._apply_blur(mask_param, 0.1).detach().squeeze(1).cpu().numpy()
            mask_results = (raw > 0.5).astype(np.float32)

        if binarize_final:
            edges = np.sum((raw > 0.1) & (raw < 0.9))
            print(f"Mask binary quality: {100 * (1 - edges / raw.size):.1f}% pixels near 0 or 1")

        return mask_results, history


    def optimise(self, target_resist, illum_quadrant, **kwargs):
        """Optimise a single mask — convenience wrapper around optimise_batch."""
        mask_results, history = self.optimise_batch(
            target_resists  = target_resist[np.newaxis],
            illum_quadrants = illum_quadrant[np.newaxis],
            **kwargs
        )
        history["mask_snapshots"] = [s[0] for s in history["mask_snapshots"]]
        return mask_results[0], history


def parse_args():
    """Parse command line arguments for optimiser."""
    parser = argparse.ArgumentParser(description="Run OPC mask optimisation")
    parser.add_argument("--checkpoint",        type=str, required=True, help="Checkpoint filename under checkpoints/")
    parser.add_argument("--batch_size",        type=int, default=8,     help="Number of masks to optimise in parallel")
    parser.add_argument("--base_iterations",   type=int, default=350,   help="Continuous phase iterations")
    parser.add_argument("--binary_iterations", type=int, default=150,   help="Binary phase iterations")
    parser.add_argument("--snapshot_every",    type=int, default=50,    help="Save snapshot every N iterations")
    parser.add_argument("--no_compile",        action="store_true",     help="Disable torch.compile")
    
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
    target_resists  = np.stack([
        masks_module.get_random_dataset_mask(**sim_config).astype(np.float32)
        for _ in range(args.batch_size)
    ])
    illum_quadrants = np.stack([
        (lambda q: q / (q.sum() + 1e-8))(
            illum_augmenter.augment_illumination(**sim_config).astype(np.float32)
        )
        for _ in range(args.batch_size)
    ])

    mask_results, history = optimiser.optimise_batch(
        target_resists  = target_resists,
        illum_quadrants = illum_quadrants,
        num_iterations      = args.base_iterations,
        binary_iterations   = args.binary_iterations,
        snapshot_every      = args.snapshot_every,
    )

    print(f"Optimised {args.batch_size} masks")
    print(f"Final loss  : {history['loss'][-1]:.6f}")
    print(f"Snapshots   : {len(history['mask_snapshots'])} × {args.batch_size} masks")

    # show_batch_results(
    #     target_resists  = target_resists,
    #     mask_results    = mask_results,
    #     illum_quadrants = illum_quadrants,
    #     save_dir        = "results",
    #     show            = True,
    # )

    for i in range(args.batch_size):
        show_snapshot_evolution(history, sample_idx=i, save_dir="results", show=True)