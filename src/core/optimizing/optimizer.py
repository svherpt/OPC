# src/core/ml/optimiser.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

import src.core.misc as misc
import src.core.simulator.masks as masks_module
from src.core.data.illumination_augmenter import IlluminationAugmenter
from src.core.ml.predict import load_model_from_checkpoint


class SourceMaskOptimiser:
    """Gradient-based optimiser for batches of masks and illuminations."""

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
        """Separable 2D Gaussian blur applied independently per sample in batch."""
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
        """Blur and clamp mask, skipping blur when sigma is negligible."""
        if blur_sigma < 0.1:
            return torch.clamp(mask_param, 0.0, 1.0)
        return torch.clamp(self._gaussian_blur(mask_param, blur_sigma), 0.0, 1.0)


    def _tv_loss(self, x):
        """Total variation loss normalised per sample."""
        N = x.shape[0]
        return (torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])) +
                torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))) / N


    def _blur_schedule(self, i, num_iterations, initial_blur, final_blur,
                       in_binary_phase, binary_iterations):
        """Compute blur sigma for current iteration."""
        if not in_binary_phase:
            progress = min(i / (0.8 * num_iterations), 1.0)
            return initial_blur * (final_blur / initial_blur) ** progress
        binary_progress = (i - num_iterations) / binary_iterations
        return final_blur * (0.1 / final_blur) ** binary_progress


    def optimise_batch(self, target_resists, illum_quadrants,
                       num_iterations=500, lr_mask=0.05, lr_illum=0.05,
                       initial_blur_mask=4.0, final_blur_mask=0.01,
                       tv_weight=0, coverage_weight=0.05,
                       binarize_final=True, binary_iterations=200,
                       binary_weight_max=0.3, snapshot_every=50,
                       optimise_illum=True):
        """Optimise a batch of masks (and optionally illuminations) in parallel."""
        N      = len(target_resists)
        target = torch.from_numpy(target_resists.astype(np.float32)).unsqueeze(1).to(self.device)

        illum_tensor = torch.from_numpy(illum_quadrants.astype(np.float32)).unsqueeze(1).to(self.device)
        mask_param   = nn.Parameter(
            torch.from_numpy(target_resists.copy().astype(np.float32)).unsqueeze(1).to(self.device)
        )

        if optimise_illum:
            illum_param = nn.Parameter(illum_tensor.clone())
            optimizer   = torch.optim.Adam([
                {"params": [mask_param],  "lr": lr_mask},
                {"params": [illum_param], "lr": lr_illum},
            ])
        else:
            illum_param = illum_tensor
            optimizer   = torch.optim.Adam([{"params": [mask_param], "lr": lr_mask}])

        history = {
            "loss": [], "resist_loss": [], "tv_loss": [], "coverage_loss": [],
            "binary_penalty": [], "mask_snapshots": [], "illum_snapshots": [],
        }

        total_iterations = num_iterations + (binary_iterations if binarize_final else 0)
        pbar = tqdm(range(total_iterations), desc=f"Optimising batch of {N}")

        for i in pbar:
            optimizer.zero_grad()
            in_binary_phase = binarize_final and i >= num_iterations
            blur_sigma      = self._blur_schedule(i, num_iterations, initial_blur_mask,
                                                  final_blur_mask, in_binary_phase, binary_iterations)
            mask  = self._apply_blur(mask_param, blur_sigma)
            illum = torch.clamp(illum_param, 0.0, 1.0) if optimise_illum else illum_param

            if self.use_amp:
                with torch.autocast(device_type="cuda"):
                    _, pred_resist = self.model(mask, illum)
            else:
                _, pred_resist = self.model(mask, illum)

            pred_resist   = pred_resist.float()
            resist_loss   = F.mse_loss(pred_resist, target, reduction='sum') / N
            tv_loss       = self._tv_loss(mask)
            coverage_loss = (pred_resist.mean(dim=[1,2,3]) - target.mean(dim=[1,2,3])).pow(2).sum() / N

            if not in_binary_phase:
                loss               = resist_loss + tv_weight * tv_loss + coverage_weight * coverage_loss
                binary_penalty_val = 0.0
            else:
                binary_progress    = (i - num_iterations) / binary_iterations
                binary_weight      = binary_weight_max * (binary_progress ** 2)
                binary_penalty     = torch.sum(4 * mask * (1 - mask)) / N
                binary_penalty_val = binary_penalty.item()
                loss               = resist_loss + tv_weight * tv_loss + coverage_weight * coverage_loss + binary_weight * binary_penalty

            loss.backward()
            params_to_clip = [mask_param, illum_param] if optimise_illum else [mask_param]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                history["loss"].append(loss.item())
                history["resist_loss"].append(resist_loss.item())
                history["tv_loss"].append(tv_loss.item())
                history["coverage_loss"].append(coverage_loss.item())
                history["binary_penalty"].append(binary_penalty_val)
                if i % snapshot_every == 0:
                    history["mask_snapshots"].append(mask.squeeze(1).cpu().numpy().copy())
                    history["illum_snapshots"].append(illum.squeeze(1).detach().cpu().numpy().copy())

            if i % 20 == 0:
                phase = "BINARY" if in_binary_phase else "CONT"
                pbar.set_postfix({
                    "phase":    phase,
                    "loss":     f"{loss.item():.6f}",
                    "resist":   f"{resist_loss.item():.6f}",
                    "coverage": f"{coverage_loss.item():.4f}",
                    "tv":       f"{tv_loss.item():.4f}",
                    "blur":     f"{blur_sigma:.2f}",
                    **({"bin": f"{binary_penalty_val:.4f}"} if in_binary_phase else {}),
                })

        with torch.no_grad():
            raw           = self._apply_blur(mask_param, 0.1).detach().squeeze(1).cpu().numpy()
            mask_results  = (raw > 0.5).astype(np.float32)
            illum_results = torch.clamp(illum_param, 0.0, 1.0).detach().squeeze(1).cpu().numpy() \
                            if optimise_illum else illum_quadrants

        if binarize_final:
            edges = np.sum((raw > 0.1) & (raw < 0.9))
            print(f"Mask binary quality: {100 * (1 - edges / raw.size):.1f}% pixels near 0 or 1")

        return mask_results, illum_results, history


    def optimise(self, target_resist, illum_quadrant, **kwargs):
        """Optimise a single mask — convenience wrapper around optimise_batch."""
        mask_results, illum_results, history = self.optimise_batch(
            target_resists  = target_resist[np.newaxis],
            illum_quadrants = illum_quadrant[np.newaxis],
            **kwargs
        )
        history["mask_snapshots"]  = [s[0] for s in history["mask_snapshots"]]
        history["illum_snapshots"] = [s[0] for s in history["illum_snapshots"]]
        return mask_results[0], illum_results[0], history


def parse_args():
    """Parse CLI arguments for optimiser."""
    parser = argparse.ArgumentParser(description="Run OPC mask optimisation")
    parser.add_argument("--checkpoint",        type=str,   required=True)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--base_iterations",   type=int,   default=350)
    parser.add_argument("--binary_iterations", type=int,   default=150)
    parser.add_argument("--snapshot_every",    type=int,   default=50)
    parser.add_argument("--coverage_weight",   type=float, default=0.05)
    parser.add_argument("--no_compile",        action="store_true")
    parser.add_argument("--fix_illum",         action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    sim_config = misc.get_simulation_config()

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

    print(f"Optimised {args.batch_size} masks")
    print(f"Final loss : {history['loss'][-1]:.6f}")
    print(f"Snapshots  : {len(history['mask_snapshots'])} × {args.batch_size} masks")