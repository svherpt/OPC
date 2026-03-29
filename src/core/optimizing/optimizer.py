# src/core/ml/optimiser.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import src.core.misc as misc
import src.core.simulator.masks as masks
import src.core.simulator.illuminator as illuminator
import src.core.simulator.lithography_simulator as simulator
from src.core.ml.predict import load_model_from_checkpoint
from src.visualizers.ml.optimisation_visualiser import show_optimisation_results


class SourceMaskOptimiser:
    """Gradient-based optimiser for mask given a fixed conventional illumination and target resist pattern."""

    def __init__(self, checkpoint_path):
        """Load surrogate model from checkpoint and freeze its parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = load_model_from_checkpoint(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"Loaded model from epoch {self.checkpoint['epoch']} (val_loss {self.checkpoint['val_loss']:.4f})")
        print(f"Optimising on: {self.device}")


    def _gaussian_blur(self, x, sigma):
        """Apply separable 2D Gaussian blur to tensor x with given sigma."""
        if sigma <= 0:
            return x
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords  = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        kernel  = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel  = kernel / kernel.sum()
        kernel_x = kernel.view(1, 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1)
        x = F.conv2d(x, kernel_x, padding=(0, kernel_size // 2))
        x = F.conv2d(x, kernel_y, padding=(kernel_size // 2, 0))
        return x


    def _tv_loss(self, x):
        """Compute total variation loss to penalise spatial noise."""
        return torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])) + \
               torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))

    def optimise(self, target_resist, illum_quadrant,
                num_iterations=500, lr_mask=0.05,
                initial_blur_mask=4.0, final_blur_mask=0.5,
                tv_weight=0.01, binarize_final=True,
                binary_iterations=200, binary_weight_max=0.3):
        """Optimise mask to match target resist with fixed illumination."""

        target = torch.from_numpy(target_resist.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        illum  = torch.from_numpy(illum_quadrant.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        mask_param = nn.Parameter(
            torch.from_numpy(target_resist.copy().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        optimizer = torch.optim.Adam([{"params": [mask_param], "lr": lr_mask}])
        history   = {"loss": [], "resist_loss": [], "tv_loss": [], "binary_penalty": [], "mask_snapshots": []}

        total_iterations = num_iterations + (binary_iterations if binarize_final else 0)
        pbar = tqdm(range(total_iterations), desc="Optimising mask")

        for i in pbar:
            optimizer.zero_grad()

            in_binary_phase = binarize_final and i >= num_iterations

            # Blur annealing
            if not in_binary_phase:
                progress   = min(i / (0.8 * num_iterations), 1.0)
                blur_sigma = initial_blur_mask * (final_blur_mask / initial_blur_mask) ** progress
            else:
                binary_progress = (i - num_iterations) / binary_iterations
                blur_sigma      = final_blur_mask * (0.1 / final_blur_mask) ** binary_progress

            mask = torch.clamp(self._gaussian_blur(mask_param, blur_sigma), 0.0, 1.0)

            _, pred_resist = self.model(mask, illum)

            resist_loss = F.mse_loss(pred_resist, target)
            tv_loss     = self._tv_loss(mask)

            if not in_binary_phase:
                loss = resist_loss + tv_weight * tv_loss
                binary_penalty_val = 0.0
            else:
                # Penalise grey values — 4*m*(1-m) is 0 at 0/1 and 1 at 0.5
                binary_progress    = (i - num_iterations) / binary_iterations
                binary_weight      = binary_weight_max * (binary_progress ** 2)
                binary_penalty     = torch.mean(4 * mask * (1 - mask))
                binary_penalty_val = binary_penalty.item()
                loss               = resist_loss + tv_weight * tv_loss + binary_weight * binary_penalty

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                history["loss"].append(loss.item())
                history["resist_loss"].append(resist_loss.item())
                history["tv_loss"].append(tv_loss.item())
                history["binary_penalty"].append(binary_penalty_val)
                if i % 10 == 0:
                    history["mask_snapshots"].append(mask.squeeze().cpu().numpy().copy())

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
            mask_result = torch.clamp(
                self._gaussian_blur(mask_param, 0.1), 0.0, 1.0
            ).squeeze().cpu().numpy()

        if binarize_final:
            edges = np.sum((mask_result > 0.1) & (mask_result < 0.9))
            print(f"Mask binary quality: {100 * (1 - edges / mask_result.size):.1f}% pixels near 0 or 1")

        return mask_result, history

if __name__ == "__main__":
    CHECKPOINT = "checkpoints/exp014_baseline/epoch_0050.pt"
    sim_config = misc.get_simulation_config()
    litho_sim  = simulator.LithographySimulator(sim_config)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    optimiser = SourceMaskOptimiser(CHECKPOINT)
    target    = masks.get_random_dataset_mask(**sim_config).astype(np.float32)

    from src.core.augmenters.illumination_augmenter import IlluminationAugmenter
    illumination_augmenter = IlluminationAugmenter()
    illum_q  = illumination_augmenter.augment_illumination(**sim_config)
    illum_q /= (illum_q.sum() + 1e-8)
    illum_q  = illum_q.astype(np.float32)

    mask_result, history = optimiser.optimise(
        target_resist=target,
        illum_quadrant=illum_q,
        num_iterations=500,
    )

    print(f"Mask  range : [{mask_result.min():.3f}, {mask_result.max():.3f}]")
    print(f"Final loss  : {history['loss'][-1]:.6f}")

    show_optimisation_results(
        target_resist=target,
        optimised_mask=mask_result,
        optimised_illum_q=illum_q,
        model=optimiser.model,
        litho_sim=litho_sim,
        sim_config=sim_config,
        device=device,
        save_dir="results",
        show=True,
    )