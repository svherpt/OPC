import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

class SourceMaskOptimizer:
    def __init__(self, modelClass, modelPath, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = modelClass().to(self.device)
        self.model.load_state_dict(torch.load(modelPath, map_location=self.device))
        self.model.eval()
        
        for p in self.model.parameters():
            p.requires_grad = False

    def gaussian_blur(self, x, sigma):
        # Separable 2D Gaussian blur for efficiency
        if sigma <= 0:
            return x

        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        kernel_x = kernel.view(1, 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1)
        
        x = F.conv2d(x, kernel_x, padding=(0, kernel_size // 2))
        x = F.conv2d(x, kernel_y, padding=(kernel_size // 2, 0))
        return x

    def total_variation_loss(self, x):
        # Horizontal differences
        tv_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        # Vertical differences
        tv_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        return torch.mean(tv_h) + torch.mean(tv_v)

    def illumination_entropy_loss(self, illumination):
        """Encourage illumination to be spread out rather than concentrated"""
        illum_flat = illumination.flatten()
        illum_prob = illum_flat / (illum_flat.sum() + 1e-8)
        entropy = -torch.sum(illum_prob * torch.log(illum_prob + 1e-8))
        return -entropy  # Negative because we want to maximize entropy

    def compute_schedules(self, i, num_iterations, initial_blur, final_blur, in_binary_phase, binary_iterations):
        # Compute blur annealing and temperature schedule
        if not in_binary_phase:
            progress = min(i / (0.8 * num_iterations), 1.0)
            blur_sigma = initial_blur * (final_blur / initial_blur) ** progress
            temperature = min(i / (0.6 * num_iterations), 1.0)
        else:
            binary_progress = (i - num_iterations) / binary_iterations
            blur_sigma = final_blur * (0.1 / final_blur) ** binary_progress
            temperature = 1.0
            
        return blur_sigma, temperature

    def compute_loss_continuous(self, pred_intensity, pred_resist, target, mask, illumination, 
                                temperature, coverage_weight=0.05, tv_weight=0.01,
                                illum_entropy_weight=0.05, illum_tv_weight=0.02):
        # Temperature-weighted loss: intensity -> resist over time
        intensity_loss = F.mse_loss(pred_intensity, target)
        resist_loss = F.mse_loss(pred_resist, target)
        main_loss = (1 - temperature) * intensity_loss + temperature * resist_loss
        
        # Coverage loss: ensure total intensity matches target
        coverage_loss = (pred_resist.mean() - target.mean()) ** 2
        
        # Total Variation loss on mask: penalize spatial noise
        tv_loss = self.total_variation_loss(mask)
        
        # Illumination regularization to prevent point sources
        illum_entropy = self.illumination_entropy_loss(illumination)
        illum_tv = self.total_variation_loss(illumination)
        
        total_loss = (main_loss + 
                      coverage_weight * coverage_loss + 
                      tv_weight * tv_loss +
                      illum_entropy_weight * illum_entropy +
                      illum_tv_weight * illum_tv)
        
        return total_loss, intensity_loss, resist_loss, tv_loss, illum_entropy

    def compute_loss_binary(self, mask, illumination, pred_resist, target, i, num_iterations, 
                           binary_iterations, coverage_weight=0.05, binary_weight_max=0.1, 
                           tv_weight=0.005, illum_entropy_weight=0.03, illum_tv_weight=0.01):
        # Binary phase: push mask values toward 0 or 1
        resist_loss = F.mse_loss(pred_resist, target)
        binary_penalty = torch.mean(4 * mask * (1 - mask))
        
        binary_progress = (i - num_iterations) / binary_iterations
        binary_weight = binary_weight_max * (binary_progress ** 2)
        coverage_loss = (pred_resist.mean() - target.mean()) ** 2
        
        # Still apply TV loss in binary phase but with reduced weight
        tv_loss = self.total_variation_loss(mask)
        
        # Illumination regularization (continue in binary phase)
        illum_entropy = self.illumination_entropy_loss(illumination)
        illum_tv = self.total_variation_loss(illumination)
        
        total_loss = (resist_loss + 
                      coverage_weight * coverage_loss + 
                      binary_weight * binary_penalty + 
                      tv_weight * tv_loss +
                      illum_entropy_weight * illum_entropy +
                      illum_tv_weight * illum_tv)
        
        return total_loss, resist_loss, binary_penalty, tv_loss, illum_entropy

    def optimize(self, target_resist, illumination_shape, num_iterations=2000, lr_mask=0.15, lr_illum=0.1, 
                 initial_blur_mask=8.0, final_blur_mask=0.5, blur_illum=1.0, binarize_final=False, 
                 binary_iterations=300, tv_weight=0.01, tv_weight_binary=0.005,
                 illum_entropy_weight=0, illum_tv_weight=0):
        
        target = torch.from_numpy(target_resist.astype(np.float32)).to(self.device)
        target = target.unsqueeze(0).unsqueeze(0)

        # Random initialization
        init_mask = np.random.rand(*target_resist.shape).astype(np.float32)
        mask_param = nn.Parameter(torch.from_numpy(init_mask).float().unsqueeze(0).unsqueeze(0).to(self.device))

        init_illum = np.random.rand(*illumination_shape).astype(np.float32)
        illum_param = nn.Parameter(torch.from_numpy(init_illum).float().unsqueeze(0).unsqueeze(0).to(self.device))

        optimizer = torch.optim.Adam([
            {'params': [mask_param], 'lr': lr_mask},
            {'params': [illum_param], 'lr': lr_illum}
        ])

        history = {
            "loss": [],
            "intensity_loss": [],
            "resist_loss": [],
            "binary_penalty": [],
            "tv_loss": [],
            "illum_entropy": [],
            "temperature": [],
            "mask_snapshots": [],
            "illum_snapshots": []
        }

        total_iterations = num_iterations + (binary_iterations if binarize_final else 0)
        pbar = tqdm(range(total_iterations), desc="Optimizing source and mask")

        for i in pbar:
            optimizer.zero_grad()

            in_binary_phase = binarize_final and i >= num_iterations
            blur_sigma_mask, temperature = self.compute_schedules(i, num_iterations, initial_blur_mask, 
                                                                  final_blur_mask, in_binary_phase, binary_iterations)
            
            # Apply blur and clamp to [0,1]
            mask_blurred = self.gaussian_blur(mask_param, blur_sigma_mask)
            mask = torch.clamp(mask_blurred, 0.0, 1.0)
            
            illum_blurred = self.gaussian_blur(illum_param, blur_illum)
            illumination = torch.clamp(illum_blurred, 0.0, 1.0)
            
            # Run through lithography model
            pred_intensity, pred_resist = self.model(mask, illumination)

            if not in_binary_phase:
                # Phase 1: Temperature annealing from intensity to resist
                loss, intensity_loss, resist_loss, tv_loss, illum_entropy = self.compute_loss_continuous(
                    pred_intensity, pred_resist, target, mask, illumination, temperature, 
                    tv_weight=tv_weight, illum_entropy_weight=illum_entropy_weight,
                    illum_tv_weight=illum_tv_weight
                )
                binary_penalty_val = 0.0
            else:
                # Phase 2: Push toward binary values
                loss, resist_loss, binary_penalty, tv_loss, illum_entropy = self.compute_loss_binary(
                    mask, illumination, pred_resist, target, i, num_iterations, binary_iterations, 
                    tv_weight=tv_weight_binary, illum_entropy_weight=illum_entropy_weight * 0.6,
                    illum_tv_weight=illum_tv_weight * 0.5
                )
                intensity_loss = torch.tensor(0.0)
                binary_penalty_val = binary_penalty.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_param, illum_param], max_norm=1.0)
            optimizer.step()

            # Log metrics
            with torch.no_grad():
                history["loss"].append(loss.item())
                history["intensity_loss"].append(intensity_loss.item() if not in_binary_phase else 0.0)
                history["resist_loss"].append(resist_loss.item())
                history["binary_penalty"].append(binary_penalty_val)
                history["tv_loss"].append(tv_loss.item())
                history["illum_entropy"].append(illum_entropy.item())
                history["temperature"].append(temperature)

                if i % 10 == 0:
                    history["mask_snapshots"].append(mask.squeeze().cpu().numpy().copy())
                    history["illum_snapshots"].append(illumination.squeeze().cpu().numpy().copy())

            if i % 20 == 0:
                phase = "BINARY" if in_binary_phase else "CONT"
                postfix = {
                    "phase": phase, 
                    "loss": f"{loss.item():.6f}", 
                    "temp": f"{temperature:.2f}", 
                    "blur_m": f"{blur_sigma_mask:.2f}",
                    "tv": f"{tv_loss.item():.4f}",
                    "i_ent": f"{illum_entropy.item():.4f}"
                }
                if in_binary_phase:
                    postfix["bin_pen"] = f"{binary_penalty_val:.4f}"
                pbar.set_postfix(postfix)

        # Final smoothing
        with torch.no_grad():
            mask_final = self.gaussian_blur(mask_param, 0.1)
            mask_final = torch.clamp(mask_final, 0.0, 1.0)
            mask_result = mask_final.squeeze().cpu().numpy()
            
            illum_final = self.gaussian_blur(illum_param, blur_illum)
            illum_final = torch.clamp(illum_final, 0.0, 1.0)
            illum_result = illum_final.squeeze().cpu().numpy()

        print(f"\nSaved {len(history['mask_snapshots'])} frames for animation")
        
        if binarize_final:
            edges = np.sum((mask_result > 0.1) & (mask_result < 0.9))
            print(f"Mask binary quality: {100*(1-edges/mask_result.size):.1f}% of pixels are near 0 or 1")
        
        print(f"Illumination range: [{illum_result.min():.3f}, {illum_result.max():.3f}], mean: {illum_result.mean():.3f}")
        
        # Compute illumination spread metric
        illum_flat = illum_result.flatten()
        illum_prob = illum_flat / (illum_flat.sum() + 1e-8)
        entropy = -np.sum(illum_prob * np.log(illum_prob + 1e-8))
        max_entropy = np.log(illum_result.size)
        print(f"Illumination entropy: {entropy:.3f} / {max_entropy:.3f} (higher = more spread out)")

        return mask_result, illum_result, history