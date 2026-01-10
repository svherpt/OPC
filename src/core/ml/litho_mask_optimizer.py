import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image


class MaskOptimizer:
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

    def compute_loss_continuous(self, pred_intensity, pred_resist, target, temperature, coverage_weight=0.05):
        # Temperature-weighted loss: intensity -> resist over time
        intensity_loss = F.mse_loss(pred_intensity, target)
        resist_loss = F.mse_loss(pred_resist, target)
        main_loss = (1 - temperature) * intensity_loss + temperature * resist_loss
        coverage_loss = (pred_resist.mean() - target.mean()) ** 2
        
        return main_loss + coverage_weight * coverage_loss, intensity_loss, resist_loss

    def compute_loss_binary(self, mask, pred_resist, target, i, num_iterations, binary_iterations, coverage_weight=0.05, binary_weight_max=0.1):
        # Binary phase: push mask values toward 0 or 1
        resist_loss = F.mse_loss(pred_resist, target)
        binary_penalty = torch.mean(4 * mask * (1 - mask))
        
        binary_progress = (i - num_iterations) / binary_iterations
        binary_weight = binary_weight_max * (binary_progress ** 2)
        coverage_loss = (pred_resist.mean() - target.mean()) ** 2
        
        return resist_loss + coverage_weight * coverage_loss + binary_weight * binary_penalty, resist_loss, binary_penalty

    def optimize(self, target_resist, num_iterations=2000, lr=0.15, initial_blur=8.0, final_blur=0.5, binarize_final=False, binary_iterations=300):
        
        target = torch.from_numpy(target_resist.astype(np.float32)).to(self.device)
        target = target.unsqueeze(0).unsqueeze(0)

        # Random initialization
        init_mask = np.random.rand(*target_resist.shape).astype(np.float32)
        mask_param = nn.Parameter(torch.from_numpy(init_mask).float().unsqueeze(0).unsqueeze(0).to(self.device))

        optimizer = torch.optim.Adam([mask_param], lr=lr)

        history = {
            "loss": [],
            "intensity_loss": [],
            "resist_loss": [],
            "binary_penalty": [],
            "temperature": [],
            "mask_snapshots": []
        }

        total_iterations = num_iterations + (binary_iterations if binarize_final else 0)
        pbar = tqdm(range(total_iterations), desc="Optimizing mask")

        for i in pbar:
            optimizer.zero_grad()

            in_binary_phase = binarize_final and i >= num_iterations
            blur_sigma, temperature = self.compute_schedules(i, num_iterations, initial_blur, final_blur, in_binary_phase, binary_iterations)
            
            # Apply blur and clamp to [0,1]
            mask_blurred = self.gaussian_blur(mask_param, blur_sigma)
            mask = torch.clamp(mask_blurred, 0.0, 1.0)
            
            # Run through lithography model
            pred_intensity, pred_resist = self.model(mask)

            if not in_binary_phase:
                # Phase 1: Temperature annealing from intensity to resist
                loss, intensity_loss, resist_loss = self.compute_loss_continuous(pred_intensity, pred_resist, target, temperature)
                binary_penalty_val = 0.0
            else:
                # Phase 2: Push toward binary values
                loss, resist_loss, binary_penalty = self.compute_loss_binary(mask, pred_resist, target, i, num_iterations, binary_iterations)
                intensity_loss = torch.tensor(0.0)
                binary_penalty_val = binary_penalty.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_param], max_norm=1.0)
            optimizer.step()

            # Log metrics
            with torch.no_grad():
                history["loss"].append(loss.item())
                history["intensity_loss"].append(intensity_loss.item() if not in_binary_phase else 0.0)
                history["resist_loss"].append(resist_loss.item())
                history["binary_penalty"].append(binary_penalty_val)
                history["temperature"].append(temperature)

                if i % 10 == 0:
                    history["mask_snapshots"].append(mask.squeeze().cpu().numpy().copy())

            if i % 20 == 0:
                phase = "BINARY" if in_binary_phase else "CONT"
                postfix = {"phase": phase, "loss": f"{loss.item():.6f}", "temp": f"{temperature:.2f}", "blur": f"{blur_sigma:.2f}"}
                if in_binary_phase:
                    postfix["bin_pen"] = f"{binary_penalty_val:.4f}"
                pbar.set_postfix(postfix)

        # Final smoothing
        with torch.no_grad():
            mask_final = self.gaussian_blur(mask_param, 0.1)
            mask_final = torch.clamp(mask_final, 0.0, 1.0)
            result = mask_final.squeeze().cpu().numpy()

        print(f"\nSaved {len(history['mask_snapshots'])} frames for animation")
        
        if binarize_final:
            edges = np.sum((result > 0.1) & (result < 0.9))
            print(f"Binary quality: {100*(1-edges/result.size):.1f}% of pixels are near 0 or 1")

        return result, history