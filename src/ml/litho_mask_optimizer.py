import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from tqdm import tqdm


class LithoMaskOptimizer:
    def __init__(self, surrogate_model, physical_simulator=None, device='cuda', img_size=512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.surrogate = surrogate_model.to(self.device)
        self.surrogate.eval()
        for param in self.surrogate.parameters():
            param.requires_grad = False

        self.physical_sim = physical_simulator
        self.img_size = img_size
        print(f"Optimizer initialized on {self.device}")
        print(f"Surrogate model frozen with {sum(p.numel() for p in self.surrogate.parameters()):,} parameters")
        if physical_simulator:
            print("Physical simulator loaded for ground truth comparison")

    def optimize_mask(self, target_resist, initial_mask=None, num_iterations=500,
                      learning_rate=0.1, binarization_weight=0.1, tv_weight=0.001,
                      binarize_final=True):
        if isinstance(target_resist, np.ndarray):
            target_resist = torch.from_numpy(target_resist).float()
        target_resist = target_resist.to(self.device)
        if target_resist.dim() == 2:
            target_resist = target_resist.unsqueeze(0).unsqueeze(0)
        elif target_resist.dim() == 3:
            target_resist = target_resist.unsqueeze(0)

        if initial_mask is None:
            mask = target_resist.clone()
        else:
            mask = initial_mask
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            mask = mask.to(self.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)

        mask = nn.Parameter(mask)
        optimizer = optim.Adam([mask], lr=learning_rate)

        history = {
            'loss': [], 'mse_loss': [], 'binary_loss': [], 'tv_loss': [],
            'mask_mean': [], 'mask_binary_score': []
        }

        pbar = tqdm(range(num_iterations))
        for iteration in pbar:
            optimizer.zero_grad()

            with torch.no_grad():
                mask.data.clamp_(0, 1)

            # Forward pass: only take resist output
            _, predicted_resist = self.surrogate(mask)

            mse_loss = ((predicted_resist - target_resist) ** 2).mean()
            binary_loss = (4 * mask * (1 - mask)).mean()
            tv_loss = self._total_variation_loss(mask)
            total_loss = mse_loss + binarization_weight * binary_loss + tv_weight * tv_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                binary_score = (4 * mask * (1 - mask)).mean().item()

            history['loss'].append(total_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['binary_loss'].append(binary_loss.item())
            history['tv_loss'].append(tv_loss.item())
            history['mask_mean'].append(mask.data.mean().item())
            history['mask_binary_score'].append(binary_score)

            pbar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'MSE': f'{mse_loss.item():.6f}',
                'Binary': f'{binary_score:.4f}'
            })

        with torch.no_grad():
            mask.data.clamp_(0, 1)
            if binarize_final:
                mask.data = (mask.data > 0.5).float()

        optimized_mask = mask.detach().cpu().squeeze().numpy()
        return optimized_mask, history

    def _total_variation_loss(self, x):
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w

    def visualize_optimization(self, target_resist, optimized_mask, history,  save_path='optimization_result.png', show=False):

        with torch.no_grad():
            # Get predictions using TARGET as mask (baseline)
            target_tensor = torch.from_numpy(target_resist).float().unsqueeze(0).unsqueeze(0).to(self.device)
            _, pred_resist_target_nn = self.surrogate(target_tensor)
            pred_resist_target_nn = pred_resist_target_nn.cpu().squeeze().numpy()
            
            # Get predictions using OPTIMIZED mask
            mask_tensor = torch.from_numpy(optimized_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
            _, pred_resist_opt_nn = self.surrogate(mask_tensor)
            pred_resist_opt_nn = pred_resist_opt_nn.cpu().squeeze().numpy()

        # Physical simulator predictions
        pred_resist_target_phys = None
        pred_resist_opt_phys = None
        if self.physical_sim is not None:
            result_target = self.physical_sim.simulate(target_resist)
            pred_resist_target_phys = result_target['resist_profile']
            
            result_opt = self.physical_sim.simulate(optimized_mask)
            pred_resist_opt_phys = result_opt['resist_profile']

        # Determine number of columns
        num_cols = 5 if pred_resist_target_phys is not None else 3
        
        # Create figure with GridSpec for equal-sized subplots
        fig = plt.figure(figsize=(4 * num_cols, 8))
        gs = gridspec.GridSpec(2, num_cols, figure=fig, hspace=0.3, wspace=0.3)

        # Scale to 0-255 for visualization
        target_vis = target_resist * 255
        opt_mask_vis = optimized_mask * 255
        pred_target_nn_vis = pred_resist_target_nn * 255
        pred_opt_nn_vis = pred_resist_opt_nn * 255
        
        error_target_nn = np.abs(pred_target_nn_vis - target_vis)
        error_opt_nn = np.abs(pred_opt_nn_vis - target_vis)

        # Target resist profile (used as mask)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(target_vis, cmap='gray', vmin=0, vmax=255)
        ax.set_title('Target Resist\n(used as mask)', fontsize=10)
        ax.axis('off')

        # NN prediction with target as mask
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(pred_target_nn_vis, cmap='gray', vmin=0, vmax=255)
        ax.set_title('NN Prediction\n(target as mask)', fontsize=10)
        ax.axis('off')

        # NN error
        ax = fig.add_subplot(gs[0, 2])
        im = ax.imshow(error_target_nn, cmap='hot', vmin=0, vmax=50)
        ax.set_title(f'NN Error\nMean: {error_target_nn.mean():.2f}', fontsize=10)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        if pred_resist_target_phys is not None:
            pred_target_phys_vis = pred_resist_target_phys * 255
            error_target_phys = np.abs(pred_target_phys_vis - target_vis)
            
            # Physical prediction with target as mask
            ax = fig.add_subplot(gs[0, 3])
            ax.imshow(pred_target_phys_vis, cmap='gray', vmin=0, vmax=255)
            ax.set_title('Physical Prediction\n(target as mask)', fontsize=10)
            ax.axis('off')

            # Physical error
            ax = fig.add_subplot(gs[0, 4])
            im = ax.imshow(error_target_phys, cmap='hot', vmin=0, vmax=50)
            ax.set_title(f'Physical Error\nMean: {error_target_phys.mean():.2f}', fontsize=10)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        # Optimized mask
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(opt_mask_vis, cmap='gray', vmin=0, vmax=255)
        ax.set_title('Optimized Mask', fontsize=10)
        ax.axis('off')

        # NN prediction with optimized mask
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(pred_opt_nn_vis, cmap='gray', vmin=0, vmax=255)
        ax.set_title('NN Prediction\n(optimized mask)', fontsize=10)
        ax.axis('off')

        # NN error
        ax = fig.add_subplot(gs[1, 2])
        im = ax.imshow(error_opt_nn, cmap='hot', vmin=0, vmax=50)
        ax.set_title(f'NN Error\nMean: {error_opt_nn.mean():.2f}', fontsize=10)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        if pred_resist_opt_phys is not None:
            pred_opt_phys_vis = pred_resist_opt_phys * 255
            error_opt_phys = np.abs(pred_opt_phys_vis - target_vis)
            
            # Physical prediction with optimized mask
            ax = fig.add_subplot(gs[1, 3])
            ax.imshow(pred_opt_phys_vis, cmap='gray', vmin=0, vmax=255)
            ax.set_title('Physical Prediction\n(optimized mask)', fontsize=10)
            ax.axis('off')

            # Physical error
            ax = fig.add_subplot(gs[1, 4])
            im = ax.imshow(error_opt_phys, cmap='hot', vmin=0, vmax=50)
            ax.set_title(f'Physical Error\nMean: {error_opt_phys.mean():.2f}', fontsize=10)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def load_target_from_file(self, filepath):
        img = Image.open(filepath).convert('L')
        array = np.array(img, dtype=np.float32) / 255.0
        return array

    def save_mask(self, mask, filepath):
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(filepath)