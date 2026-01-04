import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class MaskOptimizer:
    def __init__(self, model, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Optimizer initialized on {self.device}")
        print(f"Model frozen with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def optimize(self, target_resist, initial_mask=None, num_iterations=500, 
                 lr=0.1, binarization_weight=0.1, tv_weight=0.001, binarize_final=True):
        
        target_resist = self._to_tensor(target_resist)
        
        if initial_mask is None:
            mask = torch.rand_like(target_resist)
        else:
            mask = self._to_tensor(initial_mask)
        
        mask = nn.Parameter(mask)
        optimizer = optim.Adam([mask], lr=lr)

        history = {
            'total_loss': [],
            'mse_loss': [],
            'binary_loss': [],
            'tv_loss': [],
            'mask_mean': [],
            'binary_score': []
        }

        pbar = tqdm(range(num_iterations), desc='Optimizing')
        for iteration in pbar:
            optimizer.zero_grad()

            with torch.no_grad():
                mask.data.clamp_(0, 1)

            _, predicted_resist = self.model(mask)

            mse_loss = ((predicted_resist - target_resist) ** 2).mean()
            binary_loss = (4 * mask * (1 - mask)).mean()
            tv_loss = self._total_variation_loss(mask)
            total_loss = mse_loss + binarization_weight * binary_loss + tv_weight * tv_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                binary_score = (4 * mask * (1 - mask)).mean().item()

            history['total_loss'].append(total_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['binary_loss'].append(binary_loss.item())
            history['tv_loss'].append(tv_loss.item())
            history['mask_mean'].append(mask.data.mean().item())
            history['binary_score'].append(binary_score)

            if iteration % 50 == 0:
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

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.device)
        
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        
        return x

    def _total_variation_loss(self, x):
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w