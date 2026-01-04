import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class MaskOptimizer:
    def __init__(self, modelClass, modelPath, device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = modelClass().to(self.device)
        self.model.load_state_dict(
            torch.load('./models/' + modelPath, map_location=self.device)
        )
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        print(f"Optimizer initialised on {self.device}")
        print(f"Model frozen with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def optimize(
        self,
        target_resist,
        initial_mask=None,
        num_iterations=300,
        lr=0.05,
        binarization_weight=0.1,
        tv_weight=0.001,
        binarize_final=True,
    ):

        target_resist = self._to_tensor(target_resist)

        if initial_mask is None:
            raw_mask = torch.zeros_like(target_resist, requires_grad=True)
        else:
            init = self._to_tensor(initial_mask)
            eps = 1e-4
            init = init.clamp(eps, 1 - eps)
            raw_mask = torch.log(init / (1 - init)).detach().clone()
            raw_mask.requires_grad_(True)

        optimiser = optim.Adam([raw_mask], lr=lr)

        history = {
            'total_loss': [],
            'mse_loss': [],
            'binary_loss': [],
            'tv_loss': [],
            'mask_mean': [],
        }

        pbar = tqdm(range(num_iterations), desc='Optimising')

        for it in pbar:
            optimiser.zero_grad()

            mask = torch.sigmoid(raw_mask)

            _, predicted_resist = self.model(mask)

            mse_loss = (predicted_resist - target_resist).pow(2).mean()
            binary_loss = (4 * mask * (1 - mask)).mean()
            tv_loss = self._tv_loss(mask)

            total_loss = (
                mse_loss
                + binarization_weight * binary_loss
                + tv_weight * tv_loss
            )

            total_loss.backward()
            optimiser.step()

            history['total_loss'].append(total_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['binary_loss'].append(binary_loss.item())
            history['tv_loss'].append(tv_loss.item())
            history['mask_mean'].append(mask.mean().item())

            if it % 20 == 0:
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4e}',
                    'MSE': f'{mse_loss.item():.4e}',
                    'Mean': f'{mask.mean().item():.3f}',
                })

        with torch.no_grad():
            final_mask = torch.sigmoid(raw_mask)
            if binarize_final:
                final_mask = (final_mask > 0.5).float()

        return final_mask.cpu().squeeze().numpy(), history

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        return x

    def _tv_loss(self, x):
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w
