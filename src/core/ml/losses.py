import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.8, edge_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.edge_weight = edge_weight
        
    def forward(self, pred, target):
        mse_loss = torch.mean((pred - target) ** 2)
        
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        edge_loss = torch.mean((pred_dx - target_dx) ** 2) + \
                    torch.mean((pred_dy - target_dy) ** 2)
        
        return self.mse_weight * mse_loss + self.edge_weight * edge_loss


class EdgeAwareLoss(nn.Module):
    def __init__(self, edge_weight=2.0):
        super().__init__()
        self.edge_weight = edge_weight
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def get_edges(self, img):
        device = img.device
        edge_x = F.conv2d(img, self.sobel_x.to(device), padding=1)
        edge_y = F.conv2d(img, self.sobel_y.to(device), padding=1)
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

    
    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        edge_mse = torch.mean((pred_edges - target_edges) ** 2)
        return mse + self.edge_weight * edge_mse


class MultiHeadLoss(nn.Module):
    def __init__(self, w_resist=1.0, w_intensity=0.3, edge_weight=2.0):
        super().__init__()
        self.w_resist = w_resist
        self.w_intensity = w_intensity
        self.edge_loss = EdgeAwareLoss(edge_weight=edge_weight)

    def forward(self, pred_intensity, pred_resist, gt_intensity, gt_resist):
        loss_resist = self.edge_loss(pred_resist, gt_resist)
        loss_intensity = self.edge_loss(pred_intensity, gt_intensity)
        return self.w_resist * loss_resist + self.w_intensity * loss_intensity