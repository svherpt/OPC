import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path


def plot_training_history(history, save_dir=None, show=True):
    """Plot training history with loss and metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training History', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, history['test_loss'], label='Test Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[0, 1].plot(epochs, history['lr'], marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient Norm
    axes[0, 2].plot(epochs, history['grad_norm'], marker='o', color='red')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].set_title('Gradient Norm')
    axes[0, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Clip threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Intensity Metrics
    axes[1, 0].plot(epochs, history['intensity_mae'], label='MAE', marker='o')
    axes[1, 0].plot(epochs, history['intensity_rmse'], label='RMSE', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_title('Intensity Error Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resist Metrics
    axes[1, 1].plot(epochs, history['resist_mae'], label='MAE', marker='o')
    axes[1, 1].plot(epochs, history['resist_rmse'], label='RMSE', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_title('Resist Error Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # PSNR
    axes[1, 2].plot(epochs, history['intensity_psnr'], label='Intensity', marker='o')
    axes[1, 2].plot(epochs, history['resist_psnr'], label='Resist', marker='s')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('PSNR (dB)')
    axes[1, 2].set_title('Peak Signal-to-Noise Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def mirror_quadrant_to_full(quadrant):
    """
    Mirror a single quadrant to create full field.
    Input: bottom-right quadrant (H, W)
    Output: full field (2*H, 2*W) with proper mirroring
    """
    # Flip horizontally to get bottom-left
    bottom_left = np.fliplr(quadrant)
    # Combine bottom half
    bottom = np.concatenate([bottom_left, quadrant], axis=1)
    # Flip vertically to get top half
    top = np.flipud(bottom)
    # Combine full field
    full = np.concatenate([top, bottom], axis=0)
    return full


def plot_predictions(model, dataset, device='cuda', n=6, save_dir=None, show=True):
    """
    Plot predictions with layout:
    Each row: Mask | Illumination | Intensity Pred | Intensity True | Resist Pred | Resist True
    """
    model.eval()
    
    # Select random samples
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(n, 6, figsize=(20, 3.5*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Model Predictions', fontsize=16, y=0.998)
    
    # Column titles
    col_titles = ['Mask', 'Illumination\n(Full Field)', 'Intensity Pred', 'Intensity True', 'Resist Pred', 'Resist True']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            mask, illum_q, target_int, target_res = dataset[idx]
            
            # Add batch dimension and move to device
            mask_batch = mask.unsqueeze(0).to(device)
            illum_batch = illum_q.unsqueeze(0).to(device)
            
            # Get predictions
            pred_int, pred_res = model(mask_batch, illum_batch)
            
            # Move to CPU and squeeze
            mask_np = mask.squeeze().cpu().numpy()
            illum_q_np = illum_q.squeeze().cpu().numpy()
            pred_int_np = pred_int.squeeze().cpu().numpy()
            pred_res_np = pred_res.squeeze().cpu().numpy()
            target_int_np = target_int.squeeze().cpu().numpy()
            target_res_np = target_res.squeeze().cpu().numpy()
            
            # Mirror illumination quadrant to full field
            illum_full_np = mirror_quadrant_to_full(illum_q_np)
            
            # Plot: Mask
            im0 = axes[row, 0].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 0].axis('off')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
            
            # Plot: Illumination (full field) - keep 'hot' colormap
            im1 = axes[row, 1].imshow(illum_full_np, cmap='hot', vmin=0, vmax=1)
            axes[row, 1].axis('off')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
            
            # Plot: Intensity Prediction
            im2 = axes[row, 2].imshow(pred_int_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].axis('off')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
            
            # Plot: Intensity True
            im3 = axes[row, 3].imshow(target_int_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 3].axis('off')
            plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
            
            # Plot: Resist Prediction
            im4 = axes[row, 4].imshow(pred_res_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 4].axis('off')
            plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)
            
            # Plot: Resist True
            im5 = axes[row, 5].imshow(target_res_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 5].axis('off')
            plt.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'predictions.png'
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved predictions to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()