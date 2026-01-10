import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path


def plot_optimization_result(target_resist, optimized_mask, history, model, device, 
                             save_dir='./visualizations', show=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Images (Target, Mask, Prediction)
    with torch.no_grad():
        mask_tensor = torch.from_numpy(optimized_mask).float().unsqueeze(0).unsqueeze(0).to(device)
        _, pred_resist = model(mask_tensor)
        pred_resist_np = pred_resist.cpu().squeeze().numpy()
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(target_resist, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Target Resist', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(optimized_mask, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Optimized Mask', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_resist_np, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Predicted Resist', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Loss curves
    iterations = range(len(history['total_loss']))
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(iterations, history['total_loss'], 'b-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Total Loss')
    ax4.set_title('Total Loss')
    ax4.grid(alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(iterations, history['mse_loss'], 'r-', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('MSE Loss')
    ax5.set_title('MSE Loss')
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(iterations, history['binary_score'], 'g-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Binary Score')
    ax6.set_title('Binary Score (lower = more binary)')
    ax6.grid(alpha=0.3)
    
    # Row 3: Error map and additional metrics
    error = np.abs(target_resist - pred_resist_np)
    
    ax7 = fig.add_subplot(gs[2, 0])
    im = ax7.imshow(error, cmap='hot', vmin=0, vmax=0.5)
    ax7.set_title(f'Error Map (Mean: {error.mean():.4f})', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(iterations, history['tv_loss'], 'm-', linewidth=2)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('TV Loss')
    ax8.set_title('Total Variation Loss')
    ax8.grid(alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(iterations, history['mask_mean'], 'c-', linewidth=2)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Mask Mean')
    ax9.set_title('Mean Mask Value')
    ax9.grid(alpha=0.3)
    
    path = save_dir / 'optimization_result.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    
    if show:
        plt.show()
    plt.close()


def plot_optimization_losses(history, save_dir='./visualizations', show=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    iterations = range(len(history['total_loss']))
    
    axes[0, 0].plot(iterations, history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(iterations, history['mse_loss'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(iterations, history['binary_loss'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Binary Loss')
    axes[1, 0].set_title('Binarization Loss')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(iterations, history['tv_loss'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('TV Loss')
    axes[1, 1].set_title('Total Variation Loss')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    path = save_dir / 'optimization_losses.png'
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    
    if show:
        plt.show()
    plt.close()