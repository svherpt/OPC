import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history, save_dir='./visualizations', show=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    train_losses = history['train_loss']
    test_losses = history['test_loss']
    lrs = history.get('lr', None)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, test_losses, 'r-', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    if lrs and len(lrs) > 0:
        axes[1].plot(epochs, lrs, 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    path = save_dir / 'training_history.png'
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    
    if show:
        plt.show()
    plt.close()


def plot_predictions(model, dataset, device, n=6, save_dir='./visualizations', show=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    is_multi_target = len(dataset[0]) == 3
    
    if is_multi_target:
        _plot_multi_target_predictions(model, dataset, device, n, save_dir, show)
    else:
        _plot_single_target_predictions(model, dataset, device, n, save_dir, show)


def _plot_single_target_predictions(model, dataset, device, n, save_dir, show):
    model.eval()
    samples_per_page = 2
    num_pages = (n + samples_per_page - 1) // samples_per_page
    
    with torch.no_grad():
        for page in range(num_pages):
            fig, axes = plt.subplots(samples_per_page, 4, figsize=(16, 4 * samples_per_page))
            if samples_per_page == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Predictions - Page {page + 1}/{num_pages}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            for row in range(samples_per_page):
                sample_idx = page * samples_per_page + row
                
                if sample_idx >= n:
                    for col in range(4):
                        axes[row, col].axis('off')
                    continue
                
                inputs, targets = dataset[sample_idx]
                inputs = inputs.unsqueeze(0).to(device)
                outputs = model(inputs)
                
                input_img = inputs.cpu().squeeze().numpy() * 255
                target_img = targets.squeeze().numpy() * 255
                output_img = outputs.cpu().squeeze().numpy() * 255
                diff_img = np.abs(target_img - output_img)
                
                axes[row, 0].imshow(input_img, cmap='gray', vmin=0, vmax=255)
                axes[row, 0].set_title('Input', fontsize=12, fontweight='bold')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(output_img, cmap='gray', vmin=0, vmax=255)
                axes[row, 1].set_title('Prediction', fontsize=12, fontweight='bold')
                axes[row, 1].axis('off')
                
                axes[row, 2].imshow(target_img, cmap='gray', vmin=0, vmax=255)
                axes[row, 2].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[row, 2].axis('off')
                
                im = axes[row, 3].imshow(diff_img, cmap='hot', vmin=0, vmax=50)
                axes[row, 3].set_title(f'Error: {diff_img.mean():.2f}', 
                                      fontsize=12, fontweight='bold')
                axes[row, 3].axis('off')
                
                plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            filename = save_dir / f'predictions_page_{page + 1}.png'
            plt.savefig(filename, dpi=150)
            print(f"Saved {filename}")
            
            if show:
                plt.show()
            plt.close()


def _plot_multi_target_predictions(model, dataset, device, n, save_dir, show):
    model.eval()
    
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(n):
            x, _, y_resist = dataset[i]
            x = x.unsqueeze(0).to(device)
            _, pred_resist = model(x)

            x_img = x.cpu().squeeze().numpy() * 255
            y_img = y_resist.squeeze().numpy() * 255
            pred_img = pred_resist.cpu().squeeze().numpy() * 255
            diff = np.abs(y_img - pred_img)

            axes[i, 0].imshow(x_img, cmap='gray', vmin=0, vmax=255)
            axes[i, 1].imshow(pred_img, cmap='gray', vmin=0, vmax=255)
            axes[i, 2].imshow(y_img, cmap='gray', vmin=0, vmax=255)
            im = axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=50)

            for ax in axes[i]:
                ax.axis('off')

            axes[i, 0].set_title('Input')
            axes[i, 1].set_title('Prediction')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 3].set_title(f'Error: {diff.mean():.2f}')

            plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = save_dir / 'predictions.png'
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    
    if show:
        plt.show()
    plt.close()