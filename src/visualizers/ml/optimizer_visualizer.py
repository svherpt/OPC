import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch


def get_nn_prediction(model, mask, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        _, pred_resist = model(mask_tensor)
    
    return pred_resist.squeeze().cpu().numpy()


def get_lithosim_prediction(litho_sim, mask):
    results = litho_sim.simulate(mask)
    return results["resist_profile"]


def show_optimization_results(target_resist, optimized_mask, model, litho_sim, history=None, 
                              create_animation=True, gif_path='optimization.gif', fps=30, 
                              figsize=(18, 10), device='cuda'):
    # Get predictions
    nn_pred = get_nn_prediction(model, optimized_mask, device)
    litho_pred = get_lithosim_prediction(litho_sim, optimized_mask)
    
    # Compute errors
    nn_error = np.abs(target_resist - nn_pred)
    litho_error = np.abs(target_resist - litho_pred)
    
    # Plot final results
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    im0 = axes[0, 0].imshow(target_resist, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Resist', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[1, 0].imshow(optimized_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Optimized Mask', fontsize=14, weight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(nn_pred, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('NN Prediction', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[1, 1].imshow(litho_pred, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Lithosim Prediction', fontsize=14, weight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    im4 = axes[0, 2].imshow(nn_error, cmap='hot', vmin=0, vmax=nn_error.max())
    axes[0, 2].set_title(f'NN Error (MAE: {nn_error.mean():.4f})', fontsize=14, weight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im4, ax=axes[0, 2], fraction=0.046)
    
    im5 = axes[1, 2].imshow(litho_error, cmap='hot', vmin=0, vmax=litho_error.max())
    axes[1, 2].set_title(f'Lithosim Error (MAE: {litho_error.mean():.4f})', fontsize=14, weight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    # Create animation if requested and history is provided
    if create_animation and history is not None:
        create_optimization_animation(target_resist, history, model, litho_sim, 'results/' + gif_path, fps, figsize, device)


def create_optimization_animation(target_resist, history, model, litho_sim, output_path='optimization.gif', 
                                 fps=10, figsize=(18, 10), device='cuda'):
    mask_snapshots = history['mask_snapshots']
    
    if len(mask_snapshots) == 0:
        print("No mask snapshots in history!")
        return
    
    # Precompute all predictions
    print("Precomputing predictions for animation...")
    nn_predictions = []
    litho_predictions = []
    
    for mask in mask_snapshots:
        nn_pred = get_nn_prediction(model, mask, device)
        litho_pred = get_lithosim_prediction(litho_sim, mask)
        nn_predictions.append(nn_pred)
        litho_predictions.append(litho_pred)
    
    # Compute global error range for consistent colormap
    all_nn_errors = [np.abs(target_resist - pred) for pred in nn_predictions]
    all_litho_errors = [np.abs(target_resist - pred) for pred in litho_predictions]
    max_nn_error = max(err.max() for err in all_nn_errors)
    max_litho_error = max(err.max() for err in all_litho_errors)
    
    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Initialize plots
    im0 = axes[0, 0].imshow(target_resist, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Resist', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[1, 0].imshow(mask_snapshots[0], cmap='gray', vmin=0, vmax=1)
    title1 = axes[1, 0].set_title('Mask (Iter 0)', fontsize=14, weight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(nn_predictions[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('NN Prediction', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[1, 1].imshow(litho_predictions[0], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Lithosim Prediction', fontsize=14, weight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    im4 = axes[0, 2].imshow(all_nn_errors[0], cmap='hot', vmin=0, vmax=max_nn_error)
    title4 = axes[0, 2].set_title(f'NN Error (MAE: {all_nn_errors[0].mean():.4f})', fontsize=14, weight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im4, ax=axes[0, 2], fraction=0.046)
    
    im5 = axes[1, 2].imshow(all_litho_errors[0], cmap='hot', vmin=0, vmax=max_litho_error)
    title5 = axes[1, 2].set_title(f'Lithosim Error (MAE: {all_litho_errors[0].mean():.4f})', fontsize=14, weight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    def update(frame):
        iteration = frame * 10  # Since we save every 10 iterations
        
        im1.set_data(mask_snapshots[frame])
        title1.set_text(f'Mask (Iter {iteration})')
        
        im2.set_data(nn_predictions[frame])
        im3.set_data(litho_predictions[frame])
        
        im4.set_data(all_nn_errors[frame])
        title4.set_text(f'NN Error (MAE: {all_nn_errors[frame].mean():.4f})')
        
        im5.set_data(all_litho_errors[frame])
        title5.set_text(f'Lithosim Error (MAE: {all_litho_errors[frame].mean():.4f})')
        
        return [im1, im2, im3, im4, im5, title1, title4, title5]
    
    print(f"Creating animation with {len(mask_snapshots)} frames...")
    anim = FuncAnimation(fig, update, frames=len(mask_snapshots), interval=1000/fps, blit=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()
    
    print(f"Animation saved to {output_path}")


# Example usage:
# show_optimization_results(target_resist, optimized_mask, model, litho_sim, history)
# 
# # Or without animation:
# show_optimization_results(target_resist, optimized_mask, model, litho_sim, create_animation=False)