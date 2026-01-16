import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch
import src.core.simulator.light_sources as light_sources
from tqdm import tqdm


def get_nn_prediction(model, mask, illumination, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    
    illum_tensor = torch.from_numpy(illumination.astype(np.float32)).to(device)
    illum_tensor = illum_tensor.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        intensity, resist = model(mask_tensor, illum_tensor)
    
    return intensity.squeeze().cpu().numpy(), resist.squeeze().cpu().numpy()


def get_lithosim_prediction(litho_sim, mask, illumination):
    results = litho_sim.simulate(mask, illumination)
    return results["wafer_intensity"], results["resist_profile"]


def get_conventional_baseline(litho_sim, target_resist, sim_config):
    conventional_illum = light_sources.create_quadrant_source(sim_config)
    _, baseline_resist = get_lithosim_prediction(litho_sim, target_resist, conventional_illum)
    return baseline_resist


def getUpsampledIllumination(illum_quadrant, target_size):
    full_illum = light_sources.quadrant_to_full(illum_quadrant)
    upsampled_illum = light_sources.upsample_illumination(full_illum, target_size)
    return upsampled_illum


def show_optimization_results(target_resist, optimized_mask, optimized_illum, model, litho_sim, sim_config, history=None, create_animation=True, gif_path='optimization.mp4', fps=30, figsize=(20, 10), device='cuda'):
    
    baseline_resist = get_conventional_baseline(litho_sim, target_resist, sim_config)
    
    nn_intensity, nn_resist = get_nn_prediction(model, optimized_mask, optimized_illum, device)
    litho_intensity, litho_resist = get_lithosim_prediction(litho_sim, optimized_mask, optimized_illum)
    
    max_intensity = max(nn_intensity.max(), litho_intensity.max())
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    im0 = axes[0, 0].imshow(target_resist, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Resist', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(baseline_resist, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Baseline (Target + Conv. Light)', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(optimized_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Optimized Mask', fontsize=14, weight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[0, 3].imshow(getUpsampledIllumination(optimized_illum, target_size=optimized_mask.shape[0]), cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title('Optimized Illumination', fontsize=14, weight='bold')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    im4 = axes[1, 0].imshow(nn_intensity, cmap='magma', vmin=0, vmax=max_intensity)
    axes[1, 0].set_title('NN Wafer Intensity', fontsize=14, weight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(litho_intensity, cmap='magma', vmin=0, vmax=max_intensity)
    axes[1, 1].set_title('Lithosim Wafer Intensity', fontsize=14, weight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(nn_resist, cmap='gray', vmin=0, vmax=1)
    nn_error = np.abs(target_resist - nn_resist)
    axes[1, 2].set_title(f'NN Resist (MAE: {nn_error.mean():.4f})', fontsize=14, weight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    im7 = axes[1, 3].imshow(litho_resist, cmap='gray', vmin=0, vmax=1)
    litho_error = np.abs(target_resist - litho_resist)
    axes[1, 3].set_title(f'Lithosim Resist (MAE: {litho_error.mean():.4f})', fontsize=14, weight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    plt.tight_layout()

    #save figure
    plt.savefig('results/optimised_results.png', dpi=150)
    
    if create_animation and history is not None:
        create_optimization_animation(target_resist, history, model, litho_sim, sim_config, output_path='results/' + gif_path, fps=fps, figsize=figsize, device=device)


def create_optimization_animation(target_resist, history, model, litho_sim, sim_config, output_path='optimization.mp4', fps=10, figsize=(20, 10), device='cuda'):
    mask_snapshots = history['mask_snapshots']
    illum_snapshots = history['illum_snapshots']
    
    if len(mask_snapshots) == 0:
        print("No mask snapshots in history!")
        return
    
    baseline_resist = get_conventional_baseline(litho_sim, target_resist, sim_config)
    
    nn_intensities = []
    nn_resists = []
    litho_intensities = []
    litho_resists = []
    
    for mask, illum in tqdm(zip(mask_snapshots, illum_snapshots), total=len(mask_snapshots), desc="Processing snapshots"):
        nn_intensity, nn_resist = get_nn_prediction(model, mask, illum, device)
        litho_intensity, litho_resist = get_lithosim_prediction(litho_sim, mask, illum)
        nn_intensities.append(nn_intensity)
        nn_resists.append(nn_resist)
        litho_intensities.append(litho_intensity)
        litho_resists.append(litho_resist)
    
    max_intensity = max(max(i.max() for i in nn_intensities), max(i.max() for i in litho_intensities))
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    im0 = axes[0, 0].imshow(target_resist, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Resist', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(baseline_resist, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Baseline (Target + Quad. Light)', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(mask_snapshots[0], cmap='gray', vmin=0, vmax=1)
    title2 = axes[0, 2].set_title('Optimized Mask (Iter 0)', fontsize=14, weight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[0, 3].imshow(getUpsampledIllumination(illum_snapshots[0], target_size=mask_snapshots[0].shape[0]), cmap='hot', vmin=0, vmax=1)
    title3 = axes[0, 3].set_title('Optimized Illumination (Iter 0)', fontsize=14, weight='bold')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    im4 = axes[1, 0].imshow(nn_intensities[0], cmap='magma', vmin=0, vmax=max_intensity)
    axes[1, 0].set_title('NN Wafer Intensity', fontsize=14, weight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(litho_intensities[0], cmap='magma', vmin=0, vmax=max_intensity)
    axes[1, 1].set_title('Lithosim Wafer Intensity', fontsize=14, weight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(nn_resists[0], cmap='gray', vmin=0, vmax=1)
    title6 = axes[1, 2].set_title(f'NN Resist (MAE: {np.abs(target_resist - nn_resists[0]).mean():.4f})', fontsize=14, weight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    im7 = axes[1, 3].imshow(litho_resists[0], cmap='gray', vmin=0, vmax=1)
    title7 = axes[1, 3].set_title(f'Lithosim Resist (MAE: {np.abs(target_resist - litho_resists[0]).mean():.4f})', fontsize=14, weight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    plt.tight_layout()
    
    def update(frame):
        iteration = frame * 10
        
        im2.set_data(mask_snapshots[frame])
        title2.set_text(f'Optimized Mask (Iter {iteration})')
        
        im3.set_data(getUpsampledIllumination(illum_snapshots[frame], target_size=mask_snapshots[frame].shape[0]))
        title3.set_text(f'Optimized Illumination (Iter {iteration})')
        
        im4.set_data(nn_intensities[frame])
        im5.set_data(litho_intensities[frame])
        
        im6.set_data(nn_resists[frame])
        title6.set_text(f'NN Resist (MAE: {np.abs(target_resist - nn_resists[frame]).mean():.4f})')
        
        im7.set_data(litho_resists[frame])
        title7.set_text(f'Lithosim Resist (MAE: {np.abs(target_resist - litho_resists[frame]).mean():.4f})')
        
        return [im2, im3, im4, im5, im6, im7, title2, title3, title6, title7]
    
    anim = FuncAnimation(fig, update, frames=len(mask_snapshots), interval=1000 / fps, blit=False)
    
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=8000, extra_args=['-pix_fmt', 'yuv420p'])
    
    print(f"Saving animation to {output_path}...")
    with tqdm(total=len(mask_snapshots), desc="Rendering animation") as pbar:
        anim.save(output_path, writer=writer, dpi=150, progress_callback=lambda i, n: pbar.update(1))
    plt.close()
    
    print(f"Animation saved to {output_path}")