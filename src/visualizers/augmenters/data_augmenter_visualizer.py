import numpy as np
import matplotlib.pyplot as plt
import json
import src.core.simulator.masks as masks
import src.core.simulator.illuminator as illuminator
from src.core.augmenters.mask_augmenter import MaskAugmenter
from src.core.augmenters.illumination_augmenter import IlluminationAugmenter


def visualize_mask_augmentations(augmenter, mask):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    axes[0].imshow(mask, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    new_ops = [
        ('expansion_boundary', lambda m: augmenter.add_expansion_boundary_noise(m, 3, 0.3)),
        ('erosion_boundary', lambda m: augmenter.add_erosion_boundary_noise(m, 3, 0.3)),
        ('bidirectional_boundary', lambda m: augmenter.add_bidirectional_boundary_noise(m, 3, 0.3, 0.3)),
        ('edge_noise', lambda m: augmenter.add_edge_noise(m, 0.15, 2)),
        ('global_noise', lambda m: augmenter.add_global_salt_pepper_noise(m, 0.08)),
        ('local_dense_noise', lambda m: augmenter.add_local_dense_noise(m, 5, 50, 0.3)),
    ]
    
    for i, (name, func) in enumerate(new_ops, 1):
        aug_mask = func(mask)
        axes[i].imshow(aug_mask, cmap='gray', interpolation='nearest')
        axes[i].set_title(name.replace('_', ' ').title(), fontsize=11)
        axes[i].axis('off')
    
    # Last two: combined augmentations
    for i in range(2):
        aug_mask = augmenter.random_augmentation(mask)
        axes[7 + i].imshow(aug_mask, cmap='gray', interpolation='nearest')
        axes[7 + i].set_title(f'Combined Random {i+1}', fontsize=11)
        axes[7 + i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./figures/mask_augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_illumination_augmentations(augmenter, sim_config, n_samples=20, n_cols=5):
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)
    
    for idx in range(n_samples):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        
        # Generate quadrant illumination
        illum_quadrant = augmenter.augment_illumination(
            quadrant_illum_grid_size=32,
            numerical_aperture=sim_config["numerical_aperture"],
            wavelength_nm=sim_config["wavelength_nm"]
        )
        
        # Convert to full for visualization
        illum_full = illuminator.quadrant_to_full(illum_quadrant)
        # illum_full = illumination.upsample_illumination(illum_full, target_size=256)
        
        ax.imshow(illum_full, extent=(-1, 1, -1, 1), origin='lower', cmap='hot')
        ax.set_title(f"Sample {idx}", fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_samples, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig('./figures/illumination_augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    
    # Visualize mask augmentations
    # random_mask = masks.get_random_dataset_mask('example_masks', **sim_config)
    # mask_augmenter = MaskAugmenter()
    # visualize_mask_augmentations(mask_augmenter, random_mask)
    
    # Visualize illumination augmentations
    while(True):
        light_augmenter = LightSourceAugmenter()
        visualize_illumination_augmentations(light_augmenter, sim_config, n_samples=20)