import numpy as np
import matplotlib.pyplot as plt
import json
import src.core.simulator.masks as masks
from src.core.ml.data_augmenter import LightSourceAugmenter, MaskAugmenter


def visualize_mask_augmentations(augmenter, mask, n_augmentations=5):
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
    plt.savefig('./figures/augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()



# Visualization of multiple augmented full illuminations
def visualize_illumination_augmentations(illum_list, n_cols=5):
    n_rows = int(np.ceil(len(illum_list)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes)
    for idx, illum in enumerate(illum_list):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        ax.imshow(illum, extent=(-1,1,-1,1), origin='lower')
        ax.set_title(f"Sample {idx}", fontsize=10)
        ax.axis('off')
    # hide unused axes
    for idx in range(len(illum_list), n_rows*n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage with a simple test mask
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    
    random_mask = masks.get_random_dataset_mask('ganopc-data/artitgt', **sim_config)
    mask_augmenter = MaskAugmenter()
    
    # Visualize all augmentations
    visualize_mask_augmentations(mask_augmenter, random_mask)

    augmenter = LightSourceAugmenter()
    samples = [augmenter.augment_illumination(**sim_config) for _ in range(50)]
    visualize_illumination_augmentations(samples, n_cols=5)

    