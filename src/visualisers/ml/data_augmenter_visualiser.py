import numpy as np
import matplotlib.pyplot as plt
import json
import src.core.simulator.masks as masks
from src.core.ml.data_augmenter import MaskAugmenter


def visualize_specific_augmentation(augmenter, mask, augmentation_name, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original
    axes[0].imshow(mask, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Augmented
    method = getattr(augmenter, augmentation_name)
    augmented = method(mask, **kwargs)
    axes[1].imshow(augmented, cmap='gray', interpolation='nearest')
    
    params_str = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
    axes[1].set_title(f'{augmentation_name}, params:({params_str})', fontsize=12)
    axes[1].axis('off')

    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{augmentation_name}_example.png', dpi=150)
    plt.show()
    
    return augmented


def visualize_augmentations(augmenter, mask, n_augmentations=5):
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
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example usage with a simple test mask
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    
    random_mask = masks.get_random_dataset_mask('ganopc-data/artitgt', **sim_config)
    augmenter = MaskAugmenter()
    
    # Visualize all augmentations
    visualize_augmentations(augmenter, random_mask)
    
    random_mask = masks.get_random_dataset_mask('ganopc-data/artitgt', **sim_config)
    visualize_specific_augmentation(augmenter, random_mask, 'add_bidirectional_boundary_noise')
    