
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import random
from PIL import Image
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
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

    # new_ops = [('dilate_uniform', lambda m, s=s: augmenter.dilate_uniform(m, size=s)) for s in range(1, 5)]
    # new_ops = [('erode_uniform', lambda m, s=s: augmenter.erode_uniform(m, size=s)) for s in range(1, 9)]
    # new_ops = [('add_corner_rounding', lambda m, s=s: augmenter.add_corner_rounding(m, radius=s)) for s in range(1, 9)]
    # new_ops = [('add_noise', lambda m, s=s: augmenter.add_noise(m, noise_prob=0.01*s)) for s in range(1, 9)]
    new_ops = [('add_expansion_boundary_noise', lambda m, s=s: augmenter.add_expansion_boundary_noise(m, dilate_size=s, noise_density=0.25)) for s in range(1, 9)]
    # new_ops = [('add_erosion_boundary_noise', lambda m, s=s: augmenter.add_erosion_boundary_noise(m, erode_size=s, noise_density=0.5)) for s in range(1, 9)]
    
    for i, (name, func) in enumerate(new_ops, 1):
        aug_mask = func(mask)
        axes[i].imshow(aug_mask, cmap='gray', interpolation='nearest')
        axes[i].set_title(name.replace('_', ' ').title(), fontsize=11)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_random_mask_augmentations(augmenter, mask, n_samples=16):
    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    augmented, _ = augmenter.batch_augment([mask], augmentations_per_mask=n_samples)

    for idx, aug_mask in enumerate(augmented):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.imshow(aug_mask, cmap='gray', interpolation='nearest')
        ax.set_title(f"Sample {idx}", fontsize=10)
        ax.axis('off')

    for idx in range(n_samples, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_illumination_samplers(augmenter, sim_config, n_cols=5):
    #new_ops = [('uniform', lambda max_spatial_freq: augmenter._sample_uniform(max_spatial_freq)) for i in range(16)]
    new_ops = [('annular r={:.1f}'.format(r), lambda max_spatial_freq, r=r: augmenter._sample_radial(max_spatial_freq, r_range=(0.4, 0.8))) for _ in range(16)]
    # new_ops = [('center r={:.1f}'.format(r), lambda max_spatial_freq, r=r: augmenter._sample_radial(max_spatial_freq, r_range=(0.0, r))) for r in np.linspace(0.05, 0.5, 8)]
    # new_ops = [('clustered spread={:.2f}'.format(s), lambda max_spatial_freq, s=s: augmenter._sample_clustered(max_spatial_freq, center_range=(0.2, 0.8), spread=s)) for s in np.linspace(0.05, 0.5, 8)]

    max_spatial_freq = sim_config["numerical_aperture"] / sim_config["wavelength_nm"]
    grid_size = 32

    n_samples = len(new_ops)
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for idx, (name, sampler) in enumerate(new_ops):
        ax = axes[idx // n_cols, idx % n_cols]
        illum = augmenter._generate(grid_size, max_spatial_freq, sampler, n_modes=5)
        illum_full = illuminator.quadrant_to_full(np.clip(illum / illum.max(), 0, 1) if illum.max() > 0 else illum)
        im = ax.imshow(illum_full, extent=(-1, 1, -1, 1), origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    for idx in range(n_samples, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_random_illumination_augmentations(augmenter, sim_config, n_samples=20, n_cols=5):
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for idx in range(n_samples):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        # Generate quadrant illumination
        illum_quadrant = augmenter.augment_illumination(**sim_config)

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
    plt.show()

def visualize_dataset_samples(data_dir, sim_config, n_samples=100):
    data_dir = Path("./data/" + data_dir)

    # Pick random samples from train set
    mask_files = sorted((data_dir / 'train' / 'masks').glob('*.png'))
    selected = random.sample(mask_files, min(n_samples, len(mask_files)))

    for mask_file in selected:
        file_id = mask_file.stem

        mask = np.array(Image.open(data_dir / 'train' / 'masks' / f"{file_id}.png")) / 255.0
        illumination = np.array(Image.open(data_dir / 'train' / 'illuminations' / f"{file_id}.png")) / 255.0
        wafer_intensity = np.array(Image.open(data_dir / 'train' / 'intensities' / f"{file_id}.png")) / 255.0
        resist_profile = np.array(Image.open(data_dir / 'train' / 'resists' / f"{file_id}.png")) / 255.0

        sim_results = {"wafer_intensity": wafer_intensity, "resist_profile": resist_profile}

        # Illumination is saved as full (64x64), extract quadrant for visualizer
        h = illumination.shape[0] // 2
        illum_quadrant = illumination[h:, h:]

        simulation_visualizer.visualize_simulation_results(sim_results, mask=mask, illumination=illum_quadrant, config=sim_config)


if __name__ == "__main__":
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)

    # Visualize mask augmentations
    # while(True):
    #     random_mask = masks.get_random_dataset_mask('example_masks', **sim_config)
    #     mask_augmenter = MaskAugmenter()
    #     visualize_random_mask_augmentations(mask_augmenter, random_mask)

    # Visualize illumination augmentations
    # while(True):
    #     illum_augmenter = IlluminationAugmenter()
    #     visualize_random_illumination_augmentations(illum_augmenter, sim_config)

    visualize_dataset_samples('augmented_medium', sim_config)