import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.core.augmenters.mask_augmenter import MaskAugmenter
from src.core.augmenters.illumination_augmenter import IlluminationAugmenter
import src.core.simulator.masks as masks
import src.core.simulator.lithography_simulator as simulator
import src.core.simulator.illuminator as illuminator
import src.core.misc as misc


def generate_triplets(base_masks, mask_variants_per_mask, illum_per_mask, sim_config):
    """Generate a list of (aug_mask, full_illumination, sim_results) triplets."""
    mask_augmenter = MaskAugmenter()
    illumination_augmenter = IlluminationAugmenter()
    sim = simulator.LithographySimulator(sim_config)

    triplets = []

    for mask in base_masks:
        # Generate mask variants
        mask_variants = [mask_augmenter.random_augmentation(mask) for _ in range(mask_variants_per_mask)]

        for aug_mask in mask_variants:
            for _ in range(illum_per_mask):
                # Generate illumination quadrant and normalize
                illum_quadrant = illumination_augmenter.augment_illumination(**sim_config)
                illum_quadrant /= (illum_quadrant.sum() + 1e-8)

                sim_results = sim.simulate(aug_mask, illum_quadrant)
                full_illum = illuminator.quadrant_to_full(illum_quadrant)

                triplets.append((aug_mask, full_illum, sim_results))

    return triplets


def save_triplets(triplets, output_dir, train_split=0.8):
    """Save a list of (mask, illumination, sim_results) triplets into structured train/test folders."""
    output_dir = Path("./data") / output_dir
    splits = ['train', 'test']
    subdirs = ['masks', 'illuminations', 'intensities', 'resists']

    for split in splits:
        for subdir in subdirs:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    n_total = len(triplets)
    n_train = int(n_total * train_split)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    split_indices = {
        'train': indices[:n_train],
        'test': indices[n_train:]
    }

    for split_name, idxs in split_indices.items():
        split_dir = output_dir / split_name

        # Get current count to continue numbering
        existing_files = list((split_dir / 'masks').glob('*.png'))
        start_id = len(existing_files)

        for i, data_idx in enumerate(idxs):
            mask, illumination, sim_results = triplets[data_idx]
            file_id = start_id + i

            # Save mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(split_dir / 'masks' / f"{file_id:06d}.png")
            # Save illumination
            Image.fromarray((illumination * 255).astype(np.uint8)).save(split_dir / 'illuminations' / f"{file_id:06d}.png")
            # Save wafer intensity
            Image.fromarray((sim_results["wafer_intensity"] * 255).astype(np.uint8)).save(split_dir / 'intensities' / f"{file_id:06d}.png")
            # Save resist profile
            Image.fromarray((sim_results["resist_profile"] * 255).astype(np.uint8)).save(split_dir / 'resists' / f"{file_id:06d}.png")


def main():
    sim_config = misc.get_simulation_config()
    num_base_masks_total = 10
    mask_variants_per_mask = 5
    illum_per_mask = 5
    output_dir = 'augmented_medium'

    # Load all base masks
    base_masks = masks.get_dataset_masks('example_masks', num_base_masks_total, **sim_config)

    batch_size = 20 
    # Generate in batches, progress bar outside
    for batch_start in tqdm(range(0, num_base_masks_total, batch_size), desc="Batches"):
        batch_masks = base_masks[batch_start: batch_start + batch_size]
        triplets = generate_triplets(batch_masks, mask_variants_per_mask, illum_per_mask, sim_config)
        save_triplets(triplets, output_dir)