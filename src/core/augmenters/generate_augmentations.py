import argparse
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

    for mask in tqdm(base_masks, desc="Processing masks", disable=True):
        # Generate mask variants
        mask_variants = [
            mask_augmenter.random_augmentation(mask)
            for _ in range(mask_variants_per_mask)
        ]

        for aug_mask in mask_variants:
            for _ in range(illum_per_mask):
                # Generate illumination quadrant and normalize
                illum_quadrant = illumination_augmenter.augment_illumination(**sim_config)
                illum_quadrant /= (illum_quadrant.sum() + 1e-8)

                sim_results = sim.simulate(aug_mask, illum_quadrant)
                full_illum = illuminator.quadrant_to_full(illum_quadrant)

                triplets.append((aug_mask, full_illum, sim_results))

    return triplets


def save_triplets(triplets, output_dir):
    """Save triplets in order (no shuffling)."""
    output_dir = Path("./data") / output_dir
    subdirs = ['masks', 'illuminations', 'intensities', 'resists']

    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Continue numbering
    existing_files = list((output_dir / 'masks').glob('*.png'))
    start_id = len(existing_files)

    for i, (mask, illumination, sim_results) in enumerate(triplets):
        file_id = start_id + i

        Image.fromarray((mask * 255).astype(np.uint8)).save(
            output_dir / 'masks' / f"{file_id:06d}.png"
        )
        Image.fromarray((illumination * 255).astype(np.uint8)).save(
            output_dir / 'illuminations' / f"{file_id:06d}.png"
        )
        Image.fromarray((sim_results["wafer_intensity"] * 255).astype(np.uint8)).save(
            output_dir / 'intensities' / f"{file_id:06d}.png"
        )
        Image.fromarray((sim_results["resist_profile"] * 255).astype(np.uint8)).save(
            output_dir / 'resists' / f"{file_id:06d}.png"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lithography dataset")

    parser.add_argument(
        "--num_base_masks",
        type=int,
        required=True,
        help="Total number of base masks to generate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
        help="Batch size to process masks in chunks"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (inside ./data/)"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    sim_config = misc.get_simulation_config()
    num_base_masks_total = args.num_base_masks
    output_dir = args.output_dir

    batch_size = args.batch_size

    # Train/test split
    base_masks = masks.get_dataset_masks('example_masks', num_base_masks_total, **sim_config)
    split_idx = int(0.8 * num_base_masks_total)
    train_masks = base_masks[:split_idx]
    test_masks = base_masks[split_idx:]

    # TRAIN
    print("\n=== Generating TRAIN dataset ===")
    for batch_start in tqdm(range(0, len(train_masks), batch_size), desc="Train batches"):
        batch_masks = train_masks[batch_start : batch_start + batch_size]
        triplets = generate_triplets(batch_masks, mask_variants_per_mask=5, illum_per_mask=5, sim_config=sim_config)
        save_triplets(triplets, output_dir + "/train")

    # TEST
    print("\n=== Generating TEST dataset ===")
    for batch_start in tqdm(range(0, len(test_masks), batch_size), desc="Test batches"):
        batch_masks = test_masks[batch_start : batch_start + batch_size]
        triplets = generate_triplets(batch_masks, mask_variants_per_mask=5, illum_per_mask=5, sim_config=sim_config)
        save_triplets(triplets, output_dir + "/test")


if __name__ == "__main__":
    main()