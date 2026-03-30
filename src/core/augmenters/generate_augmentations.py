# src/data/generate_augmentations.py
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.core.augmenters.mask_augmenter import MaskAugmenter
from src.core.augmenters.illumination_augmenter import IlluminationAugmenter
import src.core.simulator.masks as masks
import src.core.simulator.lithography_simulator as simulator
import src.core.misc as misc


def get_start_id(output_dir):
    """Return next file ID based on minimum count across all subdirectories."""
    output_dir = Path(output_dir)
    counts = [
        len(list((output_dir / sub).glob("*.png")))
        for sub in ['masks', 'illuminations', 'intensities', 'resists']
        if (output_dir / sub).exists()
    ]
    return min(counts) if counts else 0

def ensure_dirs(output_dir):
    """Create subdirectories for masks, illuminations, intensities and resists."""
    for sub in ['masks', 'illuminations', 'intensities', 'resists']:
        (Path(output_dir) / sub).mkdir(parents=True, exist_ok=True)


def save_sample(output_dir, file_id, mask, illum_quadrant, sim_results):
    """Save a single sample to disk as PNG files."""
    output_dir = Path(output_dir)
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        output_dir / 'masks' / f"{file_id:06d}.png"
    )
    Image.fromarray((illum_quadrant * 255).astype(np.uint8)).save(
        output_dir / 'illuminations' / f"{file_id:06d}.png"
    )
    Image.fromarray((sim_results["wafer_intensity"] * 255).astype(np.uint8)).save(
        output_dir / 'intensities' / f"{file_id:06d}.png"
    )
    Image.fromarray((sim_results["resist_profile"] * 255).astype(np.uint8)).save(
        output_dir / 'resists' / f"{file_id:06d}.png"
    )


def process_split(base_masks, output_dir, sim_config, batch_size,
                  mask_variants, illum_per_variant, desc):
    """Generate and save augmented samples for a set of base masks."""
    ensure_dirs(output_dir)
    file_id     = get_start_id(output_dir)
    mask_aug    = MaskAugmenter()
    illum_aug   = IlluminationAugmenter()
    sim         = simulator.LithographySimulator(sim_config)

    for batch_start in tqdm(range(0, len(base_masks), batch_size), desc=desc):
        batch = base_masks[batch_start: batch_start + batch_size]
        for mask in batch:
            variants = [mask_aug.random_augmentation(mask) for _ in range(mask_variants)]
            for aug_mask in variants:
                for _ in range(illum_per_variant):
                    illum_q  = illum_aug.augment_illumination(**sim_config)
                    sim_results = sim.simulate(aug_mask, illum_q)
                    save_sample(output_dir, file_id, aug_mask, illum_q, sim_results)
                    file_id += 1


def parse_args():
    """Parse CLI arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate augmented lithography dataset")
    parser.add_argument("--num_base_masks",   type=int,   required=True,      help="Total number of base masks to use")
    parser.add_argument("--output_dir",       type=str,   required=True,      help="Output directory name under ./data/")
    parser.add_argument("--train_split",      type=float, default=0.8,        help="Fraction of masks used for train (rest goes to test)")
    parser.add_argument("--mask_variants",    type=int,   default=5,          help="Augmented mask variants per base mask")
    parser.add_argument("--illum_per_variant",type=int,   default=5,          help="Illuminations per mask variant")
    parser.add_argument("--batch_size",       type=int,   default=10,         help="Processing batch size")
    return parser.parse_args()


def main():
    args       = parse_args()
    sim_config = misc.get_simulation_config()

    base_masks = masks.get_dataset_masks('example_masks', args.num_base_masks, **sim_config)
    split_idx  = int(args.train_split * args.num_base_masks)
    train_masks = base_masks[:split_idx]
    test_masks  = base_masks[split_idx:]

    total_train = len(train_masks) * args.mask_variants * args.illum_per_variant
    total_test  = len(test_masks)  * args.mask_variants * args.illum_per_variant
    print(f"Generating {total_train} train samples and {total_test} test samples")
    print(f"Output: ./data/{args.output_dir}")

    if train_masks:
        print("\n=== TRAIN ===")
        process_split(
            train_masks,
            f"./data/{args.output_dir}/train",
            sim_config,
            args.batch_size,
            args.mask_variants,
            args.illum_per_variant,
            desc="Train batches"
        )

    if len(test_masks) > 0:
        print("\n=== TEST ===")
        process_split(
            test_masks,
            f"./data/{args.output_dir}/test",
            sim_config,
            args.batch_size,
            args.mask_variants,
            args.illum_per_variant,
            desc="Test batches"
        )


if __name__ == "__main__":
    main()