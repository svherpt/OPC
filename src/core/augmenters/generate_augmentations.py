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


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_start_id(output_dir):
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    return len(list(mask_dir.glob("*.png")))


def ensure_dirs(output_dir):
    for sub in ['masks', 'illuminations', 'intensities', 'resists']:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def save_sample(output_dir, file_id, mask, illumination, sim_results):
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


# ─────────────────────────────────────────────────────────────
# Core generation (streaming)
# ─────────────────────────────────────────────────────────────

def process_split(base_masks, output_dir, sim_config, batch_size, desc):
    output_dir = Path("./data") / output_dir
    ensure_dirs(output_dir)

    file_id = get_start_id(output_dir)

    mask_augmenter = MaskAugmenter()
    illum_augmenter = IlluminationAugmenter()
    sim = simulator.LithographySimulator(sim_config)

    for batch_start in tqdm(range(0, len(base_masks), batch_size), desc=desc):
        batch_masks = base_masks[batch_start : batch_start + batch_size]

        for mask in batch_masks:
            mask_variants = [
                mask_augmenter.random_augmentation(mask)
                for _ in range(5)
            ]

            for aug_mask in mask_variants:
                for _ in range(5):
                    illum_quadrant = illum_augmenter.augment_illumination(**sim_config)
                    illum_quadrant /= (illum_quadrant.sum() + 1e-8)

                    sim_results = sim.simulate(aug_mask, illum_quadrant)
                    full_illum = illuminator.quadrant_to_full(illum_quadrant)

                    # SAVE IMMEDIATELY
                    save_sample(output_dir, file_id, aug_mask, full_illum, sim_results)
                    file_id += 1


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate lithography dataset (streaming)")

    parser.add_argument(
        "--num_base_masks",
        type=int,
        required=True,
        help="Total number of base masks"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size (does NOT affect memory anymore)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory inside ./data/"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    sim_config = misc.get_simulation_config()

    base_masks = masks.get_dataset_masks(
        'example_masks',
        args.num_base_masks,
        **sim_config
    )

    split_idx = int(0.8 * args.num_base_masks)
    train_masks = base_masks[:split_idx]
    test_masks = base_masks[split_idx:]

    print("\n=== TRAIN ===")
    process_split(
        train_masks,
        args.output_dir + "/train",
        sim_config,
        args.batch_size,
        desc="Train batches"
    )

    print("\n=== TEST ===")
    process_split(
        test_masks,
        args.output_dir + "/test",
        sim_config,
        args.batch_size,
        desc="Test batches"
    )


if __name__ == "__main__":
    main()