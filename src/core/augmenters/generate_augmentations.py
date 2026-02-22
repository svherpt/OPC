import numpy as np
import json
import random
from pathlib import Path
from PIL import Image
from src.core.augmenters.mask_augmenter import MaskAugmenter
from src.core.augmenters.illumination_augmenter import IlluminationAugmenter
from tqdm import tqdm
import src.core.simulator.masks as masks
import src.core.simulator.lithography_simulator as simulator
import src.core.simulator.illuminator as illuminator

def save_dataset(mask_illumination_simtriplets, output_dir, train_split=0.8):
    output_dir = Path("./data/" + output_dir)
    splits = ['train', 'test']
    subdirs = ['masks', 'illuminations', 'intensities', 'resists']

    for split in splits:
        for subdir in subdirs:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    n_total = len(mask_illumination_simtriplets)
    n_train = int(n_total * train_split)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    split_indices = {
        'train': indices[:n_train],
        'test': indices[n_train:]
    }

    for split_name, split_idx in split_indices.items():
        split_dir = output_dir / split_name

        # Get num files already in masks to continue numbering
        existing_files = list((split_dir / 'masks').glob('*.png'))
        start_id = len(existing_files)

        for idx, data_idx in enumerate(split_idx):
            mask, illumination, sim_results = mask_illumination_simtriplets[data_idx]

            file_id = start_id + idx

            # Save input mask
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(split_dir / 'masks' / f"{file_id:06d}.png")

            # Save illumination
            illum_img = Image.fromarray((illumination * 255).astype(np.uint8))
            illum_img.save(split_dir / 'illuminations' / f"{file_id:06d}.png")

            # Save wafer intensity
            wafer_intensity = sim_results["wafer_intensity"]
            intensity_img = Image.fromarray((wafer_intensity * 255).astype(np.uint8))
            intensity_img.save(split_dir / 'intensities' / f"{file_id:06d}.png")

            # Save resist profile
            resist_profile = sim_results["resist_profile"]
            resist_img = Image.fromarray((resist_profile * 255).astype(np.uint8))
            resist_img.save(split_dir / 'resists' / f"{file_id:06d}.png")


def generate_n_augmentations(num_masks, augmentations_per_mask, output_dir, sim_config):
    mask_augmenter = MaskAugmenter()
    illumination_augmenter = IlluminationAugmenter()

    base_masks = masks.get_dataset_masks('example_masks', num_masks, **sim_config)

    triplets = []
    for mask in base_masks:
        for _ in range(augmentations_per_mask):
            aug_mask = mask_augmenter.random_augmentation(mask)
            illum_quadrant = illumination_augmenter.augment_illumination(**sim_config)
            sim_results = simulator.LithographySimulator(sim_config).simulate(aug_mask, illum_quadrant)
            triplets.append((aug_mask, illuminator.quadrant_to_full(illum_quadrant), sim_results))

    save_dataset(triplets, output_dir)

def main():
    with open("sim_config.json", "r") as f:
            sim_config = json.load(f)

    for i in tqdm(range(100), desc="Generating batches"):
        generate_n_augmentations(num_masks=10, augmentations_per_mask=1, 
                                output_dir='augmented_medium', 
                                sim_config=sim_config)

if __name__ == "__main__":
    main()