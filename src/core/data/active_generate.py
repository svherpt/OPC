# src/data/active_generate.py
import argparse
import numpy as np
from tqdm import tqdm

import src.core.misc as misc
import src.core.simulator.masks as masks_module
import src.core.simulator.lithography_simulator as simulator
from src.core.data.illumination_augmenter import IlluminationAugmenter
from src.core.optimizing.optimizer import SourceMaskOptimiser
from src.core.data.generate_augmentations import get_start_id, ensure_dirs, save_sample


def simulate_and_save(masks, illum_quadrants, litho_sim, output_dir, start_id):
    """Simulate each (mask, illum) pair with litho sim and save to disk."""
    file_id = start_id
    for mask, illum_q in tqdm(zip(masks, illum_quadrants), total=len(masks), desc="  Simulating"):
        sim_results = litho_sim.simulate(mask, illum_q)
        save_sample(output_dir, file_id, mask, illum_q, sim_results)
        file_id += 1
    return file_id


def run_active_generate(checkpoint, output_dir, num_batches, batch_size,
                        base_iterations, binary_iterations, snapshot_every,
                        coverage_weight, optimise_illum, sim_config):
    """Run optimiser over num_batches, collect all snapshots, simulate and save."""
    litho_sim       = simulator.LithographySimulator(sim_config)
    illum_augmenter = IlluminationAugmenter()
    optimiser       = SourceMaskOptimiser(checkpoint)

    ensure_dirs(output_dir)
    file_id = get_start_id(output_dir)

    total_snapshots = 0
    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")

        target_resists  = masks_module.get_batch('example_masks', batch_size, **sim_config)
        illum_quadrants = illum_augmenter.get_batch(batch_size, sim_config)

        _, illum_results, history = optimiser.optimise_batch(
            target_resists    = target_resists,
            illum_quadrants   = illum_quadrants,
            num_iterations    = base_iterations,
            binary_iterations = binary_iterations,
            snapshot_every    = snapshot_every,
            coverage_weight   = coverage_weight,
            optimise_illum    = optimise_illum,
            binarize_final    = False,  # keep continuous masks for training diversity
        )

        # Flatten snapshots across batch and time dimensions
        mask_snapshots  = history["mask_snapshots"]   # list of [N, H, W]
        illum_snapshots = history["illum_snapshots"]  # list of [N, H, W]

        all_masks  = np.concatenate(mask_snapshots,  axis=0)  # [N * T, H, W]
        all_illums = np.concatenate(illum_snapshots, axis=0)  # [N * T, H, W]

        print(f"  Collected {len(all_masks)} snapshots")
        file_id = simulate_and_save(all_masks, all_illums, litho_sim, output_dir, file_id)
        total_snapshots += len(all_masks)

    print(f"\nDone. Added {total_snapshots} samples to {output_dir}")
    print(f"Dataset now has {file_id} total samples")


def parse_args():
    """Parse CLI arguments for active data generation."""
    parser = argparse.ArgumentParser(description="Generate active learning data via optimiser snapshots")
    parser.add_argument("--run_name",          type=str,   required=True,  help="Experiment name matching checkpoints/<run_name>/")
    parser.add_argument("--output_dir",        type=str,   required=True,  help="Output directory under ./data/ — always appends to train")
    parser.add_argument("--num_batches",       type=int,   default=1)
    parser.add_argument("--batch_size",        type=int,   default=100)
    parser.add_argument("--base_iterations",   type=int,   default=500)
    parser.add_argument("--binary_iterations", type=int,   default=0)
    parser.add_argument("--snapshot_every",    type=int,   default=50)
    parser.add_argument("--coverage_weight",   type=float, default=0.05)
    parser.add_argument("--no_compile",        action="store_true")
    parser.add_argument("--fix_illum",         action="store_true")
    parser.add_argument("--resume", action="store_true", required=True, help="Active generation appends to existing data and uses latest checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    sim_config = misc.get_simulation_config()

    # Find latest checkpoint for this run
    from pathlib import Path
    checkpoints = sorted(Path(f"checkpoints/{args.run_name}").glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under checkpoints/{args.run_name}/")
    latest_checkpoint = str(checkpoints[-1])
    print(f"Using checkpoint: {latest_checkpoint}")

    output_dir = f"./data/{args.output_dir}/train"

    run_active_generate(
        checkpoint        = latest_checkpoint,
        output_dir        = output_dir,
        num_batches       = args.num_batches,
        batch_size        = args.batch_size,
        base_iterations   = args.base_iterations,
        binary_iterations = args.binary_iterations,
        snapshot_every    = args.snapshot_every,
        coverage_weight   = args.coverage_weight,
        optimise_illum    = not args.fix_illum,
        sim_config        = sim_config,
    )