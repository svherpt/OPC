"""
Active Learning System for Lithography Neural Network Training

This module implements an active learning loop that:
1. Optimizes source-mask pairs for random target patterns
2. Samples from optimization histories
3. Simulates with ground truth simulator
4. Identifies high-error predictions
5. Adds worst predictions to training set
6. Continues training the model

Run for 50 iterations to grow dataset from 50k to 100k samples.
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Import project modules
import src.core.simulator.masks as masks
import src.core.simulator.light_sources as light_sources
from src.core.simulator.lithography_simulator import LithographySimulator
from src.core.ml.models import LithographyUNet, LithographyDataset
from src.core.ml.trainer import Trainer
from src.core.ml.losses import MultiHeadLoss
from src.core.ml.litho_mask_optimizer import SourceMaskOptimizer


class ActiveLearner:
    
    def __init__(self, 
                 data_dir,
                 sim_config,
                 device='cuda',
                 w_resist=1.0,
                 w_intensity=3.0):
        
        self.device = device
        self.sim_config = sim_config
        self.data_dir = Path('./data') / data_dir
        self.w_resist = w_resist
        self.w_intensity = w_intensity
        
        # Model and optimizer will be initialized after initial training
        self.model = None
        self.optimizer = None
        
        # Initialize simulator
        self.simulator = LithographySimulator(sim_config, chunk_size=128)
        
        print(f"✓ Active Learner initialized")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Device: {device}")
    
    def optimize_batch(self, n_masks=100, **opt_kwargs):
        # Optimize source-mask pairs for random target patterns
        # Returns list of optimization results with histories
        
        print(f"\n{'='*60}")
        print(f"Optimizing {n_masks} random masks...")
        print(f"{'='*60}")
        
        results = []
        
        for i in tqdm(range(n_masks), desc="Batch optimization"):
            # Get random target mask
            target_resist = masks.get_random_dataset_mask(**self.sim_config)
            
            # Optimize
            optimized_mask, optimized_illum, history = self.optimizer.optimize(
                target_resist=target_resist,
                illumination_shape=(32, 32),
                **opt_kwargs
            )
            
            results.append({
                'target': target_resist,
                'final_mask': optimized_mask,
                'final_illum': optimized_illum,
                'history': history
            })
        
        return results
    
    def sample_from_histories(self, optimization_results, samples_per_mask=20):
        # Sample snapshots uniformly from optimization histories
        # Returns list of (mask, illumination) tuples
        
        print(f"\n{'='*60}")
        print(f"Sampling {samples_per_mask} snapshots from each history...")
        print(f"{'='*60}")
        
        all_samples = []
        
        for result in tqdm(optimization_results, desc="Sampling histories"):
            history = result['history']
            mask_snapshots = history['mask_snapshots']
            illum_snapshots = history['illum_snapshots']
            
            total = len(mask_snapshots)
            if total < samples_per_mask:
                # If fewer snapshots than requested, take all
                indices = list(range(total))
            else:
                # Uniform sampling
                indices = np.linspace(0, total-1, samples_per_mask, dtype=int)
            
            for idx in indices:
                all_samples.append({
                    'mask': mask_snapshots[idx],
                    'illumination': illum_snapshots[idx]
                })
        
        print(f"✓ Collected {len(all_samples)} samples")
        return all_samples
    
    def simulate_and_rank(self, samples):
        # Simulate samples with ground truth simulator and compute NN prediction errors
        # Returns list sorted by error (descending - worst first)
        
        print(f"\n{'='*60}")
        print(f"Simulating {len(samples)} samples and computing errors...")
        print(f"{'='*60}")
        
        errors = []
        
        for idx, sample in enumerate(tqdm(samples, desc="Simulating & comparing")):
            mask = sample['mask']
            illum_q = sample['illumination']
            
            # NN prediction
            pred_intensity, pred_resist = self.model.predict(mask, illum_q)
            
            # Ground truth simulation
            sim_results = self.simulator.simulate(mask, illum_q)
            sim_intensity = sim_results['wafer_intensity']
            sim_resist = sim_results['resist_profile']
            
            # Compute error (same weights as training loss)
            mse_intensity = np.mean((pred_intensity - sim_intensity) ** 2)
            mse_resist = np.mean((pred_resist - sim_resist) ** 2)
            combined_error = self.w_intensity * mse_intensity + self.w_resist * mse_resist
            
            errors.append({
                'idx': idx,
                'error': combined_error,
                'mse_intensity': mse_intensity,
                'mse_resist': mse_resist,
                'mask': mask,
                'illumination': illum_q,
                'sim_intensity': sim_intensity,
                'sim_resist': sim_resist
            })
        
        # Sort by error (descending - worst first)
        errors.sort(key=lambda x: x['error'], reverse=True)
        
        # Print statistics
        all_errors = [e['error'] for e in errors]
        print(f"\nError Statistics:")
        print(f"  Mean: {np.mean(all_errors):.6f}")
        print(f"  Std:  {np.std(all_errors):.6f}")
        print(f"  Min:  {np.min(all_errors):.6f}")
        print(f"  Max:  {np.max(all_errors):.6f}")
        print(f"  Median: {np.median(all_errors):.6f}")
        
        return errors
    
    def select_top_errors(self, ranked_errors, top_k=1000):
        # Select top K samples with highest errors
        
        selected = ranked_errors[:top_k]
        
        print(f"\n✓ Selected top {len(selected)} samples")
        print(f"  Error range: [{selected[-1]['error']:.6f}, {selected[0]['error']:.6f}]")
        
        return selected
    
    def append_to_dataset(self, samples, split='train'):
        # Append samples to existing dataset (train split only)
        # Continues file numbering from existing samples
        
        print(f"\n{'='*60}")
        print(f"Appending {len(samples)} samples to {split} split...")
        print(f"{'='*60}")
        
        split_dir = self.data_dir / split
        
        # Get current file count to continue numbering
        mask_dir = split_dir / 'masks'
        existing_files = list(mask_dir.glob('*.png'))
        start_id = len(existing_files)
        
        print(f"  Existing samples: {start_id}")
        print(f"  New start ID: {start_id:06d}")
        
        # Save each sample
        for i, sample in enumerate(tqdm(samples, desc="Saving samples")):
            file_id = start_id + i
            
            # Convert illumination quadrant to full for saving
            illum_full = light_sources.quadrant_to_full(sample['illumination'])
            
            # Save mask
            mask_img = Image.fromarray((sample['mask'] * 255).astype(np.uint8))
            mask_img.save(split_dir / 'masks' / f"{file_id:06d}.png")
            
            # Save illumination (full 64x64)
            illum_img = Image.fromarray((illum_full * 255).astype(np.uint8))
            illum_img.save(split_dir / 'illuminations' / f"{file_id:06d}.png")
            
            # Save intensity
            intensity_img = Image.fromarray((sample['sim_intensity'] * 255).astype(np.uint8))
            intensity_img.save(split_dir / 'intensities' / f"{file_id:06d}.png")
            
            # Save resist
            resist_img = Image.fromarray((sample['sim_resist'] * 255).astype(np.uint8))
            resist_img.save(split_dir / 'resists' / f"{file_id:06d}.png")
        
        print(f"✓ Saved {len(samples)} samples to {split_dir}")
    
    def train_initial(self, epochs=50, save_dir='./active_learning_checkpoints', 
                     batch_size=16, lr=1e-4):
        # Train initial model on base dataset for specified epochs
        
        print(f"\n{'='*60}")
        print(f"Training initial model for {epochs} epochs...")
        print(f"{'='*60}")
        
        # Load datasets
        train_dataset = LithographyDataset(self.data_dir.name, split='train')
        test_dataset = LithographyDataset(self.data_dir.name, split='test')
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        # Create fresh model
        model = LithographyUNet(base_ch=64)
        
        # Setup loss and trainer
        criterion = MultiHeadLoss(w_resist=self.w_resist, w_intensity=self.w_intensity, edge_weight=2.0)
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=4,
            device=self.device,
            lr=lr,
            save_dir=save_dir
        )
        
        # Train
        trainer.train(
            epochs=epochs,
            save_name='initial_best_model.pth',
            patience=15,
            viz_every=1,
            n_viz_samples=6
        )
        
        best_model_path = Path(save_dir) / 'initial_best_model.pth'
        
        # Initialize our model and optimizer with trained weights
        self.model = LithographyUNet(base_ch=64).to(self.device)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()
        
        self.optimizer = SourceMaskOptimizer(
            modelClass=LithographyUNet,
            modelPath=str(best_model_path),
            device=self.device
        )
        
        print(f"\n✓ Initial training complete")
        print(f"  Best model: {best_model_path}")
        
        return best_model_path
    
    def train_iteration(self, checkpoint_path, epochs=5, save_dir='./active_learning_checkpoints', 
                       batch_size=16, lr=1e-4, al_iter=0):
        # Train model for specified epochs, resuming from checkpoint
        # Loads weights only (not full checkpoint) to avoid low LR issues
        
        print(f"\n{'='*60}")
        print(f"Training for {epochs} epochs (AL iteration {al_iter})...")
        print(f"{'='*60}")
        
        # Reload datasets (now includes new samples)
        train_dataset = LithographyDataset(self.data_dir.name, split='train')
        test_dataset = LithographyDataset(self.data_dir.name, split='test')
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        # Create new model and load weights
        model = LithographyUNet(base_ch=64)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        # Setup loss and trainer
        criterion = MultiHeadLoss(w_resist=self.w_resist, w_intensity=self.w_intensity, edge_weight=2.0)
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=4,
            device=self.device,
            lr=lr,
            save_dir=save_dir
        )
        
        # Train for specified epochs
        save_name = f'best_model_AL{al_iter:02d}.pth'
        trainer.train(
            epochs=epochs,
            save_name=save_name,
            patience=epochs + 10,  # Don't early stop during AL
            viz_every=epochs + 1,  # No viz during AL
            n_viz_samples=0
        )
        
        # Return path to final checkpoint (for next iteration)
        final_checkpoint = Path(save_dir) / f'checkpoint_epoch_{epochs}.pth'
        
        # Update our model for next iteration
        self.model.load_state_dict(torch.load(Path(save_dir) / save_name, map_location=self.device))
        self.model.eval()
        
        # Update optimizer's model too
        self.optimizer.model.load_state_dict(torch.load(Path(save_dir) / save_name, map_location=self.device))
        self.optimizer.model.eval()
        
        return Path(save_dir) / save_name  # Return best model path, not checkpoint
    
    def run(self, 
            initial_epochs=50,
            n_iterations=50,
            masks_per_iter=100,
            samples_per_mask=20,
            top_k=1000,
            epochs_per_iter=5,
            save_dir='./active_learning_checkpoints',
            log_file='active_learning_log.csv',
            **opt_kwargs):
        # Run the complete active learning pipeline:
        # 1. Train initial model for initial_epochs
        # 2. Run n_iterations of active learning
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Step 0: Train initial model
        print(f"\n{'#'*60}")
        print(f"# STEP 0: Initial Training")
        print(f"{'#'*60}\n")
        
        current_checkpoint = self.train_initial(
            epochs=initial_epochs,
            save_dir=save_dir
        )
        
        # Initialize log file
        log_path = save_dir / log_file
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'train_samples', 'mean_error', 'median_error', 
                'max_error', 'top_k_mean', 'checkpoint'
            ])
        
        # Main active learning loop
        for al_iter in range(n_iterations):
            print(f"\n{'#'*60}")
            print(f"# Active Learning Iteration {al_iter+1}/{n_iterations}")
            print(f"{'#'*60}\n")
            
            # Step 1: Optimize batch of random masks
            opt_results = self.optimize_batch(
                n_masks=masks_per_iter,
                **opt_kwargs
            )
            
            # Step 2: Sample from histories
            samples = self.sample_from_histories(
                opt_results,
                samples_per_mask=samples_per_mask
            )
            
            # Step 3: Simulate and rank by error
            ranked_errors = self.simulate_and_rank(samples)
            
            # Step 4: Select worst predictions
            selected = self.select_top_errors(ranked_errors, top_k=top_k)
            
            # Step 5: Append to training dataset
            self.append_to_dataset(selected, split='train')
            
            # Step 6: Train for more epochs             
            current_checkpoint = self.train_iteration(
                checkpoint_path=current_checkpoint,
                epochs=epochs_per_iter,
                save_dir=save_dir,
                al_iter=al_iter
            )
            
            # Log iteration stats
            train_dataset = LithographyDataset(self.data_dir.name, split='train')
            all_errors = [e['error'] for e in ranked_errors]
            top_k_errors = [e['error'] for e in selected]
            
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    al_iter,
                    len(train_dataset),
                    np.mean(all_errors),
                    np.median(all_errors),
                    np.max(all_errors),
                    np.mean(top_k_errors),
                    str(current_checkpoint)
                ])
            
            print(f"\n✓ AL Iteration {al_iter+1} complete")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Current checkpoint: {current_checkpoint}")
        
        print(f"\n{'#'*60}")
        print(f"# Active Learning Complete!")
        print(f"{'#'*60}")
        print(f"  Initial training: {initial_epochs} epochs")
        print(f"  AL iterations: {n_iterations}")
        print(f"  Final training samples: {len(train_dataset)}")
        print(f"  Final checkpoint: {current_checkpoint}")
        print(f"  Log saved to: {log_path}")
        
        # Save final best model
        final_best = save_dir / 'final_best_model.pth'
        torch.save(self.model.state_dict(), final_best)
        print(f"  Final best model: {final_best}")


def main(data_dir):
    # Example usage of ActiveLearner
    # Everything is self-contained - no need to train separately
    
    # Load simulator config
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    
    # Initialize active learner (no pre-trained model needed)
    learner = ActiveLearner(
        data_dir=data_dir,
        sim_config=sim_config,
        device='cuda',
        w_resist=1.0,
        w_intensity=3.0
    )
    
    # Run complete pipeline:
    # - Train initial model for 50 epochs
    # - Run 50 AL iterations (5 epochs each)
    # This will grow dataset from ~50k to ~100k samples
    
    
    # learner.run(
    #     initial_epochs=50,
    #     n_iterations=50,
    #     masks_per_iter=100,
    #     samples_per_mask=10,
    #     top_k=200,
    #     epochs_per_iter=5,
    #     save_dir='./active_learning_checkpoints',
    #     log_file='active_learning_log.csv',
    #     # Optimizer kwargs
    #     num_iterations=1500,
    #     lr_mask=0.2,
    #     lr_illum=0.1,
    #     initial_blur_mask=10.0,
    #     final_blur_mask=0.5,
    #     blur_illum=1.0,
    #     binarize_final=True,
    #     binary_iterations=100,
    #     illum_entropy_weight=0.05,
    #     illum_tv_weight=0.02
    # )


    learner.run(
        initial_epochs=50,
        n_iterations=50,
        masks_per_iter=100,
        samples_per_mask=10,
        top_k=1000,
        epochs_per_iter=5,
        save_dir='./active_learning_checkpoints',
        log_file='active_learning_log.csv',
        # Optimizer kwargs
        num_iterations=1500,
        lr_mask=0.2,
        lr_illum=0.1,
        initial_blur_mask=10.0,
        final_blur_mask=0.5,
        blur_illum=1.0,
        binarize_final=True,
        binary_iterations=100,
        illum_entropy_weight=0,
        illum_tv_weight=0
    )


if __name__ == "__main__":
    main()