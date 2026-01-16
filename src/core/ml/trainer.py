from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.core.ml.models import LithographyDataset, LithographyUNet
from src.core.ml.losses import MultiHeadLoss
import src.visualizers.ml.trainer_visualizer as trainer_visualizer


def compute_mae(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()


def compute_rmse(pred, target):
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()


class Trainer:
    def __init__(self, model, criterion, train_dataset, test_dataset, 
                 batch_size=16, num_workers=4, device='cuda', lr=1e-4, 
                 save_dir='./checkpoints', seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = model
        self.criterion = criterion
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        self.max_grad_norm = 1.0
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'lr': [],
            'intensity_mae': [],
            'intensity_rmse': [],
            'intensity_psnr': [],
            'resist_mae': [],
            'resist_rmse': [],
            'resist_psnr': [],
            'grad_norm': []
        }

        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def move_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return tuple(x.to(self.device) for x in batch)
        return batch.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        grad_norms = []
        
        # Diagnostic accumulators
        intensity_pred_stats = []
        resist_pred_stats = []

        for batch in tqdm(self.train_loader, desc='Train'):
            batch = self.move_to_device(batch)
            mask, illum_q, target_int, target_res = batch

            self.optimizer.zero_grad()
            pred_int, pred_res = self.model(mask, illum_q)
            loss = self.criterion(pred_int, pred_res, target_int, target_res)

            loss.backward()
            
            # Compute gradient norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            grad_norms.append(total_norm.item())
            
            self.optimizer.step()

            total_loss += loss.item()
            
            # Collect prediction statistics
            intensity_pred_stats.append({
                'min': pred_int.min().item(),
                'max': pred_int.max().item(),
                'mean': pred_int.mean().item(),
                'std': pred_int.std().item()
            })
            resist_pred_stats.append({
                'min': pred_res.min().item(),
                'max': pred_res.max().item(),
                'mean': pred_res.mean().item(),
                'std': pred_res.std().item()
            })

        avg_grad_norm = np.mean(grad_norms)
        
        # Aggregate statistics
        intensity_stats = {
            'min': np.mean([s['min'] for s in intensity_pred_stats]),
            'max': np.mean([s['max'] for s in intensity_pred_stats]),
            'mean': np.mean([s['mean'] for s in intensity_pred_stats]),
            'std': np.mean([s['std'] for s in intensity_pred_stats])
        }
        resist_stats = {
            'min': np.mean([s['min'] for s in resist_pred_stats]),
            'max': np.mean([s['max'] for s in resist_pred_stats]),
            'mean': np.mean([s['mean'] for s in resist_pred_stats]),
            'std': np.mean([s['std'] for s in resist_pred_stats])
        }
        
        return total_loss / len(self.train_loader), avg_grad_norm, intensity_stats, resist_stats

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        
        # Metrics accumulators
        intensity_mae_sum = 0.0
        intensity_rmse_sum = 0.0
        intensity_psnr_sum = 0.0
        resist_mae_sum = 0.0
        resist_rmse_sum = 0.0
        resist_psnr_sum = 0.0
        
        # Diagnostic accumulators
        intensity_pred_stats = []
        resist_pred_stats = []
        intensity_target_stats = []
        resist_target_stats = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Eval'):
                batch = self.move_to_device(batch)
                mask, illum_q, target_int, target_res = batch

                pred_int, pred_res = self.model(mask, illum_q)
                loss = self.criterion(pred_int, pred_res, target_int, target_res)
                total_loss += loss.item()
                
                # Compute metrics
                intensity_mae_sum += compute_mae(pred_int, target_int)
                intensity_rmse_sum += compute_rmse(pred_int, target_int)
                intensity_psnr_sum += compute_psnr(pred_int, target_int)
                
                resist_mae_sum += compute_mae(pred_res, target_res)
                resist_rmse_sum += compute_rmse(pred_res, target_res)
                resist_psnr_sum += compute_psnr(pred_res, target_res)
                
                # Collect prediction statistics
                intensity_pred_stats.append({
                    'min': pred_int.min().item(),
                    'max': pred_int.max().item(),
                    'mean': pred_int.mean().item()
                })
                resist_pred_stats.append({
                    'min': pred_res.min().item(),
                    'max': pred_res.max().item(),
                    'mean': pred_res.mean().item()
                })
                intensity_target_stats.append({
                    'min': target_int.min().item(),
                    'max': target_int.max().item(),
                    'mean': target_int.mean().item()
                })
                resist_target_stats.append({
                    'min': target_res.min().item(),
                    'max': target_res.max().item(),
                    'mean': target_res.mean().item()
                })

        n_batches = len(self.test_loader)
        metrics = {
            'loss': total_loss / n_batches,
            'intensity_mae': intensity_mae_sum / n_batches,
            'intensity_rmse': intensity_rmse_sum / n_batches,
            'intensity_psnr': intensity_psnr_sum / n_batches,
            'resist_mae': resist_mae_sum / n_batches,
            'resist_rmse': resist_rmse_sum / n_batches,
            'resist_psnr': resist_psnr_sum / n_batches,
            'intensity_pred_stats': {
                'min': np.mean([s['min'] for s in intensity_pred_stats]),
                'max': np.mean([s['max'] for s in intensity_pred_stats]),
                'mean': np.mean([s['mean'] for s in intensity_pred_stats])
            },
            'resist_pred_stats': {
                'min': np.mean([s['min'] for s in resist_pred_stats]),
                'max': np.mean([s['max'] for s in resist_pred_stats]),
                'mean': np.mean([s['mean'] for s in resist_pred_stats])
            },
            'intensity_target_stats': {
                'min': np.mean([s['min'] for s in intensity_target_stats]),
                'max': np.mean([s['max'] for s in intensity_target_stats]),
                'mean': np.mean([s['mean'] for s in intensity_target_stats])
            },
            'resist_target_stats': {
                'min': np.mean([s['min'] for s in resist_target_stats]),
                'max': np.mean([s['max'] for s in resist_target_stats]),
                'mean': np.mean([s['mean'] for s in resist_target_stats])
            }
        }
        
        return metrics

    def save_checkpoint(self, path, epoch, best_loss, patience_counter):
        """Save complete training state for resuming"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'history': self.history,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        torch.save(checkpoint, path)
        print(f"  ✓ Saved full checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load complete training state for resuming"""
        print(f"\nLoading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        # Restore random states for reproducibility
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        if checkpoint['cuda_random_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint['patience_counter']
        
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best loss so far: {best_loss:.6f}")
        print(f"Patience counter: {patience_counter}")
        
        return start_epoch, best_loss, patience_counter

    def train(self, epochs=50, save_name='best_model.pth', patience=15, 
              viz_every=1, n_viz_samples=6, resume_from=None):
        """
        Train the model.
        
        Args:
            epochs: Total number of epochs to train (not additional epochs when resuming)
            save_name: Name for the best model checkpoint
            patience: Early stopping patience
            viz_every: Frequency of visualization generation
            n_viz_samples: Number of samples to visualize
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_epoch = 0
        best_loss = float('inf')
        patience_counter = 0
        
        # Load checkpoint if resuming
        if resume_from is not None:
            start_epoch, best_loss, patience_counter = self.load_checkpoint(resume_from)
        
        print(f"\nTraining from epoch {start_epoch+1} to {epochs}...")
        
        # Create visualization directory
        viz_dir = self.save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True, parents=True)

        for epoch in range(start_epoch, epochs):
            train_loss, grad_norm, train_int_stats, train_res_stats = self.train_epoch()
            metrics = self.evaluate()
            test_loss = metrics['loss']

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(lr)
            self.history['grad_norm'].append(grad_norm)
            
            # Store metrics
            self.history['intensity_mae'].append(metrics['intensity_mae'])
            self.history['intensity_rmse'].append(metrics['intensity_rmse'])
            self.history['intensity_psnr'].append(metrics['intensity_psnr'])
            self.history['resist_mae'].append(metrics['resist_mae'])
            self.history['resist_rmse'].append(metrics['resist_rmse'])
            self.history['resist_psnr'].append(metrics['resist_psnr'])

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Loss - Train: {train_loss:.6f} | Test: {test_loss:.6f}")
            print(f"  Intensity - MAE: {metrics['intensity_mae']:.4f} | RMSE: {metrics['intensity_rmse']:.4f} | PSNR: {metrics['intensity_psnr']:.2f} dB")
            print(f"  Resist    - MAE: {metrics['resist_mae']:.4f} | RMSE: {metrics['resist_rmse']:.4f} | PSNR: {metrics['resist_psnr']:.2f} dB")
            print(f"  LR: {lr:.2e} | Grad Norm: {grad_norm:.4f}")
            
            # DIAGNOSTIC OUTPUT
            print(f"\n  [DIAGNOSTIC] Train Predictions:")
            print(f"    Intensity: min={train_int_stats['min']:.4f}, max={train_int_stats['max']:.4f}, mean={train_int_stats['mean']:.4f}, std={train_int_stats['std']:.4f}")
            print(f"    Resist:    min={train_res_stats['min']:.4f}, max={train_res_stats['max']:.4f}, mean={train_res_stats['mean']:.4f}, std={train_res_stats['std']:.4f}")
            
            print(f"  [DIAGNOSTIC] Test Predictions vs Targets:")
            print(f"    Intensity Pred:   min={metrics['intensity_pred_stats']['min']:.4f}, max={metrics['intensity_pred_stats']['max']:.4f}, mean={metrics['intensity_pred_stats']['mean']:.4f}")
            print(f"    Intensity Target: min={metrics['intensity_target_stats']['min']:.4f}, max={metrics['intensity_target_stats']['max']:.4f}, mean={metrics['intensity_target_stats']['mean']:.4f}")
            print(f"    Resist Pred:      min={metrics['resist_pred_stats']['min']:.4f}, max={metrics['resist_pred_stats']['max']:.4f}, mean={metrics['resist_pred_stats']['mean']:.4f}")
            print(f"    Resist Target:    min={metrics['resist_target_stats']['min']:.4f}, max={metrics['resist_target_stats']['max']:.4f}, mean={metrics['resist_target_stats']['mean']:.4f}")

            self.scheduler.step(test_loss)

            # Always save checkpoint for current epoch (for exact reproducibility)
            self.save_checkpoint(
                self.save_dir / f'checkpoint_epoch_{epoch+1}.pth',
                epoch, best_loss, patience_counter
            )
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                # Save best model weights separately
                torch.save(self.model.state_dict(), self.save_dir / save_name)
                print(f"  ✓ New best model saved")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered")
                    break
            
            # Generate prediction visualizations
            if (epoch + 1) % viz_every == 0:
                print(f"  Generating predictions...")
                epoch_viz_dir = viz_dir / f'epoch_{epoch+1:03d}'
                epoch_viz_dir.mkdir(exist_ok=True, parents=True)
                
                trainer_visualizer.plot_predictions(
                    self.model, 
                    self.test_loader.dataset, 
                    device=self.device, 
                    n=n_viz_samples, 
                    save_dir=epoch_viz_dir, 
                    show=False
                )
                
                # Also update training history plot
                trainer_visualizer.plot_training_history(
                    self.history, 
                    save_dir=viz_dir, 
                    show=False
                )

        print(f"\nTraining complete! Best test loss: {best_loss:.6f}")

    def load(self, path):
        """Load only model weights (for inference)"""
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded model weights from {path}")


if __name__ == "__main__":
    data_dir = 'augmented_massive'
    train_dataset = LithographyDataset(data_dir, split='train')
    test_dataset = LithographyDataset(data_dir, split='test')

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Diagnostic: Check data ranges
    print("\nChecking data ranges...")
    sample_mask, sample_illum, sample_int, sample_res = train_dataset[0]
    print(f"Mask range: [{sample_mask.min():.4f}, {sample_mask.max():.4f}]")
    print(f"Illumination range: [{sample_illum.min():.4f}, {sample_illum.max():.4f}]")
    print(f"Intensity range: [{sample_int.min():.4f}, {sample_int.max():.4f}]")
    print(f"Resist range: [{sample_res.min():.4f}, {sample_res.max():.4f}]")

    model = LithographyUNet(base_ch=64)
    w_resist = 1.0
    w_intensity = 3.0
    edge_weight = 2.0

    criterion = MultiHeadLoss(w_resist=w_resist, w_intensity=w_intensity, edge_weight=edge_weight)
    
    batch_size = 64
    num_workers = 4
    lr = 1e-4
    device = 'cuda'
    save_dir = './checkpoints'
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        lr=lr,
        save_dir=save_dir
    )

    # OPTION 1: Train for 100 epochs all at once
    trainer.train(epochs=100, save_name='best_model.pth', viz_every=1, n_viz_samples=6)
    
    # OPTION 2: Train in two stages (50 + 50)
    # Stage 1: First 50 epochs
    # print("\n" + "="*60)
    # print("STAGE 1: Training for first 50 epochs")
    # print("="*60)
    # trainer.train(epochs=50, save_name='best_model.pth', viz_every=1, n_viz_samples=6)
    
    # print("\n" + "="*60)
    # print("STAGE 2: Resuming and training to 100 epochs")
    # print("="*60)
    # trainer.train(
    #     epochs=100, 
    #     save_name='best_model.pth', 
    #     viz_every=1, 
    #     n_viz_samples=6,
    #     resume_from='./checkpoints/checkpoint_epoch_50.pth'  # Adjust to actual checkpoint name
    # )
    
    # Final visualizations
    print("\nGenerating final visualizations...")
    final_viz_dir = Path(save_dir) / 'visualizations' / 'final'
    final_viz_dir.mkdir(exist_ok=True, parents=True)
    
    trainer_visualizer.plot_training_history(trainer.history, save_dir=Path(save_dir)/'visualizations', show=True)
    trainer_visualizer.plot_predictions(model, test_dataset, device=device, n=6, save_dir=final_viz_dir, show=True)