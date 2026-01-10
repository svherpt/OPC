import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(self, model, criterion, train_dataset, test_dataset, 
                 batch_size=16, num_workers=4, device='cuda', lr=1e-4, 
                 save_dir='./checkpoints', seed=42):
        self._set_seed(seed)
        
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
            'lr': []
        }
        
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _move_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return tuple(x.to(self.device) for x in batch)
        return batch.to(self.device)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc='Train'):
            batch = self._move_to_device(batch)
            
            self.optimizer.zero_grad()
            
            if len(batch) == 2:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            else:
                inputs, targets_int, targets_res = batch
                pred_int, pred_res = self.model(inputs)
                loss = self.criterion(pred_int, pred_res, targets_int, targets_res)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Eval'):
                batch = self._move_to_device(batch)
                
                if len(batch) == 2:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                else:
                    inputs, targets_int, targets_res = batch
                    pred_int, pred_res = self.model(inputs)
                    loss = self.criterion(pred_int, pred_res, targets_int, targets_res)
                
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def train(self, epochs=50, save_name='best_model.pth', patience=15):
        print(f"\nTraining for {epochs} epochs...")
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(lr)
            
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | LR: {lr:.2e}")
            
            self.scheduler.step(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_dir / save_name)
                print(f"✓ Saved checkpoint")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping")
                    break
        
        print(f"\nDone! Best loss: {best_loss:.6f}")
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded {path}")