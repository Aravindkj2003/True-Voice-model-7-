"""
Training logic for Deepfake Audio Detection System
Two-stage training: Warmup (frozen backbone) and Fine-tuning (unfrozen backend)
"""

import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm


class Trainer:
    """Trainer class for two-stage training"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None,
                 warmup_epochs=5, total_epochs=50,
                 warmup_lr=1e-4, finetune_lr=2e-5,
                 weight_decay=1e-4,
                 class_weights=None, device='cuda',
                 checkpoint_dir='./checkpoints'):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader (optional)
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            warmup_lr: Learning rate for warmup stage
            finetune_lr: Learning rate for fine-tuning stage
            weight_decay: Weight decay for optimizer
            class_weights: Class weights for weighted loss
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay
        
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training state
        self.current_epoch = 0
        self.best_val_eer = float('inf')
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_eer': [],
            'epoch': []
        }
        
        # Optimizers (will be created in setup_optimizers)
        self.warmup_optimizer = None
        self.finetune_optimizer = None
    
    def setup_optimizers(self):
        """Create optimizers for both stages"""
        
        # Warmup optimizer: only update classifier head
        self.warmup_optimizer = torch.optim.AdamW(
            self.model.backbone.fc.parameters(),
            lr=self.warmup_lr,
            weight_decay=self.weight_decay
        )
        
        # Fine-tuning optimizer: update all parameters
        self.finetune_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.finetune_lr,
            weight_decay=self.weight_decay
        )
    
    def get_current_optimizer(self):
        """Get the appropriate optimizer based on current epoch"""
        if self.current_epoch <= self.warmup_epochs:
            return self.warmup_optimizer
        else:
            return self.finetune_optimizer
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        optimizer = self.get_current_optimizer()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.total_epochs} (Train)')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.4f}'})
        
        train_loss = total_loss / len(self.train_loader)
        train_acc = correct / total
        
        return train_loss, train_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch}/{self.total_epochs} (Val)')
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """Complete two-stage training loop"""
        self.setup_optimizers()
        
        print("\n" + "="*80)
        print(f"Starting training... Total epochs: {self.total_epochs}")
        print(f"Warmup epochs: {self.warmup_epochs} (LR={self.warmup_lr})")
        print(f"Fine-tune epochs: {self.total_epochs - self.warmup_epochs} (LR={self.finetune_lr})")
        print("="*80 + "\n")
        
        for epoch in range(1, self.total_epochs + 1):
            self.current_epoch = epoch
            
            # Print stage information
            if epoch == 1:
                print(f"\n--- STAGE 1: Warmup (Epochs 1-{self.warmup_epochs}) ---")
                print("Backbone: FROZEN | Only training classification head")
                print()
            elif epoch == self.warmup_epochs + 1:
                print(f"\n--- STAGE 2: Fine-tuning (Epochs {self.warmup_epochs + 1}-{self.total_epochs}) ---")
                print("Backbone: UNFROZEN | Training entire network with low learning rate")
                print()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Save history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{self.total_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Print stage transition info
            if epoch == self.warmup_epochs:
                print("\n" + "="*80)
                print("Warmup complete! Unfreezing backbone and switching to fine-tuning...")
                self.model.unfreeze_all()
                print(f"Trainable parameters: {self.model.get_trainable_parameters()} / {self.model.get_total_parameters()}")
                print("="*80)
        
        print("\n" + "="*80)
        print("Training complete!")
        print("="*80 + "\n")
        
        return self.training_history
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'warmup_optimizer_state_dict': self.warmup_optimizer.state_dict() if self.warmup_optimizer else None,
            'finetune_optimizer_state_dict': self.finetune_optimizer.state_dict() if self.finetune_optimizer else None,
            'val_loss': val_loss,
            'training_history': self.training_history,
        }
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  [Best model saved: {best_path}]")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    def save_training_summary(self, save_path='./training_summary.json'):
        """Save training history to JSON"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        print(f"Training summary saved to {save_path}")
