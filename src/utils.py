"""
Utilities for Deepfake Audio Detection System
"""

import os
import json
import torch
import numpy as np
from pathlib import Path


def create_data_manifest(data_dir, manifest_path='./data_manifest.json'):
    """
    Create a manifest file from directory structure
    
    Directory structure expected:
    data/train/real/*.wav
    data/train/fake/*.wav
    data/val/real/*.wav
    data/val/fake/*.wav
    data/test/real/*.wav
    data/test/fake/*.wav
    
    Args:
        data_dir: Root data directory
        manifest_path: Where to save the manifest
    """
    manifest = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping")
            continue
        
        # Process real samples (label=0)
        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    manifest[split].append({
                        'audio_path': os.path.join(real_dir, file),
                        'label': 0
                    })
        
        # Process fake samples (label=1)
        fake_dir = os.path.join(split_dir, 'fake')
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    manifest[split].append({
                        'audio_path': os.path.join(fake_dir, file),
                        'label': 1
                    })
        
        print(f"{split}: {len(manifest[split])} samples " + 
              f"(Real: {len([x for x in manifest[split] if x['label']==0])}, " +
              f"Fake: {len([x for x in manifest[split] if x['label']==1])})")
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print(f"\nManifest saved to {manifest_path}")
    return manifest


def load_checkpoint(checkpoint_path, model, device='cuda'):
    """
    Load a checkpoint and return model state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load to
    
    Returns:
        model: Model with loaded weights
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save checkpoint with model state
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss value
        save_path: Where to save checkpoint
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def calculate_class_weights(train_labels):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        train_labels: List/array of training labels
    
    Returns:
        weights: Tensor of class weights
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    weights = len(train_labels) / (len(unique) * counts)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def print_model_info(model):
    """Print model architecture and parameter info"""
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    print(model)
    print("\n" + "="*80)
    print("Model Parameters")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("="*80 + "\n")


def print_training_config(config_dict):
    """Print training configuration"""
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(config_dict)
    print("="*80 + "\n")


def setup_directories(dirs):
    """Create necessary directories"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (cuda or cpu)"""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        print("GPU not available, using CPU")
        return 'cpu'
