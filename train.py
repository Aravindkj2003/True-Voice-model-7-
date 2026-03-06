"""
Main training script for Deepfake Audio Detection System
"""

import torch
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    DATASET_CONFIG, TRAINING_CONFIG, DATA_SPLIT, MODEL_CONFIG,
    IMAGENET_STATS, AUGMENTATION_CONFIG, EVAL_CONFIG, PATHS, CLASS_LABELS
)
from src.model import get_model
from src.data_loader import DataManager
from src.trainer import Trainer
from src.utils import (
    create_data_manifest, set_seed, get_device,
    print_model_info, print_training_config, setup_directories,
    calculate_class_weights
)


def main(args):
    """Main training function"""
    
    # Setup
    print("\n" + "="*80)
    print("DEEPFAKE AUDIO DETECTION SYSTEM - TRAINING")
    print("="*80 + "\n")
    
    # Set seed for reproducibility
    set_seed(EVAL_CONFIG['seed'])
    
    # Create directories
    setup_directories([PATHS['model_dir'], PATHS['log_dir'], PATHS['result_dir']])
    
    # Get device
    device = get_device()
    
    # Print configuration
    full_config = {
        'dataset': DATASET_CONFIG,
        'training': TRAINING_CONFIG,
        'data_split': DATA_SPLIT,
        'model': MODEL_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
    }
    print_training_config(full_config)
    
    # Create/Load data manifest
    manifest_path = os.path.join(PATHS['data_dir'], 'manifest.json')
    if os.path.exists(manifest_path):
        print(f"Loading dataset manifest from {manifest_path}...")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        print(f"Creating dataset manifest from {PATHS['data_dir']}...")
        if not os.path.exists(PATHS['data_dir']):
            print(f"Error: Dataset directory not found at {PATHS['data_dir']}")
            print("\nExpected directory structure:")
            print("data/")
            print("  ├── train/")
            print("  │   ├── real/*.wav")
            print("  │   └── fake/*.wav")
            print("  ├── val/")
            print("  │   ├── real/*.wav")
            print("  │   └── fake/*.wav")
            print("  └── test/")
            print("      ├── real/*.wav")
            print("      └── fake/*.wav")
            return
        
        manifest = create_data_manifest(PATHS['data_dir'], manifest_path)
    
    # Initialize data manager
    data_manager = DataManager(
        data_dir=PATHS['data_dir'],
        sample_rate=DATASET_CONFIG['sample_rate'],
        batch_size=TRAINING_CONFIG['batch_size'],
        num_workers=TRAINING_CONFIG['num_workers'],
        seed=EVAL_CONFIG['seed']
    )
    
    # Load datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    
    if args.use_manifest:
        train_loader = data_manager.load_dataset_from_manifest(manifest_path, split='train')
        val_loader = data_manager.load_dataset_from_manifest(manifest_path, split='val')
        test_loader = data_manager.load_dataset_from_manifest(manifest_path, split='test')
    else:
        train_loader = data_manager.load_from_directory_structure(split='train')
        val_loader = data_manager.load_from_directory_structure(split='val')
        test_loader = data_manager.load_from_directory_structure(split='test')
    
    # Get class weights from training data
    train_labels = []
    if 'train' in manifest:
        train_labels = [s['label'] for s in manifest['train']]
    
    class_weights = calculate_class_weights(train_labels) if train_labels else None
    
    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80 + "\n")
    
    model = get_model(
        num_classes=MODEL_CONFIG['num_classes'],
        backbone=MODEL_CONFIG['backbone'],
        pretrained=MODEL_CONFIG['pretrained'],
        device=device,
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )
    
    print_model_info(model)
    
    # Initialize trainer
    checkpoint_dir = os.path.join(PATHS['model_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        warmup_epochs=TRAINING_CONFIG['warmup_epochs'],
        total_epochs=TRAINING_CONFIG['total_epochs'],
        warmup_lr=TRAINING_CONFIG['warmup_lr'],
        finetune_lr=TRAINING_CONFIG['finetune_lr'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        class_weights=class_weights,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Start training
    print("Starting training...")
    history = trainer.train()
    
    # Save training summary
    summary_path = os.path.join(PATHS['result_dir'], 'training_summary.json')
    trainer.save_training_summary(summary_path)
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best model: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"Training summary: {summary_path}")
    
    # Test on test set
    if test_loader and args.test_after_training:
        print("\n" + "="*80)
        print("Evaluating on test set...")
        print("="*80 + "\n")
        
        # Load best model
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            trainer.load_checkpoint(best_model_path)
        
        test_loss, test_acc = trainer.validate()
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc:  {test_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deepfake Audio Detection Model')
    
    parser.add_argument('--use-manifest', action='store_true', default=False,
                       help='Use manifest file for data loading')
    parser.add_argument('--test-after-training', action='store_true', default=False,
                       help='Evaluate on test set after training')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Override device in config if specified
    if args.device == 'auto':
        TRAINING_CONFIG['device'] = get_device()
    elif args.device:
        TRAINING_CONFIG['device'] = args.device
    
    main(args)
