#!/usr/bin/env python3
"""
Extract and analyze metrics from training checkpoints
"""

import torch
import os
import time
from datetime import datetime
from pathlib import Path

def get_latest_checkpoint():
    """Get the latest checkpoint path"""
    checkpoint_dir = Path("models")
    if not checkpoint_dir.exists():
        return None
    
    # Find latest checkpoint directory
    model_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        return None
    
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    
    # Get latest checkpoint
    latest_checkpoint = latest_dir / "latest_checkpoint.pt"
    if latest_checkpoint.exists():
        return latest_checkpoint
    
    return None

def extract_metrics(checkpoint_path):
    """Extract metrics from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'training_history' not in checkpoint:
            return None
        
        history = checkpoint['training_history']
        
        if not history.get('epoch'):
            return None
        
        current_epoch = len(history['epoch'])
        
        metrics = {
            'epoch': current_epoch,
            'train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
            'val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
            'train_acc': history['train_acc'][-1] if history.get('train_acc') else None,
            'val_acc': history['val_acc'][-1] if history.get('val_acc') else None,
            'history': history
        }
        
        return metrics
    except Exception as e:
        return None

def analyze_overfitting(metrics):
    """Analyze overfitting from metrics"""
    if not metrics:
        return None
    
    train_acc = metrics['train_acc']
    val_acc = metrics['val_acc']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    
    acc_gap = train_acc - val_acc
    
    analysis = {
        'train_acc_pct': train_acc * 100,
        'val_acc_pct': val_acc * 100,
        'acc_gap_pct': acc_gap * 100,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'loss_gap': val_loss - train_loss,
    }
    
    # Determine overfitting status
    if acc_gap < 0.03:
        analysis['status'] = 'EXCELLENT'
        analysis['status_emoji'] = '🟢'
    elif acc_gap < 0.10:
        analysis['status'] = 'NORMAL'
        analysis['status_emoji'] = '🟡'
    elif acc_gap < 0.15:
        analysis['status'] = 'MODERATE OVERFITTING'
        analysis['status_emoji'] = '🟠'
    else:
        analysis['status'] = 'SEVERE OVERFITTING'
        analysis['status_emoji'] = '🔴'
    
    return analysis

def print_metrics(metrics, analysis):
    """Print metrics nicely"""
    if not metrics:
        return
    
    print("\n" + "="*80)
    print(f"EPOCH {metrics['epoch']} RESULTS - {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    
    print(f"\n📊 ACCURACY:")
    print(f"   Train Accuracy: {analysis['train_acc_pct']:6.2f}%")
    print(f"   Val Accuracy:   {analysis['val_acc_pct']:6.2f}%")
    print(f"   Gap:            {analysis['acc_gap_pct']:6.2f}%")
    
    print(f"\n📉 LOSS:")
    print(f"   Train Loss:     {analysis['train_loss']:.4f}")
    print(f"   Val Loss:       {analysis['val_loss']:.4f}")
    print(f"   Loss Gap:       {analysis['loss_gap']:.4f}")
    
    status_emoji = analysis['status_emoji']
    status = analysis['status']
    print(f"\n{status_emoji} STATUS: {status}")
    
    # Provide insights
    print(f"\n💡 INSIGHTS:")
    if analysis['acc_gap_pct'] < 3:
        print(f"   ✅ Excellent generalization! Model is learning well.")
    elif analysis['acc_gap_pct'] < 10:
        print(f"   ✅ Good performance. Small gap is expected.")
    elif analysis['acc_gap_pct'] < 15:
        print(f"   ⚠️  Moderate overfitting. Model memorizing training data.")
        print(f"   → Consider: Increase dropout, SpecAugment, or reduce LR")
    else:
        print(f"   ❌ Severe overfitting! Training acc >> validation acc")
        print(f"   → Consider: Stop training, increase regularization")
    
    if analysis['loss_gap'] > 0.5:
        print(f"   ⚠️  Validation loss much higher than training loss")

def continuous_monitor():
    """Continuously monitor metrics"""
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION - CHECKPOINT METRICS ANALYZER")
    print("="*80)
    print("Extracting metrics from training checkpoints...\n")
    
    last_epoch = 0
    first_run = True
    
    while True:
        checkpoint_path = get_latest_checkpoint()
        
        if checkpoint_path:
            metrics = extract_metrics(checkpoint_path)
            
            if metrics and metrics['epoch'] > last_epoch:
                if not first_run:  # Skip first print to avoid duplicate
                    analysis = analyze_overfitting(metrics)
                    print_metrics(metrics, analysis)
                    last_epoch = metrics['epoch']
                elif first_run:
                    first_run = False
                    last_epoch = metrics['epoch']
                    analysis = analyze_overfitting(metrics)
                    print_metrics(metrics, analysis)
        else:
            if first_run:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Waiting for checkpoints...")
                first_run = False
        
        time.sleep(15)  # Check every 15 seconds

if __name__ == '__main__':
    try:
        continuous_monitor()
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")
        print("="*80)
