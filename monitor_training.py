#!/usr/bin/env python3
"""
Monitor training progress in real-time
"""

import time
import os
import json
import glob
from pathlib import Path
from datetime import datetime
import subprocess

def get_latest_checkpoint_dir():
    """Get the latest checkpoint directory"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not dirs:
        return None
    
    dirs.sort(reverse=True)
    return os.path.join(models_dir, dirs[0])

def get_checkpoint_info(checkpoint_dir):
    """Get information about the latest checkpoint"""
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None
    
    info = {}
    
    # Check latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(latest_path):
        stat = os.stat(latest_path)
        info['latest_checkpoint'] = {
            'path': latest_path,
            'size_mb': stat.st_size / (1024*1024),
            'last_modified': datetime.fromtimestamp(stat.st_mtime)
        }
    
    # Check best model
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        stat = os.stat(best_path)
        info['best_model'] = {
            'path': best_path,
            'size_mb': stat.st_size / (1024*1024),
            'last_modified': datetime.fromtimestamp(stat.st_mtime)
        }
    
    return info

def get_training_summary():
    """Get training summary if it exists"""
    summary_path = "results/training_summary.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def get_process_info():
    """Get Python process information"""
    try:
        result = subprocess.run(
            ['powershell', '-Command', 
             'Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return int(result.stdout.strip())
    except:
        return 0

def monitor_training():
    """Monitor training continuously"""
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION SYSTEM - TRAINING MONITOR")
    print("="*80)
    print("Press Ctrl+C to stop monitoring\n")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Get checkpoint directory
        checkpoint_dir = get_latest_checkpoint_dir()
        
        # Get process count
        proc_count = get_process_info()
        
        print(f"\n[Update #{iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        if proc_count == 0:
            print("❌ No Python processes running - Training may have completed or crashed")
        else:
            print(f"✅ {proc_count} Python process(es) running")
        
        # Get checkpoint info
        if checkpoint_dir:
            print(f"📁 Checkpoint directory: {checkpoint_dir}")
            info = get_checkpoint_info(checkpoint_dir)
            
            if info:
                if 'latest_checkpoint' in info:
                    cp = info['latest_checkpoint']
                    elapsed = datetime.now() - cp['last_modified']
                    print(f"   Latest checkpoint: {cp['size_mb']:.2f}MB (updated {elapsed.total_seconds():.0f}s ago)")
                
                if 'best_model' in info:
                    bm = info['best_model']
                    elapsed = datetime.now() - bm['last_modified']
                    print(f"   Best model: {bm['size_mb']:.2f}MB (updated {elapsed.total_seconds():.0f}s ago)")
        else:
            print("⌛ Initializing training...")
        
        # Get training summary
        summary = get_training_summary()
        if summary:
            print(f"\n📊 Training Progress:")
            if 'epoch' in summary:
                epochs_completed = summary['epoch'][-1]
                print(f"   Epochs completed: {epochs_completed}/50")
                
                if 'train_loss' in summary and 'val_loss' in summary:
                    train_loss = summary['train_loss'][-1]
                    val_loss = summary['val_loss'][-1]
                    train_acc = summary.get('train_acc', [None])[-1]
                    val_acc = summary.get('val_acc', [None])[-1]
                    
                    print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    if train_acc is not None and val_acc is not None:
                        print(f"   Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
                    
                    # Estimate remaining time
                    if epochs_completed > 0:
                        time_per_epoch = datetime.now() - datetime.strptime(checkpoint_dir.split('/')[-1], '%Y%m%d_%H%M%S')
                        time_per_epoch_seconds = time_per_epoch.total_seconds() / epochs_completed
                        remaining_epochs = 50 - epochs_completed
                        remaining_seconds = time_per_epoch_seconds * remaining_epochs
                        remaining_time = time_per_epoch_seconds * remaining_epochs
                        
                        print(f"   Estimated time per epoch: {time_per_epoch_seconds/60:.1f} minutes")
                        print(f"   Estimated time remaining: {remaining_seconds/3600:.1f} hours")
        else:
            print("\n⏳ Training summary not yet created (still in early epochs)")
        
        print()
        time.sleep(30)  # Wait 30 seconds before next update

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")
        print("="*80)
