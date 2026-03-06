#!/usr/bin/env python3
"""
Advanced training metrics analyzer - detects overfitting and accuracy trends
"""

import time
import os
import json
import glob
from pathlib import Path
from datetime import datetime
import subprocess
import sys

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

def check_overfitting(summary):
    """
    Analyze training data for overfitting signs
    Returns: overfitting_status, analysis details
    """
    if not summary or 'train_acc' not in summary or 'val_acc' not in summary:
        return None, None
    
    train_acc = summary['train_acc']
    val_acc = summary['val_acc']
    train_loss = summary.get('train_loss', [])
    val_loss = summary.get('val_loss', [])
    
    if len(train_acc) < 2:
        return None, None
    
    analysis = {
        'current_epoch': len(train_acc),
        'train_acc': train_acc[-1],
        'val_acc': val_acc[-1],
        'acc_gap': train_acc[-1] - val_acc[-1],  # Higher = more overfitting
        'train_loss': train_loss[-1] if train_loss else None,
        'val_loss': val_loss[-1] if val_loss else None,
        'loss_gap': val_loss[-1] - train_loss[-1] if (val_loss and train_loss) else None,
        'overfitting_status': 'NORMAL'
    }
    
    # Check for overfitting signs
    if analysis['acc_gap'] > 0.10:  # >10% gap
        analysis['overfitting_status'] = 'MODERATE OVERFITTING'
    elif analysis['acc_gap'] > 0.15:
        analysis['overfitting_status'] = 'SEVERE OVERFITTING'
    elif analysis['acc_gap'] < 0.03:
        analysis['overfitting_status'] = 'EXCELLENT (Well-generalized)'
    
    # Check loss divergence
    if analysis['loss_gap'] is not None and analysis['loss_gap'] > 0.5:
        analysis['loss_diverging'] = True
    else:
        analysis['loss_diverging'] = False
    
    # Check recent trend
    if len(train_acc) >= 3:
        recent_train = train_acc[-3:]
        recent_val = val_acc[-3:]
        
        train_improving = recent_train[-1] > recent_train[0]
        val_improving = recent_val[-1] > recent_val[0]
        
        analysis['train_trend'] = 'IMPROVING' if train_improving else 'PLATEAUING/DECLINING'
        analysis['val_trend'] = 'IMPROVING' if val_improving else 'PLATEAUING/DECLINING'
    
    return analysis, summary

def format_analysis(analysis):
    """Format analysis for display"""
    if not analysis:
        return None
    
    status_emoji = {
        'EXCELLENT (Well-generalized)': '🟢',
        'NORMAL': '🟡',
        'MODERATE OVERFITTING': '🟠',
        'SEVERE OVERFITTING': '🔴'
    }
    
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"ACCURACY & OVERFITTING ANALYSIS - Epoch {analysis['current_epoch']}")
    output.append(f"{'='*80}")
    
    output.append(f"\n📊 ACCURACY METRICS:")
    output.append(f"   Train Accuracy: {analysis['train_acc']*100:.2f}%")
    output.append(f"   Val Accuracy:   {analysis['val_acc']*100:.2f}%")
    output.append(f"   Gap:            {analysis['acc_gap']*100:.2f}% (Generalization measure)")
    
    output.append(f"\n📉 LOSS METRICS:")
    if analysis['train_loss'] is not None:
        output.append(f"   Train Loss:     {analysis['train_loss']:.4f}")
    if analysis['val_loss'] is not None:
        output.append(f"   Val Loss:       {analysis['val_loss']:.4f}")
    if analysis['loss_gap'] is not None:
        output.append(f"   Loss Gap:       {analysis['loss_gap']:.4f}")
    
    status_emoji_str = status_emoji.get(analysis['overfitting_status'], '❓')
    output.append(f"\n{status_emoji_str} OVERFITTING STATUS: {analysis['overfitting_status']}")
    
    if 'train_trend' in analysis:
        output.append(f"   Train Trend:    {analysis['train_trend']}")
    if 'val_trend' in analysis:
        output.append(f"   Val Trend:      {analysis['val_trend']}")
    
    output.append(f"\n💡 INTERPRETATION:")
    if analysis['acc_gap'] < 0.03:
        output.append(f"   ✅ Excellent generalization! Model learning well from training data.")
    elif analysis['acc_gap'] < 0.10:
        output.append(f"   ✅ Good generalization. Small gap is healthy and expected.")
    elif analysis['acc_gap'] < 0.15:
        output.append(f"   ⚠️  Moderate overfitting detected. Consider:")
        output.append(f"       - Increasing regularization (SpecAugment, dropout)")
        output.append(f"       - Reducing learning rate")
        output.append(f"       - Early stopping if gap continues to increase")
    else:
        output.append(f"   ❌ Severe overfitting detected!")
        output.append(f"       - Training acc >> validation acc")
        output.append(f"       - May need to stop training")
    
    if analysis['loss_diverging']:
        output.append(f"   ⚠️  Validation loss diverging from training loss - sign of overfitting")
    
    return "\n".join(output)

def analyze_continuous():
    """Continuously analyze training metrics"""
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION - ACCURACY & OVERFITTING ANALYZER")
    print("="*80)
    print("Press Ctrl+C to stop\n")
    
    last_epoch = 0
    while True:
        summary = get_training_summary()
        
        if summary and 'train_acc' in summary:
            current_epoch = len(summary['train_acc'])
            
            # Only print when a new epoch completes
            if current_epoch > last_epoch:
                analysis, _ = check_overfitting(summary)
                
                if analysis:
                    print(format_analysis(analysis))
                    last_epoch = current_epoch
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏳ Waiting for training metrics...")
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == '__main__':
    try:
        analyze_continuous()
    except KeyboardInterrupt:
        print("\n\n✋ Analysis stopped")
        print("="*80)
