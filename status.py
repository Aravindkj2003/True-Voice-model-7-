"""
Training Startup Script - Ready to Train!
Check status and start training with GPU
"""

import torch
from pathlib import Path
import sys

def show_status():
    """Show current training status"""
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION SYSTEM - TRAINING READY")
    print("="*70)
    
    # GPU Status
    print("\n✅ GPU Status:")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA: {torch.version.cuda}")
    else:
        print("   ❌ GPU NOT AVAILABLE - Training will use CPU (very slow)")
        return False
    
    # Dataset Status
    print("\n✅ Dataset Status:")
    data_dir = Path("./data")
    
    if data_dir.exists():
        train_real = len(list((data_dir / "train" / "real").glob("*")))
        train_fake = len(list((data_dir / "train" / "fake").glob("*")))
        val_real = len(list((data_dir / "val" / "real").glob("*")))
        val_fake = len(list((data_dir / "val" / "fake").glob("*")))
        test_real = len(list((data_dir / "test" / "real").glob("*")))
        test_fake = len(list((data_dir / "test" / "fake").glob("*")))
        
        print(f"   Training:   {train_real + train_fake:,} samples ({train_real} real, {train_fake} fake)")
        print(f"   Validation: {val_real + val_fake:,} samples ({val_real} real, {val_fake} fake)")
        print(f"   Test:       {test_real + test_fake:,} samples ({test_real} real, {test_fake} fake)")
        print(f"   Total:      {train_real + train_fake + val_real + val_fake + test_real + test_fake:,} samples")
    else:
        print("   ❌ Dataset folder not found!")
        return False
    
    # Model Status
    print("\n✅ Model Configuration:")
    print("   Architecture: ResNet-18 (11.2M parameters)")
    print("   Backbone: Frozen (Stage 1: Epochs 1-5)")
    print("   Optimizer: AdamW")
    print("   Loss: Weighted CrossEntropyLoss")
    
    # Training Configuration
    print("\n✅ Training Configuration:")
    print("   Total Epochs: 50")
    print("   Batch Size: 32")
    print("   Learning Rates:")
    print("      Warmup (frozen): 1×10⁻⁴")
    print("      Fine-tune (unfrozen): 2×10⁻⁵")
    print("   Data Augmentation: SpecAugment")
    
    # Expected Results
    print("\n✅ Expected Results:")
    print("   Training Time: ~2-4 hours on RTX 3050")
    print("   Expected Accuracy: 85-95%")
    print("   Expected EER: <10%")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY FOR TRAINING!")
    print("="*70)
    print("\n🚀 NEXT STEP: python train.py\n")
    
    return True


if __name__ == '__main__':
    try:
        success = show_status()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
