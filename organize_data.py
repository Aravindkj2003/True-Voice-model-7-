"""
Organize existing dataset into project structure
Maps from: User Desktop folder to project data directory
"""

import os
import shutil
from pathlib import Path
import random

def organize_dataset():
    """
    Copy audio files from existing paths to project data directory
    Split: 80% train, 10% val, 10% test
    """
    
    print("="*70)
    print("ORGANIZING DATASET INTO PROJECT STRUCTURE")
    print("="*70)
    
    # Source paths
    parent_dir = Path.home() / "Desktop" / "real , fake dataset"
    real_source = str(parent_dir / "real")
    fake_source = str(parent_dir / "fake")
    
    # Target base directory
    target_base = "./data"
    
    # Create target directories
    os.makedirs(f"{target_base}/train/real", exist_ok=True)
    os.makedirs(f"{target_base}/train/fake", exist_ok=True)
    os.makedirs(f"{target_base}/val/real", exist_ok=True)
    os.makedirs(f"{target_base}/val/fake", exist_ok=True)
    os.makedirs(f"{target_base}/test/real", exist_ok=True)
    os.makedirs(f"{target_base}/test/fake", exist_ok=True)
    
    print(f"\n✓ Created target directories in {target_base}/\n")
    
    # Supported audio formats
    audio_formats = ('.wav', '.mp3', '.flac', '.m4a')
    
    # Process REAL audio
    print(f"Processing REAL audio from: {real_source}")
    real_files = []
    
    if os.path.exists(real_source):
        for file in os.listdir(real_source):
            if file.lower().endswith(audio_formats):
                real_files.append(file)
    
    if real_files:
        real_files.sort()
        n = len(real_files)
        train_count = int(n * 0.8)
        val_count = int(n * 0.1)
        
        for i, file in enumerate(real_files):
            src = os.path.join(real_source, file)
            
            if i < train_count:
                dst = f"{target_base}/train/real/{file}"
                subset = "train"
            elif i < train_count + val_count:
                dst = f"{target_base}/val/real/{file}"
                subset = "val"
            else:
                dst = f"{target_base}/test/real/{file}"
                subset = "test"
            
            shutil.copy2(src, dst)
        
        print(f"✓ Copied {len(real_files)} real audio files")
        print(f"   Train: {train_count}, Val: {val_count}, Test: {n - train_count - val_count}\n")
    else:
        print(f"⚠ No audio files found in {real_source}\n")
    
    # Process FAKE audio
    print(f"Processing FAKE audio from: {fake_source}")
    fake_files = []
    
    if os.path.exists(fake_source):
        for file in os.listdir(fake_source):
            if file.lower().endswith(audio_formats):
                fake_files.append(file)
    
    if fake_files:
        fake_files.sort()
        n = len(fake_files)
        train_count = int(n * 0.8)
        val_count = int(n * 0.1)
        
        for i, file in enumerate(fake_files):
            src = os.path.join(fake_source, file)
            
            if i < train_count:
                dst = f"{target_base}/train/fake/{file}"
                subset = "train"
            elif i < train_count + val_count:
                dst = f"{target_base}/val/fake/{file}"
                subset = "val"
            else:
                dst = f"{target_base}/test/fake/{file}"
                subset = "test"
            
            shutil.copy2(src, dst)
        
        print(f"✓ Copied {len(fake_files)} fake audio files")
        print(f"   Train: {train_count}, Val: {val_count}, Test: {n - train_count - val_count}\n")
    else:
        print(f"⚠ No audio files found in {fake_source}\n")
    
    # Verify organization
    print("="*70)
    print("DATASET ORGANIZATION SUMMARY")
    print("="*70)
    
    total_real = 0
    total_fake = 0
    
    for split in ['train', 'val', 'test']:
        real_count = len([f for f in os.listdir(f"{target_base}/{split}/real") 
                         if f.lower().endswith(audio_formats)])
        fake_count = len([f for f in os.listdir(f"{target_base}/{split}/fake") 
                         if f.lower().endswith(audio_formats)])
        
        total = real_count + fake_count
        total_real += real_count
        total_fake += fake_count
        
        print(f"\n{split.upper()} Set ({target_base}/{split}/)")
        print(f"  Real: {real_count:5d} samples → {target_base}/{split}/real/")
        print(f"  Fake: {fake_count:5d} samples → {target_base}/{split}/fake/")
        print(f"  Total: {total:5d} samples")
        
        if fake_count > 0:
            ratio = real_count / fake_count
            print(f"  Ratio: {ratio:.2f} (real:fake)")
    
    print("\n" + "="*70)
    print(f"TOTAL DATASET: {total_real + total_fake:,} samples")
    print(f"  Real: {total_real:,} samples")
    print(f"  Fake: {total_fake:,} samples")
    print(f"  Imbalance: {total_real/total_fake if total_fake > 0 else 0:.2f}x")
    print("="*70)
    
    if total_real + total_fake > 0:
        print("\n✅ Dataset successfully organized!")
        print("\nNext steps:")
        print("1. Verify GPU: python check_gpu.py")
        print("2. Start training: python train.py")
        return True
    else:
        print("\n❌ No audio files found in source directories!")
        return False


if __name__ == '__main__':
    organize_dataset()
