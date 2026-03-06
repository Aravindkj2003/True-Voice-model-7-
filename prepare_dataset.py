"""
Sample dataset preparation script
Shows how to organize audio files for training
"""

import os
import shutil
from pathlib import Path


def create_sample_directory_structure(base_path='./data'):
    """
    Create sample directory structure for deepfake dataset
    
    Example usage:
        python prepare_dataset.py
    """
    
    print("Creating directory structure...")
    
    # Create directories
    dirs = [
        'train/real',
        'train/fake',
        'val/real',
        'val/fake',
        'test/real',
        'test/fake'
    ]
    
    for d in dirs:
        path = os.path.join(base_path, d)
        os.makedirs(path, exist_ok=True)
        print(f"✓ Created {path}")
    
    print("\n" + "="*60)
    print("Directory Structure Ready!")
    print("="*60)
    print(f"\nAdd your audio files to:")
    print(f"  {base_path}/train/real/  - Real training samples")
    print(f"  {base_path}/train/fake/  - Fake training samples")
    print(f"  {base_path}/val/real/    - Real validation samples")
    print(f"  {base_path}/val/fake/    - Fake validation samples")
    print(f"  {base_path}/test/real/   - Real test samples")
    print(f"  {base_path}/test/fake/   - Fake test samples")
    print("\nSupported formats: .wav, .mp3, .flac")


def organize_from_source(source_dir, target_base='./data'):
    """
    Organize audio files from a source directory
    
    Expected source structure:
        source/
        ├── real_audio/
        │   ├── audio1.wav
        │   ├── audio2.wav
        │   └── ...
        └── fake_audio/
            ├── fake1.wav
            ├── fake2.wav
            └── ...
    
    Example usage:
        python prepare_dataset.py --source /path/to/audio_folder
    """
    
    print(f"Organizing audio from {source_dir}...")
    
    # Create target structure
    os.makedirs(f'{target_base}/train/real', exist_ok=True)
    os.makedirs(f'{target_base}/train/fake', exist_ok=True)
    os.makedirs(f'{target_base}/val/real', exist_ok=True)
    os.makedirs(f'{target_base}/val/fake', exist_ok=True)
    os.makedirs(f'{target_base}/test/real', exist_ok=True)
    os.makedirs(f'{target_base}/test/fake', exist_ok=True)
    
    # Process real audio
    real_source = os.path.join(source_dir, 'real_audio')
    if os.path.exists(real_source):
        files = [f for f in os.listdir(real_source) 
                if f.endswith(('.wav', '.mp3', '.flac'))]
        
        # Split: 80% train, 10% val, 10% test
        train_count = int(len(files) * 0.8)
        val_count = int(len(files) * 0.1)
        
        for i, file in enumerate(files):
            src = os.path.join(real_source, file)
            
            if i < train_count:
                dst = f'{target_base}/train/real/{file}'
            elif i < train_count + val_count:
                dst = f'{target_base}/val/real/{file}'
            else:
                dst = f'{target_base}/test/real/{file}'
            
            shutil.copy2(src, dst)
        
        print(f"✓ Organized {len(files)} real audio files")
    
    # Process fake audio
    fake_source = os.path.join(source_dir, 'fake_audio')
    if os.path.exists(fake_source):
        files = [f for f in os.listdir(fake_source) 
                if f.endswith(('.wav', '.mp3', '.flac'))]
        
        # Split: 80% train, 10% val, 10% test
        train_count = int(len(files) * 0.8)
        val_count = int(len(files) * 0.1)
        
        for i, file in enumerate(files):
            src = os.path.join(fake_source, file)
            
            if i < train_count:
                dst = f'{target_base}/train/fake/{file}'
            elif i < train_count + val_count:
                dst = f'{target_base}/val/fake/{file}'
            else:
                dst = f'{target_base}/test/fake/{file}'
            
            shutil.copy2(src, dst)
        
        print(f"✓ Organized {len(files)} fake audio files")


def verify_dataset_structure(base_path='./data'):
    """
    Verify dataset structure and report statistics
    """
    
    print("\nVerifying dataset structure...")
    print("="*60)
    
    total_real = 0
    total_fake = 0
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        
        real_count = len([f for f in os.listdir(f'{split_path}/real') 
                         if f.endswith(('.wav', '.mp3', '.flac'))])
        fake_count = len([f for f in os.listdir(f'{split_path}/fake') 
                         if f.endswith(('.wav', '.mp3', '.flac'))])
        
        total = real_count + fake_count
        
        print(f"\n{split.upper()} Set:")
        print(f"  Real:  {real_count:5d} samples")
        print(f"  Fake:  {fake_count:5d} samples")
        print(f"  Total: {total:5d} samples")
        print(f"  Ratio: {real_count/fake_count if fake_count > 0 else 0:.2f} (real:fake)")
        
        total_real += real_count
        total_fake += fake_count
    
    print("\n" + "="*60)
    print(f"Total Dataset Size: {total_real + total_fake:,} samples")
    print(f"  Real: {total_real:,}")
    print(f"  Fake: {total_fake:,}")
    print(f"  Imbalance Ratio: {total_real/total_fake if total_fake > 0 else 0:.2f}")
    print("="*60)
    
    # Recommendations
    print("\nRecommendations:")
    if total_real + total_fake < 2000:
        print("  ⚠️  Dataset too small (<2000 samples). Target 10k-50k samples.")
    elif total_real + total_fake < 10000:
        print("  ⚠️  Dataset small. Aim for 10k-50k samples for better performance.")
    else:
        print("  ✓ Dataset size acceptable.")
    
    if total_real < 500 or total_fake < 500:
        print("  ⚠️  One class has < 500 samples. Need at least 500 per class.")
    else:
        print("  ✓ Both classes have sufficient samples.")
    
    if abs(total_real - total_fake) / max(total_real, total_fake) > 0.3:
        print("  ⚠️  Significant class imbalance. Try to balance (50:50 is ideal).")
    else:
        print("  ✓ Classes reasonably balanced.")


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare dataset for deepfake detection training'
    )
    parser.add_argument('--source', type=str, default=None,
                       help='Source directory with real_audio/ and fake_audio/')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing dataset')
    parser.add_argument('--output', type=str, default='./data',
                       help='Output base directory')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset_structure(args.output)
    elif args.source:
        organize_from_source(args.source, args.output)
        verify_dataset_structure(args.output)
    else:
        create_sample_directory_structure(args.output)
        print("\nNext steps:")
        print("1. Copy your audio files to the directories above")
        print("2. Run: python prepare_dataset.py --verify-only")
        print("3. Run: python train.py")
