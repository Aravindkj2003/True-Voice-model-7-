"""
GPU/CUDA Verification Script
Checks if GPU is available and ready for training
"""

import torch
import sys


def check_gpu():
    """Verify GPU setup for training"""
    
    print("\n" + "="*70)
    print("GPU/CUDA VERIFICATION")
    print("="*70 + "\n")
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available!")
        print("Training will use CPU (VERY SLOW)")
        print("\nTo use GPU:")
        print("1. Update NVIDIA drivers: https://www.nvidia.com/Download/driverDetails")
        print("2. Install CUDA 11.8+: https://developer.nvidia.com/cuda-downloads")
        print("3. Reinstall PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # GPU details
    print(f"\n✅ CUDA IS AVAILABLE!")
    print(f"\nGPU Details:")
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    
    # Test CUDA
    try:
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = torch.matmul(a, b)
        print(f"\n✅ GPU Test: PASSED")
        print(f"  Successfully performed matrix multiplication on GPU")
    except Exception as e:
        print(f"\n❌ GPU Test: FAILED")
        print(f"  Error: {e}")
        return False
    
    # Memory check
    print(f"\n📊 GPU Memory Status:")
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Total: {total:.2f} GB")
    print(f"  Available: {total - allocated:.2f} GB")
    
    # Batch size recommendation
    free_memory = total - allocated
    if free_memory > 8:
        recommended_batch = 32
        print(f"\n✅ Recommended batch_size: {recommended_batch}")
    elif free_memory > 4:
        recommended_batch = 16
        print(f"\n⚠️  Recommended batch_size: {recommended_batch}")
    else:
        recommended_batch = 8
        print(f"\n⚠️  Warning: Low GPU memory!")
        print(f"   Recommended batch_size: {recommended_batch}")
    
    print("\n" + "="*70)
    print("✅ GPU SETUP READY FOR TRAINING!")
    print("="*70)
    print("\nNext step: python organize_data.py\n")
    
    return True


if __name__ == '__main__':
    success = check_gpu()
    sys.exit(0 if success else 1)
