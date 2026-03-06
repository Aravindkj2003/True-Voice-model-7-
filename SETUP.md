# Setup Instructions

## Step 1: Environment Setup

### Option A: Using Conda (Recommended)

```bash
# Create new conda environment
conda create -n deepfake python=3.10 -y

# Activate environment
conda activate deepfake

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

### Option B: Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (on Windows)
venv\Scripts\activate

# Activate (on Linux/Mac)
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

---

## Step 2: Verify Installation

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

Expected output:
```
PyTorch version: 2.0.1
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
CUDA version: 11.8
```

---

## Step 3: Dataset Preparation

### Directory Structure

Create the following directory structure in your `data/` folder:

```
data/
├── train/
│   ├── real/        # Real audio files (.wav, .mp3, .flac)
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   └── fake/        # Synthetic/deepfake audio files
│       ├── fake1.wav
│       ├── fake2.wav
│       └── ...
├── val/
│   ├── real/        # Validation real audio
│   └── fake/        # Validation fake audio
└── test/
    ├── real/        # Test real audio
    └── fake/        # Test fake audio
```

### Data Requirements

- **Minimum samples**: ~500 per class per split (2000 total minimum)
- **Recommended**: ~50,000 total samples
- **Sample rate**: Any (will be resampled to 16kHz)
- **Channels**: Mono or stereo (converted to mono)
- **Format**: .wav, .mp3, or .flac

### Good Data Sources

For this MCA project, you can use:

1. **MLAAD-tiny**: "Multilingual Large-scale Audio Authenticity Dataset"
   - Real samples + corresponding spoofed versions
   
2. **HiFi-TTS**: High-quality text-to-speech audio
   - Source: Google HiFi-TTS dataset

3. **RAVDESS**: "Ryerson Audio-Visual Emotion Database"
   - Real emotional speech recordings
   
4. **LibriSeVoc**: Text-to-speech synthesis
   - Synthetic speech data

---

## Step 4: Configuration

### Edit `src/config.py`

```python
# For GPU with 8GB VRAM
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'total_epochs': 50,
    'warmup_epochs': 5,
    'warmup_lr': 1e-4,
    'finetune_lr': 2e-5,
    'weight_decay': 1e-4,
    'device': 'cuda',
}

# For GPU with 4GB VRAM (reduce batch size)
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_workers': 2,
    'total_epochs': 50,
    'warmup_epochs': 5,
    'warmup_lr': 1e-4,
    'finetune_lr': 2e-5,
    'weight_decay': 1e-4,
    'device': 'cuda',
}

# For CPU (not recommended)
TRAINING_CONFIG = {
    'batch_size': 4,
    'num_workers': 0,
    'total_epochs': 50,
    'warmup_epochs': 5,
    'warmup_lr': 1e-4,
    'finetune_lr': 2e-5,
    'weight_decay': 1e-4,
    'device': 'cpu',
}
```

---

## Step 5: Dataset Manifest (Optional but Recommended)

Create a `data/manifest.json` file to avoid rescanning directories:

```json
{
    "train": [
        {"audio_path": "data/train/real/real1.wav", "label": 0},
        {"audio_path": "data/train/real/real2.wav", "label": 0},
        {"audio_path": "data/train/fake/fake1.wav", "label": 1},
        {"audio_path": "data/train/fake/fake2.wav", "label": 1}
    ],
    "val": [
        {"audio_path": "data/val/real/real1.wav", "label": 0},
        {"audio_path": "data/val/fake/fake1.wav", "label": 1}
    ],
    "test": [
        {"audio_path": "data/test/real/real1.wav", "label": 0},
        {"audio_path": "data/test/fake/fake1.wav", "label": 1}
    ]
}
```

Or generate automatically:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from utils import create_data_manifest
create_data_manifest('./data', './data/manifest.json')
"
```

---

## Step 6: Verify Setup

```bash
# Check Python packages
pip list | grep -E "torch|librosa|torchaudio"

# Test imports
python -c "
import torch
import torchaudio
import librosa
import numpy as np
from src.model import get_model
from src.data_loader import DataManager
print('✓ All imports successful!')
"

# Check data directory
ls -R data/train/real/ | head -5
ls -R data/train/fake/ | head -5
```

---

## Step 7: Performance Tuning

### GPU Memory Optimization

If you get CUDA out of memory errors:

```python
# src/config.py
TRAINING_CONFIG = {
    'batch_size': 8,      # Reduce batch size
    'num_workers': 2,     # Reduce data workers
    'gradient_accumulation': 4,  # (if supported)
}

# Or use mixed precision training
torch.cuda.set_per_process_memory_fraction(0.9)
```

### Data Loading Speed

If data loading is slow:

```python
# Increase number of workers (match your CPU cores)
TRAINING_CONFIG = {
    'num_workers': 8,  # For 8-core CPU
}

# Use pin_memory for faster GPU transfer
# Already enabled in data_loader.py
```

---

## Step 8: Project Structure Check

Verify your directory structure:

```
deepfake/
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── manifest.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── data_loader.py
│   ├── trainer.py
│   ├── metrics.py
│   ├── augmentation.py
│   ├── inference.py
│   └── utils.py
├── models/
├── logs/
├── results/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── SETUP.md
```

---

## Troubleshooting

### Issue: "No CUDA device found"

```bash
# Check if CUDA is really unavailable
python -c "import torch; print(torch.cuda.is_available())"

# Solution 1: Install CUDA drivers
# Visit: https://developer.nvidia.com/cuda-downloads

# Solution 2: Use CPU instead (slower)
# Edit src/config.py: 'device': 'cpu'
```

### Issue: "ModuleNotFoundError: No module named 'torch'"

```bash
# Reinstall PyTorch
pip install --upgrade torch torchvision torchaudio
```

### Issue: "data directory not found"

```bash
# Ensure you're in the right directory
pwd  # Should show: .../deepfake

# Create data directories if missing
mkdir -p data/train/real data/train/fake
mkdir -p data/val/real data/val/fake
mkdir -p data/test/real data/test/fake
```

### Issue: Slow data loading

```python
# In src/config.py, increase workers
'num_workers': 8,  # Increase this

# Or on Windows, use:
'num_workers': 0,  # Windows has issues with multiprocessing
```

---

## Next Steps

Once setup is complete:
1. See **USAGE_GUIDE.md** for training examples
2. Run `python train.py` to start training
3. Monitor progress in `logs/` directory
4. Use `python evaluate.py` to test the model

---

## System Recommendations

| Component | Minimum | Recommended |
|-----------|---------|------------|
| RAM | 8GB | 16GB |
| GPU VRAM | 4GB | 8GB+ |
| CPU | 4 cores | 8+ cores |
| Disk Space | 20GB | 100GB+ |
| OS | Windows/Linux/Mac | Linux recommended |

---

**Setup Complete!** Ready to train your deepfake detection model.
