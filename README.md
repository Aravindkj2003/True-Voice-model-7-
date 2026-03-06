# Deepfake Audio Detection System
## Complete MCA Project Implementation

This is a production-ready implementation of a **Binary Audio Classifier** that distinguishes between **Bona-fide (Real)** human speech and **Spoof (Fake)** synthetic audio using Deep Learning.

---

## Project Overview

### Goal
Build a robust classifier that detects deepfake audio by treating spectrograms as images and leveraging computer vision architectures (ResNet-18) to identify "spectral artifacts" left behind by AI vocoders.

### Key Features
- **Heterogeneous Dataset Strategy**: Mix of multiple data sources for robustness
- **Two-Stage Training**: Progressive fine-tuning with frozen/unfrozen backbone
- **SpecAugment**: Data augmentation for overfitting prevention
- **Weighted Loss**: Handles class imbalance automatically
- **EER Metric**: Professional evaluation metric (Equal Error Rate)
- **ImageNet Normalization**: Proper preprocessing for pretrained models

---

## Project Structure

```
deepfake/
├── data/                          # Dataset directory
│   ├── train/
│   │   ├── real/                 # Real audio samples
│   │   └── fake/                 # Synthetic audio samples
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── src/                           # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration file
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── model.py                 # Model architecture (ResNet-18)
│   ├── trainer.py               # Training logic (two-stage)
│   ├── metrics.py               # Evaluation metrics (including EER)
│   ├── augmentation.py          # SpecAugment implementation
│   ├── inference.py             # Inference utilities
│   └── utils.py                 # Helper functions
│
├── models/                        # Saved model checkpoints
├── logs/                          # Training logs
├── results/                       # Evaluation results
│
├── train.py                       # Main training script
├── evaluate.py                    # Evaluation script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── SETUP.md                      # Setup instructions
└── USAGE_GUIDE.md               # Usage examples

```

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: 4GB+ VRAM)
- **CPU**: Can run on CPU but will be significantly slower
- **RAM**: 8GB minimum

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

---

## Installation

### 1. Clone/Setup Project
```bash
cd deepfake
```

### 2. Create Virtual Environment
```bash
# Using conda
conda create -n deepfake python=3.10
conda activate deepfake

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Your Dataset
Place your audio files in the directory structure:
```
data/
├── train/
│   ├── real/  (place .wav files here)
│   └── fake/  (place .wav files here)
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

## Training

### Quick Start
```bash
python train.py
```

### With Options
```bash
python train.py \
    --use-manifest \
    --test-after-training \
    --device cuda
```

### What Happens During Training

**Stage 1: Warmup (Epochs 1-5)**
- ResNet-18 backbone is **FROZEN** (locked)
- Only the classification head (Linear layer 512→2) is trained
- Learning rate: 1×10⁻⁴
- Goal: Allow classifier to learn basic mapping

**Stage 2: Fine-Tuning (Epochs 6-50)**
- Entire network is **UNFROZEN**
- All layers can update weights
- Learning rate: 2×10⁻⁵ (very low to prevent catastrophic forgetting)
- Goal: Shift early layers from "recognizing objects" to "recognizing acoustic textures"

### Key Configuration Options

Edit `src/config.py` to customize:

```python
TRAINING_CONFIG = {
    'batch_size': 32,              # Change based on GPU memory
    'num_workers': 4,              # Data loading threads
    'total_epochs': 50,            # Total training epochs
    'warmup_epochs': 5,            # Frozen backbone epochs
    'warmup_lr': 1e-4,            # Warmup learning rate
    'finetune_lr': 2e-5,          # Fine-tuning learning rate
    'weight_decay': 1e-4,         # L2 regularization
    'device': 'cuda',             # 'cuda' or 'cpu'
}
```

---

## Evaluation

### Run Evaluation
```bash
python evaluate.py --checkpoint models/YYYYMMDD_HHMMSS/best_model.pt --save-results
```

### Metrics Reported

| Metric | Description | Ideal Value |
|--------|-------------|------------|
| **Accuracy** | % correct predictions | High |
| **Precision** | % Fake predictions that are correct | High |
| **Recall** | % of actual Fakes detected | High |
| **F1-Score** | Harmonic mean of Precision & Recall | High |
| **EER** | Equal Error Rate (FAR = FNR) | Low |
| **AUC** | Area Under ROC Curve | Close to 1.0 |
| **FAR** | False Alarm Rate (false positives) | Low |
| **FNR** | Miss Rate (false negatives) | Low |

---

## Dataset Details

### Required Data Format
- **Sample Rate**: 16,000 Hz (automatically resampled)
- **Channels**: Mono (stereo automatically converted)
- **Format**: .wav, .mp3, .flac
- **Duration**: Flexible (any length)

### Preprocessing Pipeline
1. Load audio at 16 kHz
2. Convert to Mel-Spectrogram (128 bands)
3. Apply log scaling (dB)
4. Normalize to [0, 1]
5. Resize to 224×224 pixels
6. Convert to 3-channel (grayscale → RGB)
7. Apply ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Data Augmentation (SpecAugment)
- **Frequency Masking**: Randomly mask 2 frequency bands (max width: 30)
- **Time Masking**: Randomly mask 2 time bands (max width: 40)
- **Purpose**: Prevents overfitting by forcing model to learn global patterns

### Class Imbalance Handling
- **Weighted Cross-Entropy Loss** automatically balances real/fake samples
- Weights are calculated from training data distribution

---

## Model Architecture

### Base Model: ResNet-18 (ImageNet Pretrained)
```
ResNet-18
├── Conv2d(3 → 64)
├── Layer1 (64 channels)
├── Layer2 (128 channels)
├── Layer3 (256 channels)
├── Layer4 (512 channels)
└── Classifier Head
    ├── Dropout(0.5)
    ├── Linear(512 → 512)
    ├── ReLU
    ├── Dropout(0.5)
    └── Linear(512 → 2)  # [Real, Fake]
```

### Why ResNet-18?
- ✅ Relatively small (11.2M parameters) → Fast training
- ✅ Pretrained on ImageNet → Transfer learning advantage
- ✅ Proven good performance on spectrogram classification
- ✅ Good balance of accuracy vs. computational cost

---

## Training Details

### Optimizer: AdamW
- Adaptive learning rate for each parameter
- Weight decay (L2 regularization) = 1×10⁻⁴
- Helps prevent overfitting

### Loss Function: Weighted CrossEntropy
```
Loss = -Σ weight[i] * log(p[i])
```
Automatically weights classes based on frequency in training data

### Data Split
- **Training**: 80% (for learning)
- **Validation**: 10% (for monitoring/early stopping)
- **Testing**: 10% (final evaluation)

### Expected Training Time
- **GPU (NVIDIA 3090)**: ~2-4 hours for 50 epochs
- **GPU (NVIDIA 2080 Ti)**: ~4-8 hours
- **CPU**: ~24-48 hours (not recommended)

---

## Key Technical Conditions

### 1. Overfitting Prevention
| Strategy | Implementation |
|----------|-----------------|
| SpecAugment | Masking time/frequency bands randomly |
| Dropout | 50% dropout in classification head |
| Weight Decay | L2 regularization (1×10⁻⁴) |
| Early Stopping | Monitor validation loss |

### 2. Imbalance Handling
- **Problem**: 30k fake vs 20k real samples create bias
- **Solution**: Weighted loss function (`weight ∝ 1/count`)
- **Result**: Model learns both classes equally

### 3. Normalization
- **Why ImageNet Stats?**: ResNet-18 was trained on ImageNet
- **Formula**: `x_norm = (x - mean) / std`
- **Values**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 4. Two-Stage Fine-Tuning
| Stage | Epochs | Backbone | LR | Purpose |
|-------|--------|----------|----|---------| 
| Warmup | 1-5 | FROZEN ❄️ | 1×10⁻⁴ | Classifier learns mapping |
| Fine-tune | 6-50 | UNFROZEN 🔥 | 2×10⁻⁵ | Early layers adapt to audio |

---

## File Descriptions

### Core Training Files
- **train.py**: Main entry point for training with argument parsing
- **evaluate.py**: Evaluation script for testing trained models
- **src/trainer.py**: Trainer class implementing two-stage training logic

### Model & Data
- **src/model.py**: ResNet-18 architecture with custom classification head
- **src/data_loader.py**: Audio loading, mel-spectrogram conversion, DataLoader management
- **src/config.py**: Centralized configuration for entire project

### Supporting Modules
- **src/metrics.py**: Metrics calculation including EER (professional standard)
- **src/augmentation.py**: SpecAugment implementation
- **src/inference.py**: Prediction on new audio files
- **src/utils.py**: Helper functions (checkpoints, manifests, etc.)

---

## Common Issues & Solutions

### 1. CUDA Out of Memory
```python
# In src/config.py
'batch_size': 16,  # Reduce from 32
'num_workers': 2,  # Reduce from 4
```

### 2. Data Not Found
```bash
# Ensure correct structure:
ls data/train/real/  # Should show .wav files
ls data/train/fake/
```

### 3. Slow Data Loading
```python
# Increase num_workers (for multi-core CPU)
'num_workers': 8,  # Match your CPU core count
```

### 4. Poor Model Performance
- **Check class balance**: `python -c "import json; m=json.load(open('data/manifest.json')); print([(s['label'], len([x for x in m['train'] if x['label']==s['label']])) for s in m['train'][:1]])"`
- **Verify audio quality**: Ensure data is correctly preprocessed
- **Increase epochs**: Try 100-150 epochs
- **Adjust learning rates**: Try 1×10⁻³ for warmup, 1×10⁻⁴ for fine-tuning

---

## Next Steps

1. **Prepare Dataset**: Organize audio files in `data/` directory
2. **Run Training**: `python train.py`
3. **Monitor Training**: Check `logs/` directory for TensorBoard
4. **Evaluate**: `python evaluate.py --checkpoint models/.../best_model.pt`
5. **Deploy**: Use trained model for inference on new audio

---

## References

- **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (2019)
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Transfer Learning**: Yosinski et al., "How Transferable are Features in Deep Neural Networks?" (2014)

---

## Support

For issues or questions:
1. Check SETUP.md for detailed setup instructions
2. Review USAGE_GUIDE.md for examples
3. Check config.py for parameter explanations
4. Examine training logs in `logs/` directory

---

**Created for MCA Project - Deepfake Audio Detection**
