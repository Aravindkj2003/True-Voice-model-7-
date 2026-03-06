# Usage Guide

## Quick Start (5 minutes)

### 1. Setup Your Data

```bash
# Create directory structure
mkdir -p data/train/real data/train/fake
mkdir -p data/val/real data/val/fake
mkdir -p data/test/real data/test/fake

# Add your audio files:
# - Real audio → data/train/real/*.wav
# - Fake audio → data/train/fake/*.wav
# - etc.
```

### 2. Install & Activate Environment

```bash
# For conda
conda activate deepfake

# For venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Train Model

```bash
python train.py
```

That's it! Training will start with:
- **Stage 1 (Epochs 1-5)**: Warmup with frozen backbone
- **Stage 2 (Epochs 6-50)**: Fine-tuning entire network

---

## Detailed Usage Examples

### Example 1: Basic Training

```bash
# Default configuration
python train.py

# Output:
# ================================================================================
# DEEPFAKE AUDIO DETECTION SYSTEM - TRAINING
# ================================================================================
# 
# Using GPU: NVIDIA GeForce RTX 3090
# Loading datasets...
#   train: 8000 samples (Real: 4000, Fake: 4000)
#   val: 1000 samples (Real: 500, Fake: 500)
#   test: 1000 samples (Real: 500, Fake: 500)
# 
# --- STAGE 1: Warmup (Epochs 1-5) ---
# Training... [============================] 100% 
# Epoch 1/50
#   Train Loss: 0.6231 | Train Acc: 0.6487
#   Val Loss:   0.5902 | Val Acc:   0.6723
#   [Best model saved: ./models/20240305_143022/best_model.pt]
# ...
```

### Example 2: Custom Configuration

Edit `src/config.py` before training:

```python
# Set GPU memory usage
TRAINING_CONFIG = {
    'batch_size': 16,        # Reduced for smaller GPU
    'num_workers': 2,        # Fewer data loaders
    'total_epochs': 100,     # Longer training
    'warmup_epochs': 10,     # More warmup
    'warmup_lr': 5e-4,      # Higher warmup rate
    'finetune_lr': 1e-5,    # Lower fine-tune rate
    'weight_decay': 2e-4,   # More regularization
}
```

Then train:
```bash
python train.py --test-after-training
```

### Example 3: Using Manifest File

For faster data loading (especially with large datasets):

```bash
# Generate manifest from directory structure
python -c "
from src.utils import create_data_manifest
create_data_manifest('./data', './data/manifest.json')
"

# Then train using manifest
python train.py --use-manifest
```

### Example 4: CPU Training (Not Recommended)

For systems without GPU:

```python
# Edit src/config.py
TRAINING_CONFIG = {
    'batch_size': 4,
    'num_workers': 0,
    'device': 'cpu',
}
```

```bash
python train.py --device cpu
```

### Example 5: Evaluation Only

Evaluate a trained model:

```bash
python evaluate.py \
    --checkpoint ./models/20240305_143022/best_model.pt \
    --save-results
```

Output:
```
================================================================================
EVALUATION RESULTS
================================================================================
Accuracy:     0.8750
Precision:    0.8643
Recall:       0.8901
F1-Score:     0.8770
AUC:          0.9423
EER:          0.1249

Confusion Matrix:
  True Negatives:  435
  False Positives: 65
  False Negatives: 55
  True Positives:  445

Error Rates:
  False Alarm Rate (FAR): 0.1302
  Miss Rate (FNR):        0.1099
================================================================================
```

---

## Advanced Usage

### Custom Data Augmentation

Edit `src/config.py`:

```python
AUGMENTATION_CONFIG = {
    'spec_augment': True,
    'freq_mask_param': 50,        # Larger masks
    'time_mask_param': 60,        # Larger masks
    'num_freq_masks': 3,          # More masks
    'num_time_masks': 3,          # More masks
}
```

### Different Model Backbones

Edit `src/config.py`:

```python
MODEL_CONFIG = {
    'backbone': 'resnet34',       # Use ResNet-34 instead
    'pretrained': True,
    'num_classes': 2,
    'dropout_rate': 0.3,          # Less dropout
}
```

Available backbones in `src/model.py`:
- `resnet18` (default) - Fast, 11.2M parameters
- `resnet34` - Medium, 21.3M parameters
- `resnet50` - Slower, 25.5M parameters

### Custom Learning Rates

```python
TRAINING_CONFIG = {
    'warmup_lr': 1e-3,    # Much higher warmup
    'finetune_lr': 5e-5,  # Much lower fine-tune
}
```

### Extended Training

```python
TRAINING_CONFIG = {
    'total_epochs': 200,      # Double the epochs
    'warmup_epochs': 20,      # Longer warmup
}
```

---

## Monitoring Training

### 1. Check Training History

```bash
# Training automatically saves logs
ls ./logs/  # Contains training logs
cat results/training_summary.json  # JSON with metrics
```

### 2. View Recent Results

```bash
# Show latest checkpoint
ls -lt models/*/best_model.pt | head -1

# Show training metrics
python -c "
import json
with open('results/training_summary.json') as f:
    data = json.load(f)
    print(f\"Final Val Acc: {data['val_acc'][-1]:.4f}\")
    print(f\"Final Val Loss: {data['val_loss'][-1]:.4f}\")
"
```

### 3. Visualize Results (Optional)

```bash
# Install matplotlib if needed
pip install matplotlib

# Create visualization script:
python -c "
import json
import matplotlib.pyplot as plt

with open('results/training_summary.json') as f:
    data = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(data['epoch'], data['train_loss'], label='Train')
plt.plot(data['epoch'], data['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(data['epoch'], data['train_acc'], label='Train')
plt.plot(data['epoch'], data['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')

plt.tight_layout()
plt.savefig('training_progress.png')
print('✓ Saved to training_progress.png')
"
```

---

## Inference / Testing on New Audio

### Single File

```python
import torch
from src.model import get_model
from src.inference import AudioAnalyzer, print_prediction_result

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(device=device)
analyzer = AudioAnalyzer(
    model,
    device=device,
    checkpoint_path='models/20240305_143022/best_model.pt'
)

# Predict
result = analyzer.predict('path/to/audio.wav')
print_prediction_result(result)

# Output:
# ============================================================
# File: path/to/audio.wav
# ============================================================
# Prediction:  Fake (Spoof)
# Confidence:  0.8734 (100%)
# Real Score:  0.1266
# Fake Score:  0.8734
# Certainty:   Very High
# ============================================================
```

### Batch Processing

```python
import torch
from src.model import get_model
from src.inference import AudioAnalyzer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(device=device)
analyzer = AudioAnalyzer(
    model,
    device=device,
    checkpoint_path='models/20240305_143022/best_model.pt'
)

# Predict on multiple files
audio_files = [
    'audio1.wav',
    'audio2.wav',
    'audio3.wav',
]

predictions = analyzer.batch_predict(audio_files)

# Print results
for pred in predictions:
    print(f\"{pred['audio_path']}: {pred['prediction']} ({pred['confidence']:.2%})\")
```

---

## Troubleshooting

### Problem 1: "CUDA out of memory"

```python
# Reduce batch size in src/config.py
TRAINING_CONFIG = {
    'batch_size': 8,    # From 32 to 8
}

# Then retrain
python train.py
```

### Problem 2: Model not improving

Check class balance:
```bash
python -c "
import os
real_count = len(os.listdir('data/train/real'))
fake_count = len(os.listdir('data/train/fake'))
print(f'Real: {real_count}, Fake: {fake_count}')
print(f'Ratio: {real_count/fake_count:.2f}')
"
```

If imbalanced, the weighted loss should handle it. But ensure:
- Data quality is good
- Both classes have sufficient samples (>1000 each)

### Problem 3: Slow training

```python
# Reduce data loading overhead
TRAINING_CONFIG = {
    'num_workers': 0,  # Disable multiprocessing (try first)
    # or increase
    'num_workers': 8,  # If Python 3.8+
}
```

### Problem 4: Validation loss not decreasing

Possible causes:
1. Learning rate too high → lower warmup_lr to 5e-5
2. Learning rate too low → increase to 5e-4
3. Insufficient data → add more samples
4. Poor data quality → verify audio files are correct

---

## Project Files Summary

| File | Purpose |
|------|---------|
| `train.py` | Main training entry point |
| `evaluate.py` | Model evaluation script |
| `src/config.py` | All configuration parameters |
| `src/model.py` | Neural network architecture |
| `src/data_loader.py` | Data loading & preprocessing |
| `src/trainer.py` | Training loop implementation |
| `src/metrics.py` | EER and other metrics |
| `src/augmentation.py` | SpecAugment implementation |
| `src/inference.py` | Prediction on new audio |
| `src/utils.py` | Helper functions |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview |
| `SETUP.md` | Setup instructions |
| `USAGE_GUIDE.md` | This file |

---

## Expected Results

### Good Model Performance
```
Accuracy: 85%+
Precision: 85%+
Recall: 85%+
F1-Score: 85%+
AUC: 0.90+
EER: <15%
```

### Excellent Model Performance
```
Accuracy: 95%+
Precision: 95%+
Recall: 95%+
F1-Score: 95%+
AUC: 0.98+
EER: <5%
```

Actual results depend on:
- Dataset size and quality
- Data diversity (different vocoders, languages)
- Training duration
- Hardware capabilities

---

## Quick Commands Reference

```bash
# Setup
conda activate deepfake
pip install -r requirements.txt

# Data preparation
python -c "from src.utils import create_data_manifest; create_data_manifest('./data')"

# Training
python train.py                      # Basic
python train.py --test-after-training  # With final test
python train.py --use-manifest       # With manifest

# Evaluation
python evaluate.py --checkpoint ./models/*/best_model.pt

# Check status
ls -lh models/*/best_model.pt       # List saved models
cat results/training_summary.json   # View metrics
```

---

**Ready to detect deepfakes! Start with `python train.py`**
