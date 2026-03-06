# QUICK REFERENCE CARD

## 📋 Command Cheat Sheet

### SETUP (First Time)
```bash
# Create environment
conda create -n deepfake python=3.10 -y
conda activate deepfake

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### DATASET (Before Training)
```bash
# Create directory structure
python prepare_dataset.py

# Add your audio files to the directories created

# Verify dataset
python prepare_dataset.py --verify-only
```

### TRAINING (Main Task)
```bash
# Start training (default config)
python train.py

# Train with custom options
python train.py --test-after-training --use-manifest

# Check progress
cat results/training_summary.json
```

### EVALUATION (After Training)
```bash
# Find best model
ls -lh models/*/best_model.pt | head -1

# Evaluate
python evaluate.py --checkpoint models/YYYYMMDD_HHMMSS/best_model.pt
```

### INFERENCE (Predict on New Audio)
```python
from src.model import get_model
from src.inference import AudioAnalyzer

# Load model
device = 'cuda'
model = get_model(device=device)
analyzer = AudioAnalyzer(model, device, 'models/xxx/best_model.pt')

# Predict
result = analyzer.predict('audio.wav')
print(f"{result['prediction']}: {result['confidence']:.2%}")
```

---

## 📊 Configuration Summary

### Hardware-Specific Settings
```python
# Edit src/config.py

# GPU 8GB+
'batch_size': 32
'num_workers': 4

# GPU 4GB
'batch_size': 16
'num_workers': 2

# CPU
'batch_size': 4
'num_workers': 0
'device': 'cpu'
```

### Training Duration
```python
# Fast (6-8 hours on RTX 3090)
'total_epochs': 50
'warmup_epochs': 5

# More thorough (12-16 hours)
'total_epochs': 100
'warmup_epochs': 10

# Very thorough (24+ hours)
'total_epochs': 200
'warmup_epochs': 20
```

---

## 📈 Training Stages

### Stage 1: Warmup (Epochs 1-5)
```
Backbone: ❌ FROZEN
Head:     ✅ TRAINING
LR:       1×10⁻⁴
Params:   262K trainable
```

### Stage 2: Fine-Tuning (Epochs 6-50)
```
Backbone: ✅ UNFROZEN
Head:     ✅ TRAINING
LR:       2×10⁻⁵ (10× lower)
Params:   11.2M trainable
```

---

## ✅ Expected Results

| Metric | Good | Excellent |
|--------|------|-----------|
| Accuracy | 85%+ | 95%+ |
| Precision | 85%+ | 95%+ |
| Recall | 85%+ | 95%+ |
| F1-Score | 85%+ | 95%+ |
| AUC | 0.90+ | 0.98+ |
| EER | <15% | <5% |

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# src/config.py
'batch_size': 8      # Reduce
'num_workers': 2     # Reduce
```

### Model Not Learning
```python
# src/config.py
'warmup_lr': 5e-4    # Try higher
'total_epochs': 100  # Try more epochs
```

### Slow Data Loading
```python
# src/config.py
'num_workers': 8     # Increase (match CPU cores)
```

### Poor Validation Accuracy
```python
# src/config.py
'weight_decay': 2e-4         # Increase from 1e-4
'freq_mask_param': 50        # Increase augmentation
'time_mask_param': 60
```

---

## 📂 Directory Structure

```
deepfake/
├── data/              # Your audio files go here
├── models/            # Saved models
├── results/           # Evaluation results
├── src/               # Source code
├── train.py          # Main script
├── evaluate.py       # Testing script
├── requirements.txt  # Dependencies
└── README.md         # Full documentation
```

---

## 📄 File Purposes

| File | Purpose |
|------|---------|
| `train.py` | Start training |
| `evaluate.py` | Test trained model |
| `prepare_dataset.py` | Organize audio files |
| `src/config.py` | Change settings |
| `src/model.py` | Neural network |
| `src/data_loader.py` | Load & process audio |
| `src/trainer.py` | Training algorithm |

---

## 📚 Key Concepts

### Mel-Spectrogram
```
Audio → 16kHz Mono → 128 mel-bands 
→ Normalize → 224×224 → RGB
```
**Why?** Transfer learning needs 3-channel RGB images

### Two-Stage Training
```
Stage 1: Learn to classify frozen features
Stage 2: Adapt features to audio task
```
**Why?** Prevents "catastrophic forgetting"

### Weighted Loss
```
If 70% fake, 30% real:
→ Penalize fake misses more
→ Balance learning
```
**Why?** Prevents bias toward majority class

### SpecAugment
```
Randomly mask time/frequency → Robustness
```
**Why?** Prevents overfitting to noise

### EER (Equal Error Rate)
```
Find threshold where FAR = Miss Rate
Lower is better
```
**Why?** Professional evaluation standard

---

## 🎯 Typical Workflow

1. **Setup** (30 mins)
   ```bash
   conda activate deepfake
   python prepare_dataset.py
   ```

2. **Verify** (5 mins)
   ```bash
   python prepare_dataset.py --verify-only
   ```

3. **Training** (2-8 hours depending on hardware)
   ```bash
   python train.py
   ```

4. **Evaluate** (5 mins)
   ```bash
   python evaluate.py --checkpoint models/*/best_model.pt
   ```

5. **Deploy** (depends on application)
   ```python
   analyzer = AudioAnalyzer(model, 'cuda', checkpoint_path)
   result = analyzer.predict('audio.wav')
   ```

---

## 💡 Pro Tips

1. **Use manifest for large datasets**
   ```bash
   python train.py --use-manifest
   ```

2. **Monitor training**
   ```bash
   tail -f results/training_summary.json
   ```

3. **Save best models automatically**
   - Happens during training
   - Check `models/YYYYMMDD_HHMMSS/best_model.pt`

4. **Try different augmentations**
   ```python
   # Strong augmentation for small datasets
   'freq_mask_param': 50
   'time_mask_param': 60
   
   # Weak augmentation for large datasets
   'freq_mask_param': 20
   'time_mask_param': 30
   ```

5. **Adjust learning rates based on results**
   ```python
   # If loss jumps: LR too high, reduce 10×
   'warmup_lr': 1e-5
   
   # If loss stagnates: LR too low, increase 2×
   'warmup_lr': 2e-4
   ```

---

## 🚨 Critical Checks

Before pressing run, ensure:

- ✅ Audio files in `data/train/real/`, `data/train/fake/`
- ✅ Have at least 500 files per class
- ✅ Audio format is .wav, .mp3, or .flac
- ✅ GPU has enough memory (test with small batch first)
- ✅ All imports working: `python -c "from src.model import *"`

---

## 🎓 Learning Journey

```
Audio → Spectrogram (2D image)
            ↓
    ResNet-18 Convolution
            ↓
    Extract 512D features
            ↓
    Classifier: 512D → 2D (Real/Fake)
            ↓
    Softmax: Convert to probabilities
            ↓
    Result: Real=0.2, Fake=0.8
```

---

## 📞 Where to Find Help

| Question | File |
|----------|------|
| "How do I start?" | SETUP.md |
| "What do I do next?" | USAGE_GUIDE.md |
| "How does it work?" | TECHNICAL_SPECS.md |
| "I got an error" | Look in README.md "Issues" section |
| "How do I configure?" | src/config.py (all commented) |

---

## 📌 Remember

- **Start small**: Try with small dataset first
- **Monitor loss**: Should decrease over time
- **Check validation**: Watch for overfitting
- **Save checkpoints**: Best models auto-saved
- **Try different configs**: ML requires experimentation
- **Be patient**: Deepfake detection is hard!

---

**🚀 Now go train your model!**

Questions? Read the full documentation:
- README.md (what is this?)
- SETUP.md (how to install?)
- USAGE_GUIDE.md (how to use?)
- TECHNICAL_SPECS.md (how does it work?)
