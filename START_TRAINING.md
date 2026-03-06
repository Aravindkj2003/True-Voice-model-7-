# 🚀 TRAINING START GUIDE

## ✅ System Status: READY TO TRAIN

### Hardware
- **GPU**: NVIDIA GeForce RTX 3050 6GB ✅
- **CUDA**: 12.1 ✅
- **PyTorch**: 2.5.1 ✅

### Dataset
- **Total**: 56,654 samples (perfectly balanced)
- **Training**: 45,323 samples
- **Validation**: 5,664 samples
- **Test**: 5,667 samples

### Expected Training Time
- **RTX 3050**: 2-4 hours (50 epochs)
- **Expected Accuracy**: 85-95%
- **Expected EER**: <10%

---

## 🎯 QUICK START (2 Commands)

### Option 1: Standard Training
```bash
python train.py
```

### Option 2: Training with Test Evaluation
```bash
python train.py --test-after-training
```

### Option 3: Training with Manifest (Faster)
```bash
python train.py --use-manifest
```

---

## 📊 What to Expect During Training

### Stage 1: Warmup (Epochs 1-5)
```
--- STAGE 1: Warmup (Epochs 1-5) ---
Backbone: FROZEN | Only training classification head
Learning Rate: 1×10⁻⁴

Epoch 1/50
  Train Loss: 0.6231 | Train Acc: 0.6487
  Val Loss:   0.5902 | Val Acc:   0.6723
  [Best model saved: ./models/20240305_143022/best_model.pt]
```

### Stage 2: Fine-Tuning (Epochs 6-50)
```
--- STAGE 2: Fine-tuning (Epochs 6-50) ---
Backbone: UNFROZEN | Training entire network with low learning rate
Learning Rate: 2×10⁻⁵

Epoch 6/50
  Train Loss: 0.4821 | Train Acc: 0.7634
  Val Loss:   0.4102 | Val Acc:   0.8123
```

---

## 📁 Output Locations

After training completes, you'll find:

### Models
```
models/20240305_HHMMSS/
├── best_model.pt          # Best trained model
└── latest_checkpoint.pt   # Latest checkpoint
```

### Results
```
results/
└── training_summary.json  # Training metrics
└── evaluation_results.json # Test results (if --test-after-training)
```

### Logs
```
logs/
└── (Training logs)
```

---

## 🎓 Training Command Reference

### Basic Training
```bash
cd "C:\Users\Aravind KJ\Desktop\deepfake 7"
python train.py
```

### With Advanced Options
```bash
# Train and evaluate on test set
python train.py --test-after-training

# Use manifest for faster loading
python train.py --use-manifest

# Train on CPU (if GPU unavailable)
python train.py --device cpu
```

---

## 📈 Monitoring Training

### Check Current Training
```bash
# View latest training metrics
cat results/training_summary.json

# View model checkpoints
ls -lh models/20240305_HHMMSS/best_model.pt
```

---

## 🔧 Adjusting Training (if needed)

### If GPU Out of Memory
Edit `src/config.py`:
```python
TRAINING_CONFIG = {
    'batch_size': 16,      # Reduce from 32
    'num_workers': 2,      # Reduce from 4
}
```

### If Training is Slow
Edit `src/config.py`:
```python
TRAINING_CONFIG = {
    'batch_size': 64,      # Increase from 32
    'num_workers': 8,      # Increase from 4
}
```

### For Better Results
Edit `src/config.py`:
```python
TRAINING_CONFIG = {
    'total_epochs': 100,   # Increase from 50
    'warmup_epochs': 10,   # Increase from 5
}
```

---

## ✨ Key Features of Your Training

✅ **Automatic Best Model Selection**
- Saves the model with best validation loss automatically

✅ **Two-Stage Training**
- Stage 1: Learn on frozen features (epochs 1-5)
- Stage 2: Adapt all layers (epochs 6-50)

✅ **GPU Acceleration**
- Full CUDA/GPU support
- Expected speedup: 50-100× vs CPU

✅ **Data Augmentation**
- SpecAugment for robustness
- Prevents overfitting

✅ **Professional Metrics**
- EER (Equal Error Rate)
- AUC, Precision, Recall, F1-Score
- Comprehensive confusion matrix

---

## 🎯 After Training Complete

### 1. View Results
```bash
# Check training summary
python -c "import json; print(json.dumps(json.load(open('results/training_summary.json')), indent=2))"
```

### 2. Evaluate on Test Set
```bash
python evaluate.py --checkpoint models/YYYYMMDD_HHMMSS/best_model.pt
```

### 3. Use for Inference
```python
from src.model import get_model
from src.inference import AudioAnalyzer

analyzer = AudioAnalyzer(
    get_model('cuda'),
    'cuda',
    'models/20240305_HHMMSS/best_model.pt'
)
result = analyzer.predict('audio.wav')
print(f"{result['prediction']}: {result['confidence']:.2%}")
```

---

## 📚 Documentation

If you need help, check:
- **Quick Help**: QUICK_REFERENCE.md
- **Detailed Guide**: USAGE_GUIDE.md
- **Technical Details**: TECHNICAL_SPECS.md
- **Full Overview**: README.md

---

## 🚨 Quick Troubleshooting

### Training Stuck?
- Check `status.py` output
- Verify dataset with `python organize_data.py --verify-only`

### GPU Not Available?
- Check with `python quick_check.py`
- Ensure NVIDIA drivers updated

### Out of Memory?
- Reduce batch_size in src/config.py
- Reduce num_workers
- Restart Python

### Training Too Slow?
- Increase batch_size
- Increase num_workers (match your CPU cores)
- Use faster GPU if available

---

## 🎉 YOU'RE READY!

**Status**: ✅ All systems ready
**GPU**: ✅ NVIDIA RTX 3050 with 6GB
**Dataset**: ✅ 56,654 perfectly balanced samples
**Config**: ✅ GPU and batch size optimized

---

## 🚀 FINAL COMMAND

```bash
python train.py
```

**That's it! Monitor output and wait for completion.**

Estimated time: 2-4 hours on your RTX 3050

---

### 📞 Need Help?
1. **Is training slow?** → Check GPU memory (python quick_check.py)
2. **Getting errors?** → Check logs in models/YYYYMMDD_HHMMSS/
3. **Want to adjust settings?** → Edit src/config.py and restart
4. **Training done?** → Run evaluate.py to test on test set

---

**Happy Training! 🚀**  
*Your Deepfake Detection Model Awaits...*
