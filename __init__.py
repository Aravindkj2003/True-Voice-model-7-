"""
Deepfake Audio Detection System
A production-ready implementation for detecting synthetic audio


Quick Start:
    >>> from deepfake_detector import DeepfakeDetector
    >>> detector = DeepfakeDetector('models/best_model.pt')
    >>> result = detector.predict('audio.wav')
    >>> print(f"{result['prediction']}: {result['confidence']:.2%}")

Installation:
    pip install -r requirements.txt

Training:
    python train.py

Evaluation:
    python evaluate.py --checkpoint models/*/best_model.pt
"""

__version__ = '1.0.0'
__author__ = 'MCA Project'
__description__ = 'Deepfake Audio Detection using ResNet-18 and Mel-Spectrograms'

# Version info
__title__ = 'deepfake-detector'
__url__ = 'https://github.com/your-repo/deepfake-detector'
__license__ = 'MIT'

print(f"Deepfake Audio Detection System v{__version__}")
