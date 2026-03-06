"""
Package initialization for deepfake detection system
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from src.model import DeepfakeDetector, get_model
from src.data_loader import AudioDataset, DataManager, MelSpectrogramTransform
from src.trainer import Trainer
from src.metrics import MetricsCalculator
from src.augmentation import SpecAugment

__all__ = [
    'DeepfakeDetector',
    'get_model',
    'AudioDataset',
    'DataManager',
    'MelSpectrogramTransform',
    'Trainer',
    'MetricsCalculator',
    'SpecAugment',
]
