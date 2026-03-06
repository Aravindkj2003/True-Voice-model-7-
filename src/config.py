"""
Configuration file for Deepfake Audio Detection System
"""

# Dataset Configuration
DATASET_CONFIG = {
    'sample_rate': 16000,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'mel_min_freq': 0,
    'mel_max_freq': 8000,
    'spectrogram_size': (224, 224),
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'total_epochs': 50,
    'warmup_epochs': 5,
    'warmup_lr': 1e-4,
    'finetune_lr': 2e-5,
    'weight_decay': 1e-4,
    'device': 'cuda',  # GPU enabled - will auto-switch to 'cpu' if CUDA unavailable
}

# Data Split Configuration
DATA_SPLIT = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
}

# Model Configuration
MODEL_CONFIG = {
    'backbone': 'resnet18',
    'pretrained': True,
    'num_classes': 2,
    'input_channels': 3,
    'dropout_rate': 0.5,
}

# ImageNet Normalization Stats
IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'spec_augment': True,
    'freq_mask_param': 30,
    'time_mask_param': 40,
    'num_freq_masks': 2,
    'num_time_masks': 2,
}

# Evaluation Configuration
EVAL_CONFIG = {
    'calculate_eer': True,
    'save_best_model': True,
    'early_stopping_patience': 10,
    'seed': 42,
}

# Paths
PATHS = {
    'data_dir': './data',
    'model_dir': './models',
    'log_dir': './logs',
    'result_dir': './results',
}

# Class Labels
CLASS_LABELS = {
    0: 'Bona-fide (Real)',
    1: 'Spoof (Fake)',
}
