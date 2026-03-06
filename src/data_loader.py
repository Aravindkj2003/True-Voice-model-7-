"""
Data loading and preprocessing for Deepfake Audio Detection System
"""

import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path
import random


class MelSpectrogramTransform:
    """Convert audio to Mel-Spectrogram with ImageNet normalization"""
    
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, 
                 hop_length=512, spectrogram_size=(224, 224), 
                 normalize_imagenet=True):
        """
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length for STFT
            spectrogram_size: Target output size (height, width)
            normalize_imagenet: Use ImageNet normalization stats
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram_size = spectrogram_size
        self.normalize_imagenet = normalize_imagenet
        
        # ImageNet stats
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=0,
            f_max=8000,
        )
    
    def __call__(self, waveform):
        """
        Convert waveform to normalized Mel-Spectrogram
        
        Args:
            waveform: Audio tensor of shape (1, num_samples) or (num_samples,)
        
        Returns:
            spectrogram: Tensor of shape (3, 224, 224) - 3 channels for RGB conversion
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Generate mel-spectrogram (shape: (1, n_mels, time_steps))
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Normalize to [0, 1]
        mel_spec_db = mel_spec_db - mel_spec_db.min()
        mel_spec_db = mel_spec_db / (mel_spec_db.max() + 1e-9)
        
        # Resize to target size
        mel_spec_db = torch.nn.functional.interpolate(
            mel_spec_db.unsqueeze(0),
            size=self.spectrogram_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Convert grayscale to 3-channel (grayscale to RGB by repeating channels)
        mel_spec_rgb = mel_spec_db.repeat(3, 1, 1)
        
        # Apply ImageNet normalization
        if self.normalize_imagenet:
            mel_spec_rgb = (mel_spec_rgb - self.imagenet_mean) / self.imagenet_std
        
        return mel_spec_rgb


class AudioDataset(Dataset):
    """PyTorch Dataset for audio files"""
    
    def __init__(self, audio_paths, labels, transform=None, sample_rate=16000):
        """
        Args:
            audio_paths: List of paths to audio files
            labels: List of labels (0=Real, 1=Fake)
            transform: Audio processing transform
            sample_rate: Target sample rate
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio with torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Apply transform (mel-spectrogram conversion)
            if self.transform:
                spectrogram = self.transform(waveform)
            else:
                spectrogram = waveform
            
            return spectrogram, label
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy tensor with the same shape as expected output
            dummy = torch.zeros((3, 224, 224))
            return dummy, label


class DataManager:
    """Manage dataset preparation and loading"""
    
    def __init__(self, data_dir='./data', sample_rate=16000, batch_size=32, 
                 num_workers=4, seed=42):
        """
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create transform
        self.transform = MelSpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=128,
            spectrogram_size=(224, 224),
            normalize_imagenet=True
        )
    
    def load_dataset_from_manifest(self, manifest_path, split='train'):
        """
        Load dataset from a manifest file
        
        Manifest format (JSON):
        {
            'train': [
                {'audio_path': 'path/to/audio.wav', 'label': 0},
                ...
            ],
            'val': [...],
            'test': [...]
        }
        
        Args:
            manifest_path: Path to manifest JSON file
            split: Which split to load ('train', 'val', 'test')
        
        Returns:
            DataLoader with audio samples
        """
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if split not in manifest:
            raise ValueError(f"Split '{split}' not found in manifest")
        
        samples = manifest[split]
        audio_paths = [s['audio_path'] for s in samples]
        labels = [s['label'] for s in samples]
        
        dataset = AudioDataset(
            audio_paths=audio_paths,
            labels=labels,
            transform=self.transform,
            sample_rate=self.sample_rate
        )
        
        return self.create_dataloader(dataset, split)
    
    def load_from_directory_structure(self, split='train'):
        """
        Load dataset from directory structure:
        data/train/real/audio1.wav
        data/train/fake/audio2.wav
        
        Args:
            split: Directory name (train, val, test)
        
        Returns:
            DataLoader with audio samples
        """
        audio_paths = []
        labels = []
        
        split_dir = os.path.join(self.data_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        
        # Load real files (label=0)
        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_paths.append(os.path.join(real_dir, file))
                    labels.append(0)
        
        # Load fake files (label=1)
        fake_dir = os.path.join(split_dir, 'fake')
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_paths.append(os.path.join(fake_dir, file))
                    labels.append(1)
        
        print(f"Loaded {len(audio_paths)} samples from {split} set")
        print(f"  Real: {labels.count(0)}, Fake: {labels.count(1)}")
        
        dataset = AudioDataset(
            audio_paths=audio_paths,
            labels=labels,
            transform=self.transform,
            sample_rate=self.sample_rate
        )
        
        return self.create_dataloader(dataset, split)
    
    def create_dataloader(self, dataset, split='train'):
        """Create DataLoader from dataset"""
        shuffle = (split == 'train')
        drop_last = (split == 'train')
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_weights(self, labels):
        """
        Calculate class weights for imbalanced dataset
        
        Args:
            labels: List of labels
        
        Returns:
            weights: Tensor of shape (2,) with weights for each class
        """
        unique, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(unique) * counts)
        weights = weights / weights.sum()  # Normalize
        return torch.tensor(weights, dtype=torch.float32)
