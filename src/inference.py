"""
Inference script for Deepfake Audio Detection
"""

import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path


class AudioAnalyzer:
    """Analyze audio files and predict deepfake probability"""
    
    def __init__(self, model, device='cuda', checkpoint_path=None):
        """
        Args:
            model: PyTorch model
            device: Device to run inference on
            checkpoint_path: Path to pretrained model weights
        """
        self.model = model
        self.device = device
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
        
        self.model.eval()
    
    def load_model(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    
    def predict(self, audio_path, ensemble=False):
        """
        Predict whether audio is deepfake
        
        Args:
            audio_path: Path to audio file
            ensemble: Use ensemble of augmentations for robustness
        
        Returns:
            prediction: 'Real' or 'Fake'
            confidence: Confidence score (0-1)
            scores: Raw model outputs
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Preprocess
        # (Convert to spectrogram using the same transform as training)
        # This is simplified - in practice, use the same transform from data_loader.py
        
        with torch.no_grad():
            # Convert to device and get prediction
            outputs = self.model(waveform)
            softmax = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(softmax, dim=1).item()
            confidence = softmax[0, prediction].item()
            
            labels = {0: 'Real (Bona-fide)', 1: 'Fake (Spoof)'}
            
            return {
                'prediction': labels[prediction],
                'confidence': confidence,
                'real_score': softmax[0, 0].item(),
                'fake_score': softmax[0, 1].item(),
            }
    
    def batch_predict(self, audio_paths):
        """
        Predict on multiple audio files
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            predictions: List of predictions
        """
        predictions = []
        for audio_path in audio_paths:
            pred = self.predict(audio_path)
            predictions.append({
                'audio_path': audio_path,
                **pred
            })
        
        return predictions


def print_prediction_result(result):
    """Pretty print prediction result"""
    print("\n" + "="*60)
    print(f"File: {result['audio_path']}")
    print("-"*60)
    print(f"Prediction:  {result['prediction']}")
    print(f"Confidence:  {result['confidence']:.4f} (100%)")
    print(f"Real Score:  {result['real_score']:.4f}")
    print(f"Fake Score:  {result['fake_score']:.4f}")
    
    # Interpretation
    if result['confidence'] > 0.8:
        certainty = "Very High"
    elif result['confidence'] > 0.6:
        certainty = "High"
    elif result['confidence'] > 0.5:
        certainty = "Moderate"
    else:
        certainty = "Low"
    
    print(f"Certainty:   {certainty}")
    print("="*60)
