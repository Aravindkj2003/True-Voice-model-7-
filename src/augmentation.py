"""
SpecAugment implementation for audio spectrograms
"""

import torch
import random


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    Randomly masks time and frequency bands in the spectrogram
    """
    
    def __init__(self, freq_mask_param=30, time_mask_param=40, 
                 num_freq_masks=2, num_time_masks=2):
        """
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram):
        """
        Apply SpecAugment to spectrogram
        
        Args:
            spectrogram: Input spectrogram of shape (C, H, W) or (1, H, W)
                        where H is frequency dimension and W is time dimension
        
        Returns:
            augmented_spectrogram: Spectrogram with masks applied
        """
        spec = spectrogram.clone()
        
        # Get dimensions (assuming mel-spectrogram format)
        if spec.dim() == 3:
            _, num_mels, num_steps = spec.shape
        else:
            num_mels, num_steps = spec.shape
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if f > 0:
                f_0 = random.randint(0, num_mels - f)
                spec[..., f_0:f_0 + f, :] = 0
        
        # Apply time masks
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if t > 0:
                t_0 = random.randint(0, num_steps - t)
                spec[..., :, t_0:t_0 + t] = 0
        
        return spec


class RandomTimeShift:
    """Randomly shift spectrogram in time dimension"""
    
    def __init__(self, max_shift=10):
        self.max_shift = max_shift
    
    def __call__(self, spectrogram):
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift > 0:
            return torch.cat([torch.zeros_like(spectrogram[..., :shift]), spectrogram[..., :-shift]], dim=-1)
        elif shift < 0:
            return torch.cat([spectrogram[..., -shift:], torch.zeros_like(spectrogram[..., :shift])], dim=-1)
        else:
            return spectrogram


class Compose:
    """Compose multiple augmentations"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
