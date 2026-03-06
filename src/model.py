"""
Model architecture for Deepfake Audio Detection System
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    """
    ResNet-18 based binary classifier for deepfake audio detection
    Uses mel-spectrogram as input (converted to RGB 3-channel images)
    """
    
    def __init__(self, num_classes=2, backbone='resnet18', pretrained=True, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            backbone: Backbone architecture ('resnet18', 'resnet34', etc.)
            pretrained: Whether to use pretrained weights from ImageNet
            dropout_rate: Dropout rate for regularization
        """
        super(DeepfakeDetector, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Replace the final classification layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze all backbone layers for warmup training
        
        Args:
            freeze: If True, freeze layers. If False, unfreeze.
        """
        for param in self.backbone.parameters():
            if hasattr(param, 'requires_grad'):
                # Get the parent module
                for module in self.backbone.modules():
                    if isinstance(module, nn.Linear) and module is self.backbone.fc:
                        continue
                    for p in module.parameters():
                        p.requires_grad = not freeze
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_frozen_layers(self):
        """Get list of frozen layers"""
        frozen = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                frozen.append(name)
        return frozen
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class DeepfakeDetectorWithAuxiliary(nn.Module):
    """
    ResNet-18 with auxiliary outputs for multi-task learning
    (Optional enhancement for more robust features)
    """
    
    def __init__(self, num_classes=2, backbone='resnet18', pretrained=True):
        super(DeepfakeDetectorWithAuxiliary, self).__init__()
        
        # Main classification branch
        self.main_detector = DeepfakeDetector(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained
        )
        
        # Auxiliary branch for feature extraction confidence
        # This helps the network learn better representations
        self.auxiliary_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_auxiliary=False):
        logits = self.main_detector(x)
        
        if return_auxiliary:
            # Extract features from the backbone
            aux_logits = self.auxiliary_head(logits)
            return logits, aux_logits
        else:
            return logits


def get_model(num_classes=2, backbone='resnet18', pretrained=True, 
              device='cuda', dropout_rate=0.5):
    """
    Utility function to get a model instance
    
    Args:
        num_classes: Number of output classes
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        device: Device to move model to
        dropout_rate: Dropout rate
    
    Returns:
        model: PyTorch model instance
    """
    model = DeepfakeDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    return model
