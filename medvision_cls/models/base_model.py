"""
Base model class for medical image classification
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseClassifier(nn.Module, ABC):
    """Base class for all medical image classifiers"""
    
    def __init__(self):
        super().__init__()
        self.backbone = None
        self.classifier = None
    
    @abstractmethod
    def _create_backbone(self, **kwargs):
        """Create the backbone network"""
        pass
    
    @abstractmethod  
    def _create_classifier(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        """Create the classifier head"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        return self.backbone(x)
    
    def _get_feature_dim(self, in_channels: int = 3):
        """Get the feature dimension of the backbone"""
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            features = self.backbone(dummy_input)
            return features.shape[1]
