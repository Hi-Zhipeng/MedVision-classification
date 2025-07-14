"""
ResNet models for classification
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from .base_model import BaseClassifier


class ResNetClassifier(BaseClassifier):
    """ResNet classifier for medical images"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Create backbone and classifier
        self._create_backbone(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            **kwargs
        )
        
        feature_dim = self._get_feature_dim(in_channels)
        self._create_classifier(feature_dim, num_classes, dropout)
    
    def _create_backbone(self, model_name: str, pretrained: bool, in_channels: int, **kwargs):
        """Create the ResNet backbone"""
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=in_channels,
            **kwargs
        )
    
    def _create_classifier(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        """Create the classifier head"""
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
