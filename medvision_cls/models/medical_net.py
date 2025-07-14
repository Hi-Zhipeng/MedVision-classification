"""
Medical-specific models for classification
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class MedicalNetClassifier(nn.Module):
    """Medical-specific classifier using RadImageNet or other medical pretrained models"""
    
    def __init__(
        self,
        model_name: str = "medical_resnet50",
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
        
        # Extract base model name
        base_model = model_name.replace("medical_", "")
        
        # Create base model
        self.backbone = timm.create_model(
            base_model,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=in_channels,
            **kwargs
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Create classifier with medical-specific modifications
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        return self.backbone(x)
