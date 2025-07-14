"""
Losses module for MedVision Classification
"""

# Base classes
from .base_loss import BaseLoss

# Specific loss implementations
from .cross_entropy_losses import CrossEntropyLoss, WeightedCrossEntropyLoss
from .focal_losses import FocalLoss, AdaptiveFocalLoss
from .smoothing_losses import LabelSmoothingLoss, AdaptiveLabelSmoothingLoss

# Factory and utilities
from .loss_factory import (
    LossFactory,
    create_loss,
    list_available_losses,
    LOSS_REGISTRY,
)

__all__ = [
    # Base classes
    "BaseLoss",
    
    # Cross entropy losses
    "CrossEntropyLoss",
    "WeightedCrossEntropyLoss",
    
    # Focal losses
    "FocalLoss",
    "AdaptiveFocalLoss",
    
    # Smoothing losses
    "LabelSmoothingLoss",
    "AdaptiveLabelSmoothingLoss",
    
    # Factory and utilities
    "LossFactory",
    "create_loss",
    "list_available_losses",
    "LOSS_REGISTRY",
]
