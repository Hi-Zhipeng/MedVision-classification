"""
Cross entropy loss implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Cross entropy loss with optional label smoothing"""
    
    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(name="cross_entropy", **kwargs)
        self.weight = weight
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, labels, 
            weight=self.weight, 
            label_smoothing=self.label_smoothing
        )


class WeightedCrossEntropyLoss(BaseLoss):
    """Weighted cross entropy loss for handling class imbalance"""
    
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(name="weighted_cross_entropy", **kwargs)
        self.class_weights = class_weights
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.class_weights)
