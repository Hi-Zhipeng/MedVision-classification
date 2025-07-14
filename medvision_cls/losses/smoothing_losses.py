"""
Label smoothing loss implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class LabelSmoothingLoss(BaseLoss):
    """Label smoothing loss"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, **kwargs):
        super().__init__(name="label_smoothing", **kwargs)
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / self.num_classes
        loss = (-targets * log_probs).sum(dim=1).mean()
        return loss


class AdaptiveLabelSmoothingLoss(BaseLoss):
    """Adaptive label smoothing with learnable smoothing parameter"""
    
    def __init__(self, num_classes: int, initial_smoothing: float = 0.1, **kwargs):
        super().__init__(name="adaptive_label_smoothing", **kwargs)
        self.num_classes = num_classes
        # Learnable smoothing parameter
        self.smoothing = nn.Parameter(torch.tensor(initial_smoothing))
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Clamp smoothing to valid range [0, 1)
        smoothing = torch.clamp(self.smoothing, 0.0, 0.99)
        
        log_probs = F.log_softmax(logits, dim=1)
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - smoothing) * targets + smoothing / self.num_classes
        loss = (-targets * log_probs).sum(dim=1).mean()
        return loss
