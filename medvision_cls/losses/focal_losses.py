"""
Focal loss implementation for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class FocalLoss(BaseLoss):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean", **kwargs):
        super().__init__(name="focal", **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(BaseLoss):
    """Adaptive focal loss with learnable alpha parameter"""
    
    def __init__(self, num_classes: int, gamma: float = 2.0, reduction: str = "mean", **kwargs):
        super().__init__(name="adaptive_focal", **kwargs)
        self.gamma = gamma
        self.reduction = reduction
        # Learnable alpha parameter for each class
        self.alpha = nn.Parameter(torch.ones(num_classes))
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Get alpha for each sample based on its label
        alpha_t = self.alpha[labels]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
