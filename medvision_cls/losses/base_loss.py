"""
Base loss class for medical image classification
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLoss(nn.Module, ABC):
    """Base class for all loss functions"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.name = name
        self.kwargs = kwargs
    
    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass for loss computation"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
