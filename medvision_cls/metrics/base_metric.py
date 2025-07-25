"""
Base metric class for medical image classification
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import copy


class BaseMetric(nn.Module, ABC):
    """Base class for all metrics"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.name = name
        self.kwargs = kwargs
    
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with predictions and targets"""
        pass
    
    @abstractmethod
    def compute(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute final metric value"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state"""
        pass
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass - update and compute metric"""
        self.update(preds, targets)
        return self.compute()
    
    def clone(self):
        """Clone the metric for independent usage"""
        # Create a new instance with the same parameters
        cloned = self.__class__(**self.kwargs)
        return cloned
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
