"""
Loss factory and registry for creating loss functions
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union
from .base_loss import BaseLoss
from .cross_entropy_losses import CrossEntropyLoss, WeightedCrossEntropyLoss
from .focal_losses import FocalLoss, AdaptiveFocalLoss
from .smoothing_losses import LabelSmoothingLoss, AdaptiveLabelSmoothingLoss


# Loss registry mapping
LOSS_REGISTRY = {
    "cross_entropy": CrossEntropyLoss,
    "weighted_cross_entropy": WeightedCrossEntropyLoss,
    "focal": FocalLoss,
    "adaptive_focal": AdaptiveFocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "adaptive_label_smoothing": AdaptiveLabelSmoothingLoss,
}


class LossFactory:
    """Factory class for creating loss functions"""
    
    @staticmethod
    def create_loss(loss_config: Dict[str, Any]) -> Union[BaseLoss, nn.Module]:
        """
        Create a loss function based on configuration.
        
        Args:
            loss_config: Dictionary containing loss configuration
            
        Returns:
            Loss function instance
            
        Raises:
            ValueError: If loss type is not supported
        """
        loss_type = loss_config.get("type", "cross_entropy").lower()
        
        if loss_type not in LOSS_REGISTRY:
            raise ValueError(
                f"Unsupported loss type: {loss_type}. "
                f"Available types: {list(LOSS_REGISTRY.keys())}"
            )
        
        # Get loss class
        loss_class = LOSS_REGISTRY[loss_type]
        
        # Extract parameters, excluding 'type'
        params = {k: v for k, v in loss_config.items() if k != "type"}
        
        # Handle special cases for parameter processing
        if loss_type in ["cross_entropy", "weighted_cross_entropy"]:
            weight = params.get("weight")
            if weight is not None and not isinstance(weight, torch.Tensor):
                params["weight"] = torch.tensor(weight, dtype=torch.float32)
            
            class_weights = params.get("class_weights")
            if class_weights is not None and not isinstance(class_weights, torch.Tensor):
                params["class_weights"] = torch.tensor(class_weights, dtype=torch.float32)
        
        try:
            return loss_class(**params)
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for {loss_type} loss: {e}. "
                f"Provided parameters: {params}"
            )
    
    @staticmethod
    def list_available_losses() -> list:
        """List all available loss functions"""
        return list(LOSS_REGISTRY.keys())
    
    @staticmethod
    def get_loss_info(loss_type: str) -> Dict[str, Any]:
        """Get information about a specific loss type"""
        if loss_type not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        loss_class = LOSS_REGISTRY[loss_type]
        
        return {
            "name": loss_type,
            "class": loss_class.__name__,
            "module": loss_class.__module__,
            "docstring": loss_class.__doc__
        }


# Convenience function for backward compatibility
def create_loss(loss_config: Dict[str, Any]) -> Union[BaseLoss, nn.Module]:
    """Create a loss function based on configuration (backward compatibility)"""
    return LossFactory.create_loss(loss_config)


def list_available_losses() -> list:
    """List all available loss functions (backward compatibility)"""
    return LossFactory.list_available_losses()
