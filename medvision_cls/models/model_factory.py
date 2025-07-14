"""
Model factory for creating different classification models
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional

from .resnet import ResNetClassifier
from .resnet3d import ResNet3DClassifier
from .densenet import DenseNetClassifier
from .efficientnet import EfficientNetClassifier
from .vit import ViTClassifier
from .convnext import ConvNeXtClassifier
from .medical_net import MedicalNetClassifier


MODEL_REGISTRY = {
    # ResNet models (2D)
    "resnet18": ResNetClassifier,
    "resnet34": ResNetClassifier,
    "resnet50": ResNetClassifier,
    "resnet101": ResNetClassifier,
    "resnet152": ResNetClassifier,
    
    # ResNet models (3D)
    "resnet3d_18": ResNet3DClassifier,
    "resnet3d_34": ResNet3DClassifier,
    "resnet3d_50": ResNet3DClassifier,
    "resnet3d_101": ResNet3DClassifier,
    "resnet3d_152": ResNet3DClassifier,
    
    # DenseNet models
    "densenet121": DenseNetClassifier,
    "densenet161": DenseNetClassifier,
    "densenet169": DenseNetClassifier,
    "densenet201": DenseNetClassifier,
    
    # EfficientNet models
    "efficientnet_b0": EfficientNetClassifier,
    "efficientnet_b1": EfficientNetClassifier,
    "efficientnet_b2": EfficientNetClassifier,
    "efficientnet_b3": EfficientNetClassifier,
    "efficientnet_b4": EfficientNetClassifier,
    "efficientnet_b5": EfficientNetClassifier,
    "efficientnet_b6": EfficientNetClassifier,
    "efficientnet_b7": EfficientNetClassifier,
    
    # Vision Transformer models
    "vit_base_patch16_224": ViTClassifier,
    "vit_base_patch32_224": ViTClassifier,
    "vit_large_patch16_224": ViTClassifier,
    "vit_large_patch32_224": ViTClassifier,
    
    # ConvNeXt models
    "convnext_tiny": ConvNeXtClassifier,
    "convnext_small": ConvNeXtClassifier,
    "convnext_base": ConvNeXtClassifier,
    "convnext_large": ConvNeXtClassifier,
    
    # Medical-specific models
    "medical_resnet50": MedicalNetClassifier,
    "medical_densenet121": MedicalNetClassifier,
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a classification model.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model arguments
        
    Returns:
        PyTorch model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry. "
                        f"Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    return model_class(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing model information
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
    
    # Get model info from timm if available
    try:
        if hasattr(timm, 'get_model_info'):
            return timm.get_model_info(model_name)
        else:
            # Fallback for older versions
            model = timm.create_model(model_name, pretrained=False)
            return {
                "model_name": model_name,
                "num_params": sum(p.numel() for p in model.parameters()),
                "input_size": (3, 224, 224),  # Default
            }
    except Exception:
        return {
            "model_name": model_name,
            "num_params": "Unknown",
            "input_size": (3, 224, 224),
        }


def list_available_models() -> list:
    """
    List all available models.
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())
