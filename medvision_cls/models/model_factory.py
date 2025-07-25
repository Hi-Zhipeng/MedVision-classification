"""
Model factory for creating different classification models
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional
import torchvision.models as tv_models

from .resnet import ResNetClassifier
from .resnet3d import ResNet3DClassifier
from .densenet import DenseNetClassifier
from .efficientnet import EfficientNetClassifier
from .vit import ViTClassifier
from .convnext import ConvNeXtClassifier
from .medical_net import MedicalNetClassifier


class TorchVisionModelWrapper(nn.Module):
    """Wrapper for torchvision models to standardize interface"""
    
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Get the model creation function from torchvision
        model_fn = getattr(tv_models, model_name)
        
        # Create model with pretrained weights
        if pretrained:
            try:
                # Try with weights parameter (newer torchvision)
                weights = getattr(tv_models, f"{model_name.upper()}_Weights", None)
                if weights:
                    self.model = model_fn(weights=weights.DEFAULT, **kwargs)
                else:
                    self.model = model_fn(pretrained=True, **kwargs)
            except:
                # Fallback to old pretrained parameter
                self.model = model_fn(pretrained=True, **kwargs)
        else:
            self.model = model_fn(pretrained=False, **kwargs)
        
        # Modify the classifier/fc layer for the desired number of classes
        self._modify_classifier()
    
    def _modify_classifier(self):
        """Modify the final classification layer"""
        if hasattr(self.model, 'classifier'):
            # Models with 'classifier' attribute (AlexNet, VGG, etc.)
            if isinstance(self.model.classifier, nn.Sequential):
                # Find the last Linear layer
                for i, layer in enumerate(reversed(self.model.classifier)):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        self.model.classifier[-i-1] = nn.Linear(in_features, self.num_classes)
                        break
            elif isinstance(self.model.classifier, nn.Linear):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, self.num_classes)
                
        elif hasattr(self.model, 'fc'):
            # Models with 'fc' attribute (ResNet, ResNeXt, etc.)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif hasattr(self.model, 'head'):
            # Models with 'head' attribute (EfficientNet, Vision Transformer, etc.)
            if isinstance(self.model.head, nn.Linear):
                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, self.num_classes)
            elif isinstance(self.model.head, nn.Sequential):
                for i, layer in enumerate(reversed(self.model.head)):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        self.model.head[-i-1] = nn.Linear(in_features, self.num_classes)
                        break
                        
        elif hasattr(self.model, 'heads'):
            # Swin Transformer
            if hasattr(self.model.heads, 'head'):
                in_features = self.model.heads.head.in_features
                self.model.heads.head = nn.Linear(in_features, self.num_classes)
                
        else:
            # Fallback: try to find any Linear layer and modify the last one
            linear_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append((name, module))
            
            if linear_layers:
                last_name, last_layer = linear_layers[-1]
                in_features = last_layer.in_features
                # Replace the last linear layer
                parent_name = '.'.join(last_name.split('.')[:-1])
                child_name = last_name.split('.')[-1]
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(parent, child_name, nn.Linear(in_features, self.num_classes))
                else:
                    setattr(self.model, child_name, nn.Linear(in_features, self.num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


MODEL_REGISTRY = {
    # === Custom Models ===
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
    
    # === TorchVision Models ===
    # AlexNet
    "alexnet": TorchVisionModelWrapper,
    
    # VGG
    "vgg11": TorchVisionModelWrapper,
    "vgg11_bn": TorchVisionModelWrapper,
    "vgg13": TorchVisionModelWrapper,
    "vgg13_bn": TorchVisionModelWrapper,
    "vgg16": TorchVisionModelWrapper,
    "vgg16_bn": TorchVisionModelWrapper,
    "vgg19": TorchVisionModelWrapper,
    "vgg19_bn": TorchVisionModelWrapper,
    
    # ResNet (TorchVision versions)
    "tv_resnet18": TorchVisionModelWrapper,
    "tv_resnet34": TorchVisionModelWrapper,
    "tv_resnet50": TorchVisionModelWrapper,
    "tv_resnet101": TorchVisionModelWrapper,
    "tv_resnet152": TorchVisionModelWrapper,
    
    # ResNeXt
    "resnext50_32x4d": TorchVisionModelWrapper,
    "resnext101_32x8d": TorchVisionModelWrapper,
    "resnext101_64x4d": TorchVisionModelWrapper,
    
    # Wide ResNet
    "wide_resnet50_2": TorchVisionModelWrapper,
    "wide_resnet101_2": TorchVisionModelWrapper,
    
    # DenseNet (TorchVision versions)
    "tv_densenet121": TorchVisionModelWrapper,
    "tv_densenet161": TorchVisionModelWrapper,
    "tv_densenet169": TorchVisionModelWrapper,
    "tv_densenet201": TorchVisionModelWrapper,
    
    # Inception
    "inception_v3": TorchVisionModelWrapper,
    "googlenet": TorchVisionModelWrapper,
    
    # MobileNet
    "mobilenet_v2": TorchVisionModelWrapper,
    "mobilenet_v3_large": TorchVisionModelWrapper,
    "mobilenet_v3_small": TorchVisionModelWrapper,
    
    # MNASNet
    "mnasnet0_5": TorchVisionModelWrapper,
    "mnasnet0_75": TorchVisionModelWrapper,
    "mnasnet1_0": TorchVisionModelWrapper,
    "mnasnet1_3": TorchVisionModelWrapper,
    
    # SqueezeNet
    "squeezenet1_0": TorchVisionModelWrapper,
    "squeezenet1_1": TorchVisionModelWrapper,
    
    # ShuffleNet V2
    "shufflenet_v2_x0_5": TorchVisionModelWrapper,
    "shufflenet_v2_x1_0": TorchVisionModelWrapper,
    "shufflenet_v2_x1_5": TorchVisionModelWrapper,
    "shufflenet_v2_x2_0": TorchVisionModelWrapper,
    
    # EfficientNet (TorchVision versions)
    "tv_efficientnet_b0": TorchVisionModelWrapper,
    "tv_efficientnet_b1": TorchVisionModelWrapper,
    "tv_efficientnet_b2": TorchVisionModelWrapper,
    "tv_efficientnet_b3": TorchVisionModelWrapper,
    "tv_efficientnet_b4": TorchVisionModelWrapper,
    "tv_efficientnet_b5": TorchVisionModelWrapper,
    "tv_efficientnet_b6": TorchVisionModelWrapper,
    "tv_efficientnet_b7": TorchVisionModelWrapper,
    
    # EfficientNetV2
    "efficientnet_v2_s": TorchVisionModelWrapper,
    "efficientnet_v2_m": TorchVisionModelWrapper,
    "efficientnet_v2_l": TorchVisionModelWrapper,
    
    # RegNet
    "regnet_y_400mf": TorchVisionModelWrapper,
    "regnet_y_800mf": TorchVisionModelWrapper,
    "regnet_y_1_6gf": TorchVisionModelWrapper,
    "regnet_y_3_2gf": TorchVisionModelWrapper,
    "regnet_y_8gf": TorchVisionModelWrapper,
    "regnet_y_16gf": TorchVisionModelWrapper,
    "regnet_y_32gf": TorchVisionModelWrapper,
    "regnet_y_128gf": TorchVisionModelWrapper,
    "regnet_x_400mf": TorchVisionModelWrapper,
    "regnet_x_800mf": TorchVisionModelWrapper,
    "regnet_x_1_6gf": TorchVisionModelWrapper,
    "regnet_x_3_2gf": TorchVisionModelWrapper,
    "regnet_x_8gf": TorchVisionModelWrapper,
    "regnet_x_16gf": TorchVisionModelWrapper,
    "regnet_x_32gf": TorchVisionModelWrapper,
    
    # Vision Transformer (TorchVision versions)
    "tv_vit_b_16": TorchVisionModelWrapper,
    "tv_vit_b_32": TorchVisionModelWrapper,
    "tv_vit_l_16": TorchVisionModelWrapper,
    "tv_vit_l_32": TorchVisionModelWrapper,
    "tv_vit_h_14": TorchVisionModelWrapper,
    
    # Swin Transformer
    "swin_t": TorchVisionModelWrapper,
    "swin_s": TorchVisionModelWrapper,
    "swin_b": TorchVisionModelWrapper,
    "swin_v2_t": TorchVisionModelWrapper,
    "swin_v2_s": TorchVisionModelWrapper,
    "swin_v2_b": TorchVisionModelWrapper,
    
    # ConvNeXt (TorchVision versions)  
    "tv_convnext_tiny": TorchVisionModelWrapper,
    "tv_convnext_small": TorchVisionModelWrapper,
    "tv_convnext_base": TorchVisionModelWrapper,
    "tv_convnext_large": TorchVisionModelWrapper,
    
    # MaxVit
    "maxvit_t": TorchVisionModelWrapper,
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
    
    # Handle TorchVision models differently
    if model_class == TorchVisionModelWrapper:
        # Map model names to actual torchvision function names
        tv_model_mapping = {
            # VGG
            "vgg11": "vgg11", "vgg11_bn": "vgg11_bn",
            "vgg13": "vgg13", "vgg13_bn": "vgg13_bn", 
            "vgg16": "vgg16", "vgg16_bn": "vgg16_bn",
            "vgg19": "vgg19", "vgg19_bn": "vgg19_bn",
            
            # ResNet (TorchVision)
            "tv_resnet18": "resnet18", "tv_resnet34": "resnet34",
            "tv_resnet50": "resnet50", "tv_resnet101": "resnet101",
            "tv_resnet152": "resnet152",
            
            # DenseNet (TorchVision)
            "tv_densenet121": "densenet121", "tv_densenet161": "densenet161",
            "tv_densenet169": "densenet169", "tv_densenet201": "densenet201",
            
            # EfficientNet (TorchVision)
            "tv_efficientnet_b0": "efficientnet_b0", "tv_efficientnet_b1": "efficientnet_b1",
            "tv_efficientnet_b2": "efficientnet_b2", "tv_efficientnet_b3": "efficientnet_b3",
            "tv_efficientnet_b4": "efficientnet_b4", "tv_efficientnet_b5": "efficientnet_b5",
            "tv_efficientnet_b6": "efficientnet_b6", "tv_efficientnet_b7": "efficientnet_b7",
            
            # Vision Transformer (TorchVision)
            "tv_vit_b_16": "vit_b_16", "tv_vit_b_32": "vit_b_32",
            "tv_vit_l_16": "vit_l_16", "tv_vit_l_32": "vit_l_32",
            "tv_vit_h_14": "vit_h_14",
            
            # ConvNeXt (TorchVision)
            "tv_convnext_tiny": "convnext_tiny", "tv_convnext_small": "convnext_small",
            "tv_convnext_base": "convnext_base", "tv_convnext_large": "convnext_large",
        }
        
        # Get the actual torchvision model name
        tv_model_name = tv_model_mapping.get(model_name, model_name)
        
        return model_class(
            model_name=tv_model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
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


def list_available_models() -> Dict[str, list]:
    """
    List all available models by category.
    
    Returns:
        Dictionary with model categories and their available models
    """
    models_by_category = {
        "Custom ResNet (2D)": [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ],
        "Custom ResNet (3D)": [
            "resnet3d_18", "resnet3d_34", "resnet3d_50", "resnet3d_101", "resnet3d_152"
        ],
        "Custom DenseNet": [
            "densenet121", "densenet161", "densenet169", "densenet201"
        ],
        "Custom EfficientNet": [
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
            "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
        ],
        "Custom Vision Transformer": [
            "vit_base_patch16_224", "vit_base_patch32_224", 
            "vit_large_patch16_224", "vit_large_patch32_224"
        ],
        "Custom ConvNeXt": [
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        ],
        "Medical-specific": [
            "medical_resnet50", "medical_densenet121"
        ],
        "TorchVision AlexNet": [
            "alexnet"
        ],
        "TorchVision VGG": [
            "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", 
            "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"
        ],
        "TorchVision ResNet": [
            "tv_resnet18", "tv_resnet34", "tv_resnet50", "tv_resnet101", "tv_resnet152"
        ],
        "TorchVision ResNeXt": [
            "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"
        ],
        "TorchVision Wide ResNet": [
            "wide_resnet50_2", "wide_resnet101_2"
        ],
        "TorchVision DenseNet": [
            "tv_densenet121", "tv_densenet161", "tv_densenet169", "tv_densenet201"
        ],
        "TorchVision Inception": [
            "inception_v3", "googlenet"
        ],
        "TorchVision MobileNet": [
            "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"
        ],
        "TorchVision MNASNet": [
            "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"
        ],
        "TorchVision SqueezeNet": [
            "squeezenet1_0", "squeezenet1_1"
        ],
        "TorchVision ShuffleNet V2": [
            "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", 
            "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"
        ],
        "TorchVision EfficientNet": [
            "tv_efficientnet_b0", "tv_efficientnet_b1", "tv_efficientnet_b2", "tv_efficientnet_b3",
            "tv_efficientnet_b4", "tv_efficientnet_b5", "tv_efficientnet_b6", "tv_efficientnet_b7"
        ],
        "TorchVision EfficientNetV2": [
            "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"
        ],
        "TorchVision RegNet": [
            "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
            "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf",
            "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf",
            "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf"
        ],
        "TorchVision Vision Transformer": [
            "tv_vit_b_16", "tv_vit_b_32", "tv_vit_l_16", "tv_vit_l_32", "tv_vit_h_14"
        ],
        "TorchVision Swin Transformer": [
            "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b"
        ],
        "TorchVision ConvNeXt": [
            "tv_convnext_tiny", "tv_convnext_small", "tv_convnext_base", "tv_convnext_large"
        ],
        "TorchVision MaxVit": [
            "maxvit_t"
        ]
    }
    
    return models_by_category


def get_all_model_names() -> list:
    """
    Get a flat list of all available model names.
    
    Returns:
        List of all model names
    """
    return list(MODEL_REGISTRY.keys())


def is_3d_model(model_name: str) -> bool:
    """
    Check if a model is designed for 3D input.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model is for 3D input, False otherwise
    """
    return model_name.startswith("resnet3d_") or "3d" in model_name.lower()


def get_model_input_size(model_name: str) -> tuple:
    """
    Get the recommended input size for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (channels, height, width) or (channels, depth, height, width) for 3D
    """
    if is_3d_model(model_name):
        return (1, 32, 224, 224)  # Default for 3D medical images
    
    # Vision Transformers
    if "vit" in model_name and "patch16" in model_name:
        return (3, 224, 224)
    elif "vit" in model_name and "patch32" in model_name:
        return (3, 224, 224)
    
    # Inception v3 requires larger input
    elif model_name == "inception_v3":
        return (3, 299, 299)
    
    # Most other models
    else:
        return (3, 224, 224)


def print_model_summary():
    """Print a summary of all available models"""
    models_by_category = list_available_models()
    
    print("🏥 MedVision-Classification Available Models")
    print("=" * 50)
    
    total_models = 0
    for category, models in models_by_category.items():
        print(f"\n📂 {category} ({len(models)} models):")
        for model in models:
            input_size = get_model_input_size(model)
            if len(input_size) == 4:  # 3D
                print(f"  • {model:<25} (3D: {input_size[1]}×{input_size[2]}×{input_size[3]})")
            else:  # 2D
                print(f"  • {model:<25} (2D: {input_size[1]}×{input_size[2]})")
        total_models += len(models)
    
    print(f"\n📊 Total: {total_models} models available")
    print("\n💡 Usage: create_model('model_name', num_classes=2)")
    print("💡 3D models: Suitable for medical volumes (CT, MRI)")
    print("💡 2D models: Suitable for medical images (X-ray, ultrasound)")


# Backward compatibility
def list_available_models_flat() -> list:
    """List all available models (flat list for backward compatibility)"""
    return get_all_model_names()