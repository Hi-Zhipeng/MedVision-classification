"""Transforms module for MedVision Classification using MONAI."""

try:
    import monai.transforms as M
    from monai.transforms import Compose
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

import numpy as np
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from PIL import Image
import torch
from torchvision import transforms


def get_transforms(config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Optional[Callable]:
    """
    Factory function to create MONAI transforms based on configuration.
    
    Args:
        config: Transform configuration dictionary or list of transform configs
        
    Returns:
        MONAI transform callable
    """
    if not config:
        return None
    
    if not HAS_MONAI:
        raise ImportError(
            "MONAI is required for transforms. "
            "Install it using `pip install monai`."
        )
    
    transforms = []
    
    # Handle list format config (new style)
    if isinstance(config, list):
        for transform_config in config:
            transform_name = transform_config.get("name", "")
            transform_params = {k: v for k, v in transform_config.items() if k != "name"}
            
            # Get the MONAI transform class
            if hasattr(M, transform_name):
                transform_class = getattr(M, transform_name)
                transforms.append(transform_class(**transform_params))
            else:
                print(f"Warning: Transform {transform_name} not found in MONAI")
    
    # Handle dict format config (legacy style)  
    elif isinstance(config, dict):
        for transform_name, transform_params in config.items():
            transform_name_lower = transform_name.lower()
            
            if transform_name_lower == "orientation":
                transforms.append(M.Orientation(
                    keys=transform_params.get("keys", ["image", "label"]),
                    axcodes=transform_params.get("axcodes", "RAS"),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "spacing":
                transforms.append(M.Spacing(
                    keys=transform_params.get("keys", ["image", "label"]),
                    pixdim=transform_params["pixdim"],
                    mode=transform_params.get("mode", ["bilinear", "nearest"]),
                    align_corners=transform_params.get("align_corners", [True, True]),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "cropforeground":
                transforms.append(M.CropForeground(
                    keys=transform_params.get("keys", ["image", "label"]),
                    source_key=transform_params.get("source_key", "image"),
                    k_divisible=transform_params.get("k_divisible", 1),
                    mode=transform_params.get("mode", "constant"),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower in ["resize", "resized"]:
                transforms.append(M.Resized(
                    keys=transform_params.get("keys", ["image", "label"]),
                    spatial_size=transform_params["spatial_size"],
                    mode=transform_params.get("mode", ["area", "nearest"]),
                    align_corners=transform_params.get("align_corners", [None, None]),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "resizewithpadorcrop":
                transforms.append(M.ResizeWithPadOrCrop(
                    keys=transform_params.get("keys", ["image", "label"]),
                    spatial_size=transform_params["spatial_size"],
                    mode=transform_params.get("mode", "constant"),
                    value=transform_params.get("value", 0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "randcropbyposneglabel":
                transforms.append(M.RandCropByPosNegLabel(
                    keys=transform_params.get("keys", ["image", "label"]),
                    label_key=transform_params.get("label_key", "label"),
                    spatial_size=transform_params["spatial_size"],
                    pos=transform_params.get("pos", 1),
                    neg=transform_params.get("neg", 1),
                    num_samples=transform_params.get("num_samples", 4),
                    image_key=transform_params.get("image_key", "image"),
                    image_threshold=transform_params.get("image_threshold", 0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "randrotate90":
                transforms.append(M.RandRotate90(
                    keys=transform_params.get("keys", ["image", "label"]),
                    prob=transform_params.get("prob", 0.1),
                    max_k=transform_params.get("max_k", 3),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "randrotate":
                transforms.append(M.RandRotate(
                    keys=transform_params.get("keys", ["image", "label"]),
                    range_x=transform_params.get("range_x", 0.0),
                    range_y=transform_params.get("range_y", 0.0),
                    range_z=transform_params.get("range_z", 0.0),
                    prob=transform_params.get("prob", 0.1),
                    keep_size=transform_params.get("keep_size", True),
                    mode=transform_params.get("mode", ["bilinear", "nearest"]),
                    padding_mode=transform_params.get("padding_mode", ["zeros", "zeros"]),
                    align_corners=transform_params.get("align_corners", [True, True]),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "randflip":
                transforms.append(M.RandFlip(
                    keys=transform_params.get("keys", ["image", "label"]),
                    spatial_axis=transform_params.get("spatial_axis", None),
                    prob=transform_params.get("prob", 0.1),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "randaffine":
                transforms.append(M.RandAffine(
                    keys=transform_params.get("keys", ["image", "label"]),
                    prob=transform_params.get("prob", 0.1),
                    rotate_range=transform_params.get("rotate_range", None),
                    shear_range=transform_params.get("shear_range", None),
                    translate_range=transform_params.get("translate_range", None),
                    scale_range=transform_params.get("scale_range", None),
                    mode=transform_params.get("mode", ["bilinear", "nearest"]),
                    padding_mode=transform_params.get("padding_mode", ["zeros", "zeros"]),
                    cache_grid=transform_params.get("cache_grid", False),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower in ["randzoom", "randzoomd"]:
                transforms.append(M.RandZoomd(
                    keys=transform_params.get("keys", ["image", "label"]),
                    min_zoom=transform_params.get("min_zoom", 0.9),
                    max_zoom=transform_params.get("max_zoom", 1.1),
                    mode=transform_params.get("mode", ["area", "nearest"]),
                    align_corners=transform_params.get("align_corners", [None, None]),
                    prob=transform_params.get("prob", 1.0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower in ["randscaleintensity", "randscaleintensityd"]:
                transforms.append(M.RandScaleIntensityd(
                    keys=transform_params.get("keys", ["image"]),
                    factors=transform_params.get("factors", 0.1),
                    prob=transform_params.get("prob", 1.0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower in ["randshiftintensity", "randshiftintensityd"]:
                transforms.append(M.RandShiftIntensityd(
                    keys=transform_params.get("keys", ["image"]),
                    offsets=transform_params.get("offsets", 0.1),
                    prob=transform_params.get("prob", 1.0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower in ["randgaussiansmooth", "randgaussiansmoothd"]:
                transforms.append(M.RandGaussianSmoothd(
                    keys=transform_params.get("keys", ["image"]),
                    sigma_x=transform_params.get("sigma_x", [0.25, 1.5]),
                    sigma_y=transform_params.get("sigma_y", [0.25, 1.5]),
                    sigma_z=transform_params.get("sigma_z", [0.25, 1.5]),
                    prob=transform_params.get("prob", 1.0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower in ["randbiasfield", "randbiasfieldd"]:
                transforms.append(M.RandBiasFieldd(
                    keys=transform_params.get("keys", ["image"]),
                    degree=transform_params.get("degree", 3),
                    coeff_range=transform_params.get("coeff_range", [0.0, 0.1]),
                    prob=transform_params.get("prob", 1.0),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
            
            elif transform_name_lower == "ensuretype":
                transforms.append(M.EnsureType(
                    keys=transform_params.get("keys", ["image", "label"]),
                    data_type=transform_params.get("data_type", "tensor"),
                    dtype=transform_params.get("dtype", None),
                    device=transform_params.get("device", None),
                    wrap_sequence=transform_params.get("wrap_sequence", False),
                    allow_missing_keys=transform_params.get("allow_missing_keys", False),
                ))
    
    if not transforms:
        return None
        
    try:
        return M.Compose(transforms)
    except Exception as e:
        print(f"Failed to create MONAI transforms: {e}")
        print("Returning identity transform")
        
        def identity_transform(data):
            return data
        
        return identity_transform


def create_basic_transforms_2d(
    spatial_size: tuple = (224, 224),
    pixdim: tuple = (1.0, 1.0),
    intensity_range: tuple = None,
    augmentation: bool = True
) -> Callable:
    """
    Create basic 2D transforms for medical image classification.
    只对图像进行变换，不涉及标签变换。
    
    Args:
        spatial_size: Target spatial size (H, W)
        pixdim: Target pixel spacing (x, y)
        intensity_range: Intensity range for normalization (min, max)
        augmentation: Whether to include data augmentation
        
    Returns:
        MONAI transform compose
    """
    if not HAS_MONAI:
        raise ImportError("MONAI is required for transforms")
    
    transforms = [
        # 基础预处理：只对图像操作
        M.Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
    ]
    
    # 强度归一化
    if intensity_range:
        transforms.append(
            M.ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        )
    else:
        transforms.append(M.NormalizeIntensityd(keys=["image"], nonzero=True))
    
    # 数据增强：只对图像进行增强
    if augmentation:
        transforms.extend([
            M.RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            M.RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
            M.RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
            M.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            M.RandAdjustContrastd(keys=["image"], prob=0.3),
            M.RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
        ])
    
    # 确保输出为tensor格式
    transforms.append(M.EnsureTyped(keys=["image"], data_type="tensor"))
    
    return M.Compose(transforms)


def create_basic_transforms_3d(
    spatial_size: tuple = (64, 64, 64),
    pixdim: tuple = (1.0, 1.0, 1.0),
    intensity_range: tuple = None,
    augmentation: bool = True
) -> Callable:
    """
    Create basic 3D transforms for medical image classification.
    
    Args:
        spatial_size: Target spatial size (D, H, W)
        pixdim: Target pixel spacing (x, y, z)
        intensity_range: Intensity range for normalization (min, max)
        augmentation: Whether to include data augmentation
        
    Returns:
        MONAI transform compose
    """
    if not HAS_MONAI:
        raise ImportError("MONAI is required for transforms")
    
    transforms = [
        M.Orientationd(keys=["image"], axcodes="RAS"),
        M.Spacingd(keys=["image"], pixdim=pixdim, mode=["bilinear"]),
        M.CropForegroundd(keys=["image"], source_key="image"),
        M.ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
    ]
    
    if intensity_range:
        transforms.append(
            M.ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        )
    else:
        transforms.append(M.NormalizeIntensityd(keys=["image"], nonzero=True))
    
    if augmentation:
        transforms.extend([
            M.RandRotate90d(keys=["image"], prob=0.1, max_k=3),
            M.RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
            M.RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
            M.RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
            M.RandAffined(
                keys=["image"],
                mode=["bilinear"],
                prob=0.2,
                spatial_size=spatial_size,
                rotate_range=[0.0, 0.0, np.pi/15],
                scale_range=[0.1, 0.1, 0.1],
            ),
            M.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            M.RandAdjustContrastd(keys=["image"], prob=0.1),
            M.RandGaussianNoised(keys=["image"], prob=0.1, std=0.1),
        ])
    
    transforms.append(M.EnsureTyped(keys=["image"], data_type="tensor"))
    
    return M.Compose(transforms)


# Wrapper functions for compatibility with existing code
def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    augment: bool = True,
    use_monai: bool = True
) -> Callable:
    """
    Get training transforms for classification.
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize images
        augment: Whether to apply augmentation
        use_monai: Whether to use MONAI transforms
        
    Returns:
        Transform function
    """
    if use_monai and HAS_MONAI:
        return create_basic_transforms_2d(
            spatial_size=image_size,
            augmentation=augment
        )
    else:
        return get_torchvision_train_transforms(image_size, normalize, augment)


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    use_monai: bool = True
) -> Callable:
    """
    Get validation transforms for classification.
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize images
        use_monai: Whether to use MONAI transforms
        
    Returns:
        Transform function
    """
    if use_monai and HAS_MONAI:
        return create_basic_transforms_2d(
            spatial_size=image_size,
            augmentation=False
        )
    else:
        return get_torchvision_val_transforms(image_size, normalize)


def get_torchvision_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    augment: bool = True
) -> transforms.Compose:
    """Get training transforms using torchvision (fallback)"""
    
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize(image_size))
    
    if augment:
        # Augmentation
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    return transforms.Compose(transform_list)


def get_torchvision_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> transforms.Compose:
    """Get validation transforms using torchvision (fallback)"""
    
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    return transforms.Compose(transform_list)


# Legacy compatibility functions
def get_medical_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    augment: bool = True
) -> Callable:
    """Get medical-specific training transforms using MONAI"""
    return get_train_transforms(image_size, normalize, augment, use_monai=True)


class MedicalImageTransform:
    """Custom transform wrapper for medical images"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.transform = get_val_transforms(image_size, normalize, use_monai=True)
    
    def __call__(self, image):
        if isinstance(image, str):
            # Image path - load image first
            image = Image.open(image).convert("RGB")
        
        # For MONAI compatibility, create dict format
        if HAS_MONAI:
            import numpy as np
            image_array = np.array(image)
            data_dict = {"image": image_array}
            result = self.transform(data_dict)
            return result["image"]
        else:
            return self.transform(image)


class MonaiTransformAdapter:
    """Adapter to make MONAI transforms work with PIL images"""
    
    def __init__(self, monai_transform: Callable):
        self.monai_transform = monai_transform
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            # Convert PIL to numpy array
            import numpy as np
            image_array = np.array(image)
            
            # Create a dictionary for MONAI
            data_dict = {"image": image_array}
            
            # Apply MONAI transform
            transformed = self.monai_transform(data_dict)
            
            # Return the transformed image
            return transformed["image"]
        else:
            # For tensor inputs, return as is
            return image


def create_monai_transforms_for_pil(
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    augment: bool = True,
    mode: str = "train"
) -> MonaiTransformAdapter:
    """
    Create MONAI transforms that work with PIL images
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize
        augment: Whether to augment (only for train mode)
        mode: 'train' or 'val'
    
    Returns:
        MonaiTransformAdapter instance
    """
    if mode == "train":
        monai_transform = create_basic_transforms_2d(
            spatial_size=image_size,
            augmentation=augment
        )
    else:
        monai_transform = create_basic_transforms_2d(
            spatial_size=image_size,
            augmentation=False
        )
    
    return MonaiTransformAdapter(monai_transform)
