"""
I/O utilities for medical image classification.
"""

import os
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

import torch
from PIL import Image

# Try to import common medical image libraries
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def load_image(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load medical image from file for classification.
    
    Supported formats:
    - NIfTI (.nii, .nii.gz) via nibabel or SimpleITK
    - DICOM via SimpleITK
    - PNG, JPG, etc. via PIL
    
    Args:
        path: Path to image file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Handle different file formats
    if suffix in ['.nii', '.gz']:
        if HAS_NIBABEL:
            return _load_nifti_nibabel(path, return_meta)
        elif HAS_SITK:
            return _load_sitk(path, return_meta)
        else:
            raise ImportError("Neither nibabel nor SimpleITK is installed. "
                              "Please install one of them to load NIfTI files.")
    
    elif suffix in ['.dcm', '.dicom'] or (path.is_dir() and any(p.suffix.lower() == '.dcm' for p in path.glob('*'))):
        if HAS_SITK:
            return _load_dicom_sitk(path, return_meta)
        else:
            raise ImportError("SimpleITK is not installed. "
                              "Please install it to load DICOM files.")
    
    # Other formats (PNG, JPG, etc.)
    elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        return _load_pil_image(path, return_meta)
    
    # Try SimpleITK for other formats
    elif HAS_SITK:
        return _load_sitk(path, return_meta)
    
    # Fallback to PIL
    else:
        return _load_pil_image(path, return_meta)


def _load_pil_image(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load image using PIL.
    
    Args:
        path: Path to image file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    img = Image.open(str(path)).convert('RGB')
    
    # Convert to numpy array
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Convert to torch tensor [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    
    if return_meta:
        meta = {
            "spacing": (1.0, 1.0),
            "origin": (0.0, 0.0),
            "direction": (1.0, 0.0, 0.0, 1.0),
            "size": img_tensor.shape,
            "format": img.format,
            "mode": img.mode,
        }
        return img_tensor, meta
    
    return img_tensor


def _load_nifti_nibabel(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load NIfTI file using nibabel.
    
    Args:
        path: Path to NIfTI file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    import nibabel as nib
    
    # Load NIfTI file
    nii_img = nib.load(str(path))
    
    # Get image data as numpy array
    img_np = nii_img.get_fdata().astype(np.float32)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_np)
    
    # For classification, we typically want 2D slices
    # If 3D volume, take middle slice or convert to RGB
    if len(img_tensor.shape) == 3:  # 3D volume
        # Take middle slice
        middle_slice = img_tensor.shape[2] // 2
        img_tensor = img_tensor[:, :, middle_slice]
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 2:  # 2D image
        img_tensor = img_tensor.unsqueeze(0)  # [C, H, W]
    
    # Convert to 3-channel for compatibility with pretrained models
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    
    if return_meta:
        # Get metadata
        meta = {
            "affine": nii_img.affine,
            "header": nii_img.header,
            "spacing": nii_img.header.get_zooms(),
            "shape": img_np.shape,
        }
        return img_tensor, meta
    
    return img_tensor


def _load_sitk(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load image file using SimpleITK.
    
    Args:
        path: Path to image file
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    import SimpleITK as sitk
    
    # Load image
    sitk_img = sitk.ReadImage(str(path))
    
    # Get numpy array
    img_np = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_np)
    
    # Handle different dimensions
    if len(img_tensor.shape) == 3:  # 3D volume or multi-slice
        # For classification, take middle slice
        if img_tensor.shape[0] > 1:  # Multiple slices
            middle_slice = img_tensor.shape[0] // 2
            img_tensor = img_tensor[middle_slice]
        else:
            img_tensor = img_tensor.squeeze(0)
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Convert to 3-channel for compatibility
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    
    if return_meta:
        meta = {
            "spacing": sitk_img.GetSpacing(),
            "origin": sitk_img.GetOrigin(),
            "direction": sitk_img.GetDirection(),
            "size": sitk_img.GetSize(),
        }
        return img_tensor, meta
    
    return img_tensor


def _load_dicom_sitk(
    path: Union[str, Path],
    return_meta: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Load DICOM file/series using SimpleITK.
    
    Args:
        path: Path to DICOM file or directory
        return_meta: Whether to return metadata
        
    Returns:
        Image tensor or tuple of (image tensor, metadata)
    """
    import SimpleITK as sitk
    
    if Path(path).is_dir():
        # Load DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(dicom_names)
        sitk_img = reader.Execute()
    else:
        # Load single DICOM file
        sitk_img = sitk.ReadImage(str(path))
    
    return _process_sitk_image(sitk_img, return_meta)


def _process_sitk_image(sitk_img, return_meta: bool = False):
    """Process SimpleITK image to tensor"""
    # Get numpy array
    img_np = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_np)
    
    # Handle different dimensions
    if len(img_tensor.shape) == 3:  # 3D volume
        # Take middle slice for classification
        middle_slice = img_tensor.shape[0] // 2
        img_tensor = img_tensor[middle_slice]
    
    # Add channel dimension if needed
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Convert to 3-channel
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    
    if return_meta:
        meta = {
            "spacing": sitk_img.GetSpacing(),
            "origin": sitk_img.GetOrigin(),
            "direction": sitk_img.GetDirection(),
            "size": sitk_img.GetSize(),
        }
        return img_tensor, meta
    
    return img_tensor
