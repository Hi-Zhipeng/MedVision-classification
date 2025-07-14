"""
Medical image classification dataset
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from ..transforms import get_train_transforms, get_val_transforms
from ..utils.io import load_image


class MedicalImageDataset(Dataset):
    """
    Generic dataset for medical image classification.
    
    Directory structure should be:
    - data_dir/
        - train/
            - class1/
                - img1.nii.gz
                - img2.nii.gz
                - ...
            - class2/
                - img3.nii.gz
                - img4.nii.gz
                - ...
        - val/
            - class1/
            - class2/
        - test/
            - class1/
            - class2/
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 mode: str = "train",
                 transform: Optional[Callable] = None,
                 image_format: str = "*.nii.gz",
                 image_loader: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize the medical image classification dataset.
        
        Args:
            data_dir: Path to data directory
            mode: Dataset mode ('train', 'val', 'test')
            transform: Transform to apply to images
            image_format: File pattern for images (e.g., "*.nii.gz", "*.png")
            image_loader: Custom loader for images
            class_names: Optional list of class names
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.image_format = image_format
        self.image_loader = image_loader or load_image
        
        # Build dataset
        self.samples = self._load_folder_data()
        
        # Get class names if not provided
        if class_names is None:
            self.class_names = sorted(list(set([sample["label"] for sample in self.samples])))
        else:
            self.class_names = class_names
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Set number of classes
        self.num_classes = len(self.class_names)
        
        # Convert string labels to indices
        for sample in self.samples:
            if isinstance(sample["label"], str):
                sample["label"] = self.label_to_idx[sample["label"]]
        
        print(f"Dataset {mode}: {len(self.samples)} samples, {self.num_classes} classes")
    
    def _load_folder_data(self) -> List[Dict[str, Any]]:
        """Load data from folder structure"""
        samples = []
        
        mode_dir = self.data_dir / self.mode
        if not mode_dir.exists():
            raise ValueError(f"Mode directory {mode_dir} does not exist")
        
        for class_dir in mode_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            
            # Get all image files in this class directory
            image_files = glob.glob(str(class_dir / self.image_format))
            
            for img_path in image_files:
                samples.append({
                    "image_path": img_path,
                    "label": class_name
                })
        
        if len(samples) == 0:
            raise ValueError(f"No samples found in {mode_dir} with pattern {self.image_format}")
        
        return samples
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, label, and image_path
        """
        sample = self.samples[idx]
        
        # Load image using unified io.py loader
        image = self.image_loader(sample["image_path"])
        
        # Apply transforms if available
        if self.transform is not None:
            try:
                # For MONAI transforms, input should be dictionary format
                sample_dict = {"image": image}
                transformed = self.transform(sample_dict)
                image = transformed["image"]
            except Exception as e:
                print(f"Transform error: {e}")
                # Fallback: use image as is
                pass
        
        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "image_path": sample["image_path"]
        }



