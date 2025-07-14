"""
Datasets module for MedVision Classification.
"""

import os
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

from .medical_dataset import MedicalImageDataset


def get_datamodule(config: Dict[str, Any]) -> pl.LightningDataModule:
    """
    Factory function to create a datamodule based on configuration.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        A LightningDataModule implementation
    """
    dataset_type = config["type"].lower()
    
    if dataset_type == "medical":
        datamodule_class = MedicalDataModule
    elif dataset_type == "custom":
        # Add your custom datamodule implementation here
        raise NotImplementedError(f"Custom dataset type not implemented")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return datamodule_class(config)


class MedicalDataModule(pl.LightningDataModule):
    """
    Base DataModule for medical image classification datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the medical data module.
        
        Args:
            config: Dataset configuration
        """
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", os.cpu_count() or 4)
        self.pin_memory = config.get("pin_memory", True)
        self.data_dir = config.get("data_dir", "./data")
        self.train_val_split = config.get("train_val_split", [0.8, 0.2])
        self.seed = config.get("seed", 42)
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Class information
        self.num_classes = None
        self.class_names = None
        
    def prepare_data(self):
        """
        Download and prepare data if needed.
        """
        # This method is called once and on only one GPU
        pass
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets based on stage.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        # Setup transforms
        from ..transforms import get_transforms
        
        # Use new config structure if available, fallback to old structure
        if "train_transforms" in self.config:
            # New structure with explicit transform configs
            self.train_transforms = get_transforms(self.config.get("train_transforms", {}))
            self.val_transforms = get_transforms(self.config.get("val_transforms", {}))
            self.test_transforms = get_transforms(self.config.get("test_transforms", {}))
        else:
            # Fallback: generate transforms from general config
            from ..transforms import get_train_transforms, get_val_transforms
            transforms_config = self.config.get("transforms", {})
            
            self.train_transforms = get_train_transforms(
                image_size=tuple(transforms_config.get("image_size", [224, 224])),
                normalize=transforms_config.get("normalize", True),
                augment=transforms_config.get("augment", True)
            )
            self.val_transforms = get_val_transforms(
                image_size=tuple(transforms_config.get("image_size", [224, 224])),
                normalize=transforms_config.get("normalize", True)
            )
            self.test_transforms = self.val_transforms
        
        # Create datasets based on stage
        if stage == "fit":
            # Only create train and validation datasets for training
            self._setup_train_val_datasets()
            
        elif stage == "test":
            # Only create test dataset for testing
            self._setup_test_dataset()
            
        elif stage == "validate":
            # Only create validation dataset for standalone validation
            self._setup_val_dataset()
            
        elif stage == "predict":
            # For prediction, use test dataset
            self._setup_test_dataset()
            
        elif stage is None:
            # If no stage specified, setup all datasets (for compatibility)
            self._setup_train_val_datasets()
            self._setup_test_dataset()
    
    def _setup_train_val_datasets(self):
        """Setup training and validation datasets"""
        # Get image format from config
        image_format = self.config.get("image_format", "*.png")
        
        dataset = MedicalImageDataset(
            data_dir=self.data_dir,
            mode="train",
            transform=self.train_transforms,
            image_format=image_format
        )
        
        # Get class information from first dataset
        self.num_classes = dataset.num_classes
        self.class_names = dataset.class_names
        
        # Split dataset for training and validation
        try:
            # Try standard split
            train_size = int(self.train_val_split[0] * len(dataset))
            val_size = len(dataset) - train_size
            
            # Set random seed for reproducible split
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )
            
            # Create separate validation dataset with validation transforms
            self.val_dataset.dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                mode="val",
                transform=self.val_transforms,
                image_format=image_format
            )
        except Exception as e:
            print(f"Dataset split error: {e}")
            print("Using separate train and validation sets")
            
            # Create separate training and validation datasets
            self.train_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                mode="train",
                transform=self.train_transforms,
                image_format=image_format
            )
            
            self.val_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                mode="val",
                transform=self.val_transforms,
                image_format=image_format
            )
            
            # Get class information
            self.num_classes = self.train_dataset.num_classes
            self.class_names = self.train_dataset.class_names
    
    def _setup_val_dataset(self):
        """Setup validation dataset only"""
        # Get image format from config
        image_format = self.config.get("image_format", "*.png")
        
        self.val_dataset = MedicalImageDataset(
            data_dir=self.data_dir,
            mode="val",
            transform=self.val_transforms,
            image_format=image_format
        )
        
        # Get class information
        self.num_classes = self.val_dataset.num_classes
        self.class_names = self.val_dataset.class_names
    
    def _setup_test_dataset(self):
        """Setup test dataset"""
        # Get image format from config
        image_format = self.config.get("image_format", "*.png")
        
        self.test_dataset = MedicalImageDataset(
            data_dir=self.data_dir,
            mode="test",
            transform=self.test_transforms,
            image_format=image_format
        )
        
        # Get class information if not already set
        if self.num_classes is None:
            self.num_classes = self.test_dataset.num_classes
            self.class_names = self.test_dataset.class_names
            
    def train_dataloader(self):
        """
        Create training dataloader.
        
        Returns:
            Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """
        Create validation dataloader.
        
        Returns:
            Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        """
        Create test dataloader.
        
        Returns:
            Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# Legacy compatibility classes
class MedicalImageDataModule(MedicalDataModule):
    """Legacy alias for MedicalDataModule"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        normalize: bool = True,
        augment: bool = True,
        seed: int = 42,
        **kwargs
    ):
        # Convert legacy parameters to new config format
        config = {
            "type": "medical",
            "data_dir": data_dir,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "train_val_split": train_val_test_split[:2],  # Only use train/val split
            "seed": seed,
            "transforms": {
                "image_size": image_size,
                "normalize": normalize,
                "augment": augment,
            }
        }
        super().__init__(config)
