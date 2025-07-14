"""
Inference module for MedVision Classification
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from PIL import Image
import json

from .helpers import load_config, load_image


class MedicalImageInference:
    """Class for medical image inference"""
    
    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize inference class
        
        Args:
            model: Trained model
            config: Configuration dictionary
            device: Device to use for inference
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(device)
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
        # Get class names
        self.class_names = self._get_class_names()
    
    def _setup_transforms(self):
        """Setup validation transforms for inference"""
        from ..transforms import get_val_transforms
        
        data_config = self.config.get("data", {})
        transform_config = data_config.get("transforms", {})
        
        # Use validation transforms for inference
        return get_val_transforms(transform_config)
    
    def _get_class_names(self) -> List[str]:
        """Get class names from config or create default names"""
        data_config = self.config.get("data", {})
        class_names = data_config.get("class_names")
        
        if class_names is None:
            num_classes = getattr(self.model, 'num_classes', 2)
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        return class_names
    
    def predict_single_image(
        self,
        image_path: Union[str, Path],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict on a single image
        
        Args:
            image_path: Path to image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = load_image(str(image_path))
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Prepare results
        results = {
            "image_path": str(image_path),
            "predicted_class": predicted_class.item(),
            "predicted_class_name": self.class_names[predicted_class.item()],
        }
        
        if return_probabilities:
            results["probabilities"] = probabilities.cpu().numpy().tolist()[0]
            results["class_probabilities"] = {
                class_name: prob for class_name, prob 
                in zip(self.class_names, results["probabilities"])
            }
        
        return results
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load and preprocess batch
            for image_path in batch_paths:
                image = load_image(str(image_path))
                if self.transforms:
                    image = self.transforms(image)
                batch_images.append(image)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            # Process results
            for j, image_path in enumerate(batch_paths):
                result = {
                    "image_path": str(image_path),
                    "predicted_class": predicted_classes[j].item(),
                    "predicted_class_name": self.class_names[predicted_classes[j].item()],
                }
                
                if return_probabilities:
                    result["probabilities"] = probabilities[j].cpu().numpy().tolist()
                    result["class_probabilities"] = {
                        class_name: prob for class_name, prob 
                        in zip(self.class_names, result["probabilities"])
                    }
                
                results.append(result)
        
        return results
    
    def predict_directory(
        self,
        input_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        return_probabilities: bool = True,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    ) -> List[Dict[str, Any]]:
        """
        Predict on all images in a directory
        
        Args:
            input_dir: Input directory path
            output_file: Optional output file to save results
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            image_extensions: Valid image extensions
            
        Returns:
            List of prediction results
        """
        input_dir = Path(input_dir)
        
        # Find all image files
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(f"**/*{ext}"))
            image_paths.extend(input_dir.glob(f"**/*{ext.upper()}"))
        
        if not image_paths:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(image_paths)} images for inference")
        
        # Run prediction
        results = self.predict_batch(
            image_paths=image_paths,
            batch_size=batch_size,
            return_probabilities=return_probabilities
        )
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def load_model_for_inference(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda"
) -> Any:
    """
    Load a trained model for inference
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Optional configuration (for num_classes if needed)
        device: Device to load model on
        
    Returns:
        Loaded model ready for inference
    """
    from ..models import ClassificationLightningModule
    
    if config and "model" in config:
        model_config = config["model"]
        num_classes = model_config.get("num_classes")
        
        if num_classes:
            model = ClassificationLightningModule.load_from_checkpoint(
                checkpoint_path,
                num_classes=num_classes
            )
        else:
            model = ClassificationLightningModule.load_from_checkpoint(checkpoint_path)
    else:
        model = ClassificationLightningModule.load_from_checkpoint(checkpoint_path)
    
    model.eval()
    model = model.to(device)
    
    return model


def run_inference(
    model: Any,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Dict[str, Any],
    batch_size: int = 32,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Run inference on images
    
    Args:
        model: Trained model
        input_path: Input path (file or directory)
        output_path: Output path for results
        config: Configuration dictionary
        batch_size: Batch size for processing
        device: Device to use
        
    Returns:
        List of prediction results
    """
    # Create inference object
    inference = MedicalImageInference(
        model=model,
        config=config,
        device=device
    )
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single image inference
        result = inference.predict_single_image(input_path)
        results = [result]
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif input_path.is_dir():
        # Directory inference
        results = inference.predict_directory(
            input_dir=input_path,
            output_file=output_path,
            batch_size=batch_size
        )
    
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")
    
    return results


def run_inference_from_config(
    config_file: str,
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    batch_size: int = 32,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Run inference using configuration file
    
    Args:
        config_file: Path to configuration file
        input_path: Input path (file or directory)
        output_path: Output path for results
        checkpoint_path: Path to model checkpoint
        batch_size: Batch size for processing
        device: Device to use
        
    Returns:
        List of prediction results
    """
    # Load configuration
    config = load_config(config_file)
    
    # Load model
    model = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device
    )
    
    # Run inference
    results = run_inference(
        model=model,
        input_path=input_path,
        output_path=output_path,
        config=config,
        batch_size=batch_size,
        device=device
    )
    
    return results
