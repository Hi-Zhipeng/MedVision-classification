"""
Utility functions for MedVision Classification
"""

import os
import yaml
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('medvision_cls.log')
        ]
    )


def create_output_dirs(paths_config: Dict[str, str]):
    """Create output directories"""
    default_paths = {
        "output_dir": "outputs",
        "checkpoint_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
        "onnx_dir": "outputs/onnx_models"
    }
    
    paths = {**default_paths, **paths_config}

    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_predictions(
    predictions: List[int],
    probabilities: List[List[float]],
    image_paths: List[str],
    class_names: List[str],
    output_path: str,
    format: str = "json"
):
    """Save predictions to file"""
    
    results = []
    for i, (pred, prob, path) in enumerate(zip(predictions, probabilities, image_paths)):
        result = {
            "image_path": path,
            "predicted_class": class_names[pred],
            "predicted_index": pred,
            "probabilities": {
                class_names[j]: float(prob[j]) for j in range(len(class_names))
            },
            "confidence": float(max(prob))
        }
        results.append(result)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str
):
    """Save classification report"""
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str,
    normalize: bool = True
):
    """Save confusion matrix plot"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_image(image_path: str) -> Image.Image:
    """Load image from path"""
    return Image.open(image_path).convert('RGB')


def preprocess_image(image: Image.Image, transform) -> torch.Tensor:
    """Preprocess image for inference"""
    return transform(image).unsqueeze(0)


def postprocess_predictions(
    logits: torch.Tensor,
    return_probabilities: bool = True
) -> Dict[str, torch.Tensor]:
    """Postprocess model predictions"""
    
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    result = {"predictions": predictions}
    
    if return_probabilities:
        result["probabilities"] = probabilities
    
    return result


def run_inference(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run inference on input"""
    
    from ..transforms import get_val_transforms
    
    # Setup
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get transforms
    data_config = config.get("data", {})
    transform = get_val_transforms(
        image_size=data_config.get("dataset_args", {}).get("image_size", [224, 224]),
        normalize=data_config.get("dataset_args", {}).get("normalize", True)
    )
    
    # Inference config - simplified
    import os
    from pathlib import Path
    
    # Get config sections
    output_config = config.get("output", {})
    
    # Determine input type automatically
    if os.path.isfile(input_path):
        input_type = "single"
    elif os.path.isdir(input_path):
        input_type = "folder"
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")
    
    results = {"predictions": [], "probabilities": [], "image_paths": []}
    
    if input_type == "single":
        # Single image inference
        image = load_image(input_path)
        image_tensor = preprocess_image(image, transform).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            output = postprocess_predictions(
                logits, 
                return_probabilities=True
            )
        
        results["predictions"] = output["predictions"].cpu().numpy().tolist()
        if "probabilities" in output:
            results["probabilities"] = output["probabilities"].cpu().numpy().tolist()
        results["image_paths"] = [input_path]
    
    elif input_type == "folder":
        # Folder inference
        input_dir = Path(input_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for img_path in input_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                image = load_image(str(img_path))
                image_tensor = preprocess_image(image, transform).to(device)
                
                with torch.no_grad():
                    logits = model(image_tensor)
                    output = postprocess_predictions(
                        logits,
                        return_probabilities=True
                    )
                
                results["predictions"].extend(output["predictions"].cpu().numpy().tolist())
                if "probabilities" in output:
                    results["probabilities"].extend(output["probabilities"].cpu().numpy().tolist())
                results["image_paths"].append(str(img_path))
    
    # Save results
    if output_config.get("save_predictions", True):
        save_predictions(
            predictions=results["predictions"],
            probabilities=results["probabilities"],
            image_paths=results["image_paths"],
            class_names=getattr(model, 'class_names', [f"class_{i}" for i in range(model.hparams.num_classes)]),
            output_path=output_path,
            format=output_config.get("output_format", "json")
        )
    
    return results


def visualize_predictions(
    image_paths: List[str],
    predictions: List[int],
    probabilities: List[List[float]],
    class_names: List[str],
    output_path: str,
    num_samples: int = 16
):
    """Visualize predictions"""
    
    num_samples = min(num_samples, len(image_paths))
    indices = np.random.choice(len(image_paths), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= 16:
            break
            
        # Load and display image
        image = Image.open(image_paths[idx])
        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Add prediction info
        pred_class = class_names[predictions[idx]]
        confidence = max(probabilities[idx])
        title = f"Pred: {pred_class}\nConf: {confidence:.3f}"
        axes[i].set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_class_weights(labels: List[int], num_classes: int) -> List[float]:
    """Calculate class weights for imbalanced datasets"""
    
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    
    return class_weights.tolist()


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
