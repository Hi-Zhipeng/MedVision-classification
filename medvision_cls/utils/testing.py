"""
Testing module for MedVision Classification
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from .helpers import setup_logging, load_config


def test_model(
    config_file: str,
    checkpoint_path: str,
    debug: bool = False,
    save_predictions: bool = True,
    output_dir: Optional[str] = None
):
    """
    Test a trained classification model
    
    Args:
        config_file: Path to configuration file
        checkpoint_path: Path to model checkpoint
        debug: Enable debug mode
        save_predictions: Whether to save predictions
        output_dir: Directory to save outputs
    """
    # Import here to avoid circular imports
    from ..models import ClassificationLightningModule
    from ..datasets import get_datamodule
    
    # Load configuration
    config = load_config(config_file)
    
    # Setup logging
    setup_logging(debug=debug)
    
    # Set seed
    pl.seed_everything(config.get("seed", 42))
    
    # Setup data module
    data_config = config.get("data", {})
    data_module = get_datamodule(data_config)
    
    # Setup data module for testing
    data_module.setup("test")
    
    # Load model from checkpoint
    model_config = config.get("model", {})
    num_classes = model_config.get("num_classes", data_module.num_classes)
    
    model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes
    )
    
    # Setup trainer for testing
    test_config = config.get("test", {})
    trainer = pl.Trainer(
        accelerator=test_config.get("accelerator", "gpu"),
        devices=test_config.get("devices", 1),
        precision=test_config.get("precision", 16),
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    # Run test
    test_results = trainer.test(model, data_module)
    
    # Save detailed results if requested
    if save_predictions and output_dir:
        save_test_results(
            model=model,
            data_module=data_module,
            test_results=test_results,
            output_dir=output_dir,
            config=config
        )
    
    return test_results


def save_test_results(
    model: Any,
    data_module: Any,
    test_results: list,
    output_dir: str,
    config: Dict[str, Any]
):
    """
    Save detailed test results including predictions, metrics, and visualizations
    
    Args:
        model: Trained model
        data_module: Data module
        test_results: Test results from trainer
        output_dir: Output directory
        config: Configuration
    """
    import os
    import json
    import numpy as np
    from pathlib import Path
    from .helpers import save_predictions, save_classification_report, save_confusion_matrix
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Get test dataloader
    test_dataloader = data_module.test_dataloader()
    
    # Collect predictions and labels
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Handle dictionary format batch
            if isinstance(batch, dict):
                images = batch["image"]
                labels = batch.get("label", None)
                image_paths = batch.get("image_path", [])
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                image_paths = []
            else:
                images = batch
                labels = None
                image_paths = []
            
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            
            if labels is not None:
                all_labels.extend(labels.cpu().numpy().tolist())
                
            # Handle image paths
            if isinstance(image_paths, (list, tuple)):
                all_image_paths.extend(image_paths)
            else:
                # If no image paths available, create placeholder
                batch_size = len(predictions)
                all_image_paths.extend([f"image_{i}" for i in range(len(all_image_paths), len(all_image_paths) + batch_size)])
    
    # Get class names
    num_classes = getattr(model, 'hparams', {}).get('num_classes', None)
    if num_classes is None:
        num_classes = getattr(data_module, 'num_classes', 2)
    
    class_names = getattr(data_module, 'class_names', [f"Class_{i}" for i in range(num_classes)])
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "predictions.json")
    save_predictions(
        predictions=all_predictions,
        probabilities=all_probabilities,
        image_paths=all_image_paths,
        class_names=class_names,
        output_path=predictions_path
    )
    
    # Save metrics if we have labels
    if all_labels:
        # Save classification report
        report_path = os.path.join(output_dir, "classification_report.json")
        save_classification_report(
            y_true=all_labels,
            y_pred=all_predictions,
            class_names=class_names,
            output_path=report_path
        )
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        save_confusion_matrix(
            y_true=all_labels,
            y_pred=all_predictions,
            class_names=class_names,
            output_path=cm_path
        )
    
    # Save test results summary
    summary_path = os.path.join(output_dir, "test_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "test_results": test_results,
            "num_samples": len(all_predictions),
            "num_classes": len(class_names),
            "class_names": class_names,
            "config": config
        }, f, indent=2)
    
    print(f"Test results saved to: {output_dir}")


def evaluate_model_on_dataset(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Evaluate model on a given dataset
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                images = batch
                labels = None
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if labels is not None:
                labels = labels.to(device)
                all_labels.extend(labels.cpu().numpy())
                
                # Calculate loss
                loss = model.loss_fn(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
    
    # Calculate metrics
    results = {
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "num_samples": len(all_predictions)
    }
    
    if all_labels:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        results.update({
            "labels": all_labels,
            "accuracy": accuracy_score(all_labels, all_predictions),
            "avg_loss": total_loss / num_batches if num_batches > 0 else 0,
        })
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        results.update({
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
        })
        
        # Calculate macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro"
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )
        
        results.update({
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
        })
    
    return results
