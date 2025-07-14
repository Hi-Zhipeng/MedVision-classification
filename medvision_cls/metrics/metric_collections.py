"""
Metric collections and compositions for medical image classification
"""

import torch
from typing import Dict, Any, List, Optional
from torchmetrics import MetricCollection
from .base_metric import BaseMetric
from .basic_metrics import AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric
from .advanced_metrics import AUROCMetric, SpecificityMetric, ConfusionMatrixMetric, SensitivityMetric


class ClassificationMetrics(BaseMetric):
    """Collection of classification metrics"""
    
    def __init__(self, num_classes: int = 2, include_confusion_matrix: bool = False, **kwargs):
        super().__init__(name="classification_metrics", **kwargs)
        self.num_classes = num_classes
        
        # Create metrics collection
        metrics = {
            "accuracy": AccuracyMetric(num_classes).metric,
            "f1": F1Metric(num_classes).metric,
            "precision": PrecisionMetric(num_classes).metric,
            "recall": RecallMetric(num_classes).metric,
            "auroc": AUROCMetric(num_classes).metric,
        }
        
        # Add specificity for binary classification
        if num_classes == 2:
            metrics["specificity"] = SpecificityMetric(num_classes).metric
            metrics["sensitivity"] = SensitivityMetric(num_classes).metric
        
        # Optionally add confusion matrix
        if include_confusion_matrix:
            metrics["confusion_matrix"] = ConfusionMatrixMetric(num_classes).metric
        
        self.metrics = MetricCollection(metrics)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics"""
        self.metrics.update(preds, targets)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics"""
        return self.metrics.compute()
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.reset()


class BinaryClassificationMetrics(ClassificationMetrics):
    """Specialized metrics collection for binary classification"""
    
    def __init__(self, include_confusion_matrix: bool = True, **kwargs):
        super().__init__(num_classes=2, include_confusion_matrix=include_confusion_matrix, **kwargs)


class MultiClassificationMetrics(ClassificationMetrics):
    """Specialized metrics collection for multi-class classification"""
    
    def __init__(self, num_classes: int, include_confusion_matrix: bool = False, **kwargs):
        if num_classes < 3:
            raise ValueError("Multi-class classification requires at least 3 classes")
        super().__init__(num_classes=num_classes, include_confusion_matrix=include_confusion_matrix, **kwargs)


class CustomMetricCollection(BaseMetric):
    """Custom collection of user-defined metrics"""
    
    def __init__(self, metrics: Dict[str, BaseMetric], **kwargs):
        super().__init__(name="custom_metrics", **kwargs)
        
        # Convert BaseMetric instances to their underlying torchmetrics
        torch_metrics = {}
        for name, metric in metrics.items():
            if hasattr(metric, 'metric'):
                torch_metrics[name] = metric.metric
            else:
                torch_metrics[name] = metric
        
        self.metrics = MetricCollection(torch_metrics)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics"""
        self.metrics.update(preds, targets)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics"""
        return self.metrics.compute()
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.reset()
