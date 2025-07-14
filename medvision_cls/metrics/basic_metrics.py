"""
Basic classification metrics (accuracy, precision, recall, F1)
"""

import torch
from typing import Dict, Any, Optional
from torchmetrics import Accuracy, Precision, Recall, F1Score
from .base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """Accuracy metric with auto task detection"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(name="accuracy", num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Accuracy(task=task, num_classes=num_classes, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class PrecisionMetric(BaseMetric):
    """Precision metric with auto task detection"""
    
    def __init__(self, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(name="precision", num_classes=num_classes, average=average, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Precision(task=task, num_classes=num_classes, average=average, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class RecallMetric(BaseMetric):
    """Recall metric with auto task detection"""
    
    def __init__(self, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(name="recall", num_classes=num_classes, average=average, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Recall(task=task, num_classes=num_classes, average=average, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class F1Metric(BaseMetric):
    """F1 score metric with auto task detection"""
    
    def __init__(self, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(name="f1", num_classes=num_classes, average=average, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = F1Score(task=task, num_classes=num_classes, average=average, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()
