"""
Advanced classification metrics (AUROC, specificity, confusion matrix)
"""

import torch
from typing import Dict, Any, Optional
from torchmetrics import AUROC, Specificity, ConfusionMatrix
from .base_metric import BaseMetric


class AUROCMetric(BaseMetric):
    """AUROC metric with auto task detection"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(name="auroc", num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = AUROC(task=task, num_classes=num_classes, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class SpecificityMetric(BaseMetric):
    """Specificity metric with auto task detection"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(name="specificity", num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Specificity(task=task, num_classes=num_classes, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class ConfusionMatrixMetric(BaseMetric):
    """Confusion matrix metric with auto task detection"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(name="confusion_matrix", num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = ConfusionMatrix(task=task, num_classes=num_classes, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class SensitivityMetric(BaseMetric):
    """Sensitivity metric (same as recall) with auto task detection"""
    
    def __init__(self, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(name="sensitivity", num_classes=num_classes, average=average, **kwargs)
        self.num_classes = num_classes
        from torchmetrics import Recall
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Recall(task=task, num_classes=num_classes, average=average, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()
