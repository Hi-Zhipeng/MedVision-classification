"""
Advanced classification metrics (AUROC, specificity, confusion matrix, NPV, PPV)
"""

import torch
from typing import Dict, Any, Optional
from torchmetrics import AUROC, Specificity, ConfusionMatrix, Metric
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
        super().__init__(name="sensitivity", **kwargs)
        from torchmetrics import Recall
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Recall(task=task, num_classes=num_classes, average=average, **kwargs)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
    
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()


class NPVMetric(BaseMetric):
    """Negative Predictive Value metric"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(name="npv", **kwargs)
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"
        self.confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.confusion_matrix.update(preds, targets)
        
    def compute(self) -> torch.Tensor:
        cm = self.confusion_matrix.compute()
        if self.num_classes == 2:
            # For binary: NPV = TN / (TN + FN)
            tn = cm[0, 0]
            fn = cm[1, 0]
            npv = tn / (tn + fn + 1e-7)  # Add small epsilon to avoid division by zero
            return npv
        else:
            # For multiclass: compute NPV for each class and return macro average
            npvs = []
            for i in range(self.num_classes):
                tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                fn = cm[:, i].sum() - cm[i, i]
                npv = tn / (tn + fn + 1e-7)
                npvs.append(npv)
            return torch.stack(npvs).mean()
    
    def reset(self) -> None:
        self.confusion_matrix.reset()
    
    def clone(self):
        return NPVMetric(self.num_classes, **self.kwargs)


class PPVMetric(BaseMetric):
    """Positive Predictive Value metric (same as Precision)"""
    
    def __init__(self, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(name="ppv", **kwargs)
        from torchmetrics import Precision
        task = "binary" if num_classes == 2 else "multiclass"
        self.metric = Precision(task=task, num_classes=num_classes, average=average, **kwargs)
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric.update(preds, targets)
        
    def compute(self) -> torch.Tensor:
        return self.metric.compute()
    
    def reset(self) -> None:
        self.metric.reset()
    
    def clone(self):
        return PPVMetric(self.num_classes, **self.kwargs)
