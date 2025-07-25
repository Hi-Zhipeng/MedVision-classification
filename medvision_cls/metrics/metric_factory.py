"""
Metric factory and registry for creating metrics
"""

import torch
from typing import Dict, Any, Union, Type
from .base_metric import BaseMetric
from .basic_metrics import AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric
from .advanced_metrics import AUROCMetric, SpecificityMetric, ConfusionMatrixMetric, SensitivityMetric, NPVMetric, PPVMetric
from .metric_collections import (
    ClassificationMetrics, 
    BinaryClassificationMetrics, 
    MultiClassificationMetrics,
    CustomMetricCollection
)


# Metric registry mapping
METRIC_REGISTRY = {
    "accuracy": AccuracyMetric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "f1": F1Metric,
    "auroc": AUROCMetric,
    "specificity": SpecificityMetric,
    "sensitivity": SensitivityMetric,
    "npv": NPVMetric,
    "ppv": PPVMetric,
    "confusion_matrix": ConfusionMatrixMetric,
    "classification_metrics": ClassificationMetrics,
    "binary_metrics": BinaryClassificationMetrics,
    "multiclass_metrics": MultiClassificationMetrics,
}


class MetricFactory:
    """Factory class for creating metrics"""
    
    @staticmethod
    def create_metric(metric_config: Dict[str, Any], num_classes: int = 2) -> BaseMetric:
        """
        Create a metric based on configuration.
        
        Args:
            metric_config: Dictionary containing metric configuration
            num_classes: Number of classes (automatically determines task type)
            
        Returns:
            Metric instance
            
        Raises:
            ValueError: If metric type is not supported
        """
        metric_type = metric_config.get("type", "accuracy").lower()
        
        if metric_type not in METRIC_REGISTRY:
            raise ValueError(
                f"Unsupported metric type: {metric_type}. "
                f"Available types: {list(METRIC_REGISTRY.keys())}"
            )
        
        # Get metric class
        metric_class = METRIC_REGISTRY[metric_type]
        
        # Extract parameters, excluding 'type'
        params = {k: v for k, v in metric_config.items() if k != "type"}
        
        # Add num_classes if not specified and required
        if "num_classes" not in params:
            params["num_classes"] = num_classes
        
        try:
            return metric_class(**params)
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for {metric_type} metric: {e}. "
                f"Provided parameters: {params}"
            )
    
    @staticmethod
    def create_metric_collection(metrics_config: Dict[str, Dict[str, Any]], 
                               num_classes: int = 2) -> CustomMetricCollection:
        """
        Create a collection of metrics based on configuration.
        
        Args:
            metrics_config: Dictionary mapping metric names to their configs
            num_classes: Number of classes
            
        Returns:
            CustomMetricCollection instance
        """
        metrics = {}
        for name, config in metrics_config.items():
            metrics[name] = MetricFactory.create_metric(config, num_classes)
        
        return CustomMetricCollection(metrics)
    
    @staticmethod
    def list_available_metrics() -> list:
        """List all available metrics"""
        return list(METRIC_REGISTRY.keys())
    
    @staticmethod
    def get_metric_info(metric_type: str) -> Dict[str, Any]:
        """Get information about a specific metric type"""
        if metric_type not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        metric_class = METRIC_REGISTRY[metric_type]
        
        return {
            "name": metric_type,
            "class": metric_class.__name__,
            "module": metric_class.__module__,
            "docstring": metric_class.__doc__
        }
    
    @staticmethod
    def create_standard_metrics(num_classes: int = 2, 
                              include_confusion_matrix: bool = False) -> ClassificationMetrics:
        """
        Create a standard set of classification metrics.
        
        Args:
            num_classes: Number of classes
            include_confusion_matrix: Whether to include confusion matrix
            
        Returns:
            ClassificationMetrics instance
        """
        if num_classes == 2:
            return BinaryClassificationMetrics(include_confusion_matrix=include_confusion_matrix)
        else:
            return MultiClassificationMetrics(
                num_classes=num_classes, 
                include_confusion_matrix=include_confusion_matrix
            )


# Convenience functions for backward compatibility
def create_metric(metric_config: Dict[str, Any], num_classes: int = 2) -> BaseMetric:
    """Create a metric based on configuration (backward compatibility)"""
    return MetricFactory.create_metric(metric_config, num_classes)


def list_available_metrics() -> list:
    """List all available metrics (backward compatibility)"""
    return MetricFactory.list_available_metrics()


# Smart metric factory functions (backward compatibility)
def smart_accuracy(num_classes: int, **kwargs):
    """Create accuracy metric with auto task detection"""
    return AccuracyMetric(num_classes, **kwargs)


def smart_f1(num_classes: int, **kwargs):
    """Create F1 metric with auto task detection"""
    kwargs.setdefault("average", "macro")
    return F1Metric(num_classes, **kwargs)


def smart_precision(num_classes: int, **kwargs):
    """Create precision metric with auto task detection"""
    kwargs.setdefault("average", "macro")
    return PrecisionMetric(num_classes, **kwargs)


def smart_recall(num_classes: int, **kwargs):
    """Create recall metric with auto task detection"""
    kwargs.setdefault("average", "macro")
    return RecallMetric(num_classes, **kwargs)


def smart_auroc(num_classes: int, **kwargs):
    """Create AUROC metric with auto task detection"""
    return AUROCMetric(num_classes, **kwargs)


def smart_specificity(num_classes: int, **kwargs):
    """Create specificity metric with auto task detection"""
    return SpecificityMetric(num_classes, **kwargs)


def smart_npv(num_classes: int, **kwargs):
    """Create NPV metric with auto task detection"""
    return NPVMetric(num_classes, **kwargs)


def smart_ppv(num_classes: int, **kwargs):
    """Create PPV metric with auto task detection"""
    kwargs.setdefault("average", "macro")
    return PPVMetric(num_classes, **kwargs)


def smart_sensitivity(num_classes: int, **kwargs):
    """Create sensitivity metric with auto task detection"""
    kwargs.setdefault("average", "macro")
    return SensitivityMetric(num_classes, **kwargs)


def smart_confusion_matrix(num_classes: int, **kwargs):
    """Create confusion matrix with auto task detection"""
    return ConfusionMatrixMetric(num_classes, **kwargs)


# Smart factory functions registry


# Smart metrics registry (backward compatibility)
SMART_METRICS = {
    "accuracy": smart_accuracy,
    "f1": smart_f1,
    "precision": smart_precision,
    "recall": smart_recall,
    "auroc": smart_auroc,
    "specificity": smart_specificity,
    "sensitivity": smart_sensitivity,
    "npv": smart_npv,
    "ppv": smart_ppv,
    "confusion_matrix": smart_confusion_matrix,
}
