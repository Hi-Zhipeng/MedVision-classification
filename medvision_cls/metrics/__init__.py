"""
Metrics module for MedVision Classification
"""

# Base classes
from .base_metric import BaseMetric

# Basic metrics
from .basic_metrics import AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric

# Advanced metrics
from .advanced_metrics import (
    AUROCMetric, 
    SpecificityMetric, 
    SensitivityMetric,
    NPVMetric,
    PPVMetric,
    ConfusionMatrixMetric
)

# Metric collections
from .metric_collections import (
    ClassificationMetrics,
    BinaryClassificationMetrics,
    MultiClassificationMetrics,
    CustomMetricCollection,
)

# Factory and utilities
from .metric_factory import (
    MetricFactory,
    create_metric,
    list_available_metrics,
    METRIC_REGISTRY,
    SMART_METRICS,
    smart_accuracy,
    smart_f1,
    smart_precision,
    smart_recall,
    smart_auroc,
    smart_specificity,
    smart_sensitivity,
    smart_npv,
    smart_ppv,
    smart_confusion_matrix,
)

__all__ = [
    # Base classes
    "BaseMetric",
    
    # Basic metrics
    "AccuracyMetric",
    "PrecisionMetric",
    "RecallMetric",
    "F1Metric",
    
    # Advanced metrics
    "AUROCMetric",
    "SpecificityMetric",
    "SensitivityMetric",
    "NPVMetric", 
    "PPVMetric",
    "ConfusionMatrixMetric",
    
    # Metric collections
    "ClassificationMetrics",
    "BinaryClassificationMetrics",
    "MultiClassificationMetrics",
    "CustomMetricCollection",
    
    # Factory and utilities
    "MetricFactory",
    "create_metric",
    "list_available_metrics",
    "METRIC_REGISTRY",
    "SMART_METRICS",
    
    # Smart metric functions (backward compatibility)
    "smart_accuracy",
    "smart_f1",
    "smart_precision",
    "smart_recall",
    "smart_auroc",
    "smart_specificity",
    "smart_sensitivity",
    "smart_npv",
    "smart_ppv",
    "smart_confusion_matrix",
]
