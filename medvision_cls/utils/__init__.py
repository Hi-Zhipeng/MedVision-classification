"""
Utils module for MedVision Classification
"""

from .helpers import (
    load_config,
    save_config,
    setup_logging,
    create_output_dirs,
    count_parameters,
    get_model_size,
    save_predictions,
    save_classification_report,
    save_confusion_matrix,
    load_image,
    preprocess_image,
    postprocess_predictions,
    visualize_predictions,
    calculate_class_weights,
    seed_everything,
)

# Import training functions
from .training import (
    train_model,
    setup_callbacks,
    setup_logger,
)

# Import testing functions
from .testing import (
    test_model,
    save_test_results,
    evaluate_model_on_dataset,
)

# Import inference functions
from .inference import (
    MedicalImageInference,
    load_model_for_inference,
    run_inference,
    run_inference_from_config,
)

# Conditional import for plotting utilities
try:
    from .plotting import (
        ROCPlotter,
        ConfusionMatrixPlotter,
        plot_training_curves
    )
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    # Create dummy classes to avoid AttributeError
    class ROCPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("Plotting dependencies (matplotlib, sklearn) not available. Please install them.")
    
    class ConfusionMatrixPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("Plotting dependencies (matplotlib, sklearn) not available. Please install them.")
    
    def plot_training_curves(*args, **kwargs):
        raise ImportError("Plotting dependencies (matplotlib, sklearn) not available. Please install them.")

__all__ = [
    # Helper functions
    "load_config",
    "save_config",
    "setup_logging",
    "create_output_dirs",
    "count_parameters",
    "get_model_size",
    "save_predictions",
    "save_classification_report",
    "save_confusion_matrix",
    "load_image",
    "preprocess_image",
    "postprocess_predictions",
    "visualize_predictions",
    "calculate_class_weights",
    "seed_everything",
    
    # Training functions
    "train_model",
    "setup_callbacks",
    "setup_logger",
    
    # Testing functions
    "test_model",
    "save_test_results",
    "evaluate_model_on_dataset",
    
    # Inference functions
    "MedicalImageInference",
    "load_model_for_inference",
    "run_inference",
    "run_inference_from_config",
    
    # Plotting functions
    "ROCPlotter",
    "ConfusionMatrixPlotter",
    "plot_training_curves",
]
