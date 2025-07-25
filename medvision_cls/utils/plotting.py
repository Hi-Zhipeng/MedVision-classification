"""
Plotting utilities for medical image classification
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any, Optional
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


class ROCPlotter:
    """Class for plotting ROC curves"""
    
    def __init__(self, save_dir: str = "outputs"):
        """
        Initialize ROC plotter
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_from_outputs(self, test_outputs: List[Dict[str, Any]], num_classes: int) -> None:
        """
        Plot ROC curves from test step outputs
        
        Args:
            test_outputs: List of test step outputs containing 'preds' and 'labels'
            num_classes: Number of classes
        """
        try:
            # Collect all predictions and labels
            all_probs = []
            all_labels = []
            
            for output in test_outputs:
                probs = output["preds"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(labels)
            
            all_probs = np.vstack(all_probs)
            all_labels = np.array(all_labels)
            
            if num_classes == 2:
                # Binary classification
                self.plot_binary_roc(all_probs[:, 1], all_labels)
            else:
                # Multiclass classification
                self.plot_multiclass_roc(all_probs, all_labels, num_classes)
                
        except Exception as e:
            print(f"❌ Failed to plot ROC curve: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_binary_roc(self, all_probs: np.ndarray, all_labels: np.ndarray) -> Dict[str, float]:
        """
        Plot ROC curve for binary classification
        
        Args:
            all_probs: Predicted probabilities for positive class
            all_labels: True labels
            
        Returns:
            Dictionary containing AUC and optimal threshold info
        """
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Binary Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        save_path = os.path.join(self.save_dir, "roc_curve_binary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        
        results = {
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'sensitivity': optimal_sensitivity,
            'specificity': optimal_specificity,
            'save_path': save_path
        }
        
        print(f"✅ Binary ROC curve saved to: {save_path}")
        print(f"📊 Test AUC: {roc_auc:.4f}")
        print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
        print(f"   Sensitivity: {optimal_sensitivity:.4f}")
        print(f"   Specificity: {optimal_specificity:.4f}")
        
        return results
    
    def plot_multiclass_roc(self, all_probs: np.ndarray, all_labels: np.ndarray, 
                           num_classes: int) -> Dict[str, Any]:
        """
        Plot ROC curves for multiclass classification
        
        Args:
            all_probs: Predicted probabilities for all classes
            all_labels: True labels
            num_classes: Number of classes
            
        Returns:
            Dictionary containing AUC values for each class and averages
        """
        # Binarize the labels for multiclass ROC
        all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
        if num_classes == 2:
            all_labels_bin = np.hstack((1 - all_labels_bin, all_labels_bin))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 
                       'navy', 'red', 'green', 'purple', 'brown', 'pink'])
        
        # Plot each class
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro and macro averages
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='gold', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves - Multiclass Classification')
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        save_path = os.path.join(self.save_dir, "roc_curve_multiclass.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        results = {
            'micro_auc': roc_auc["micro"],
            'macro_auc': roc_auc["macro"],
            'class_aucs': {i: roc_auc[i] for i in range(num_classes)},
            'save_path': save_path
        }
        
        print(f"✅ Multiclass ROC curves saved to: {save_path}")
        print(f"📊 Micro-average AUC: {roc_auc['micro']:.4f}")
        print(f"📊 Macro-average AUC: {roc_auc['macro']:.4f}")
        
        # Print individual class AUCs
        for i in range(num_classes):
            print(f"   Class {i} AUC: {roc_auc[i]:.4f}")
        
        return results


class ConfusionMatrixPlotter:
    """Class for plotting confusion matrices"""
    
    def __init__(self, save_dir: str = "outputs"):
        """
        Initialize confusion matrix plotter
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None,
                            normalize: bool = False, title: str = 'Confusion Matrix') -> str:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save the plot
        filename = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {save_path}")
        return save_path


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]],
                        save_dir: str = "outputs") -> str:
    """
    Plot training curves
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Training metrics per epoch
        val_metrics: Validation metrics per epoch
        save_dir: Directory to save plots
        
    Returns:
        Path to saved plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(2, (num_metrics + 1) // 2 + 1, figsize=(15, 8))
    axes = axes.flatten()
    
    # Plot loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics
    for idx, (metric_name, train_values) in enumerate(train_metrics.items(), 1):
        if idx < len(axes):
            val_values = val_metrics.get(metric_name, [])
            axes[idx].plot(epochs, train_values, 'b-', label=f'Training {metric_name}')
            if val_values:
                axes[idx].plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
            axes[idx].set_title(f'Training and Validation {metric_name.title()}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name.title())
            axes[idx].legend()
            axes[idx].grid(True)
    
    # Hide unused subplots
    for idx in range(len(train_metrics) + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to: {save_path}")
    return save_path
