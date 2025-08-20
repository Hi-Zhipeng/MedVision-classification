"""
PyTorch Lightning module for medical image classification
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from typing import Dict, Any
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC, Specificity

from .model_factory import create_model
from ..losses import create_loss


class ClassificationLightningModule(pl.LightningModule):
    """PyTorch Lightning module for medical image classification"""
    
    def __init__(
        self,
        model_config: Dict[str, Any] = None,
        loss_config: Dict[str, Any] = None,
        optimizer_config: Dict[str, Any] = None,
        scheduler_config: Dict[str, Any] = None,
        metrics_config: Dict[str, Any] = None,
        **model_kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = model_config.get("num_classes", 2)

        # Remove num_classes from model_kwargs to avoid duplicate arguments
        model_kwargs_filtered = {k: v for k, v in model_kwargs.items() if k != 'num_classes'}
        
        # Extract network-specific parameters
        network_config = model_config.get('network', {})
        network_kwargs = {k: v for k, v in network_config.items() 
                         if k not in ['name', 'pretrained']}
        
        # Merge network kwargs with model kwargs
        final_kwargs = {**network_kwargs, **model_kwargs_filtered}

        # Create model
        self.model = create_model(
            model_name=network_config.get("name", "resnet50"),
            num_classes=self.num_classes,
            pretrained=network_config.get("pretrained", True),
            **final_kwargs
        )

        # Setup loss
        loss_config = loss_config or {"type": "cross_entropy"}
        self.loss_fn = create_loss(loss_config)
        
        # Setup metrics
        metrics_config = metrics_config or {}
        
        self.setup_metrics(metrics_config)
        
        # Store configs
        self.optimizer_config = optimizer_config or {"type": "adam", "lr": 1e-3}
        self.scheduler_config = scheduler_config or {"type": "cosine", "T_max": 100}
        
        # For logging
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def setup_metrics(self, metrics_config: Dict[str, Any]):
        """Setup metrics for training, validation, and testing"""
        
        print(f"Setting up metrics with config: {metrics_config}")
        
        # Create metric collections using torchmetrics directly
        train_metrics = {}
        val_metrics = {}
        test_metrics = {}
        
        # Only create metrics specified in config
        for metric_name, metric_config in metrics_config.items():
            print(f"Creating metric: {metric_name} with config: {metric_config}")
            try:
                metric_type = metric_config.get("type", "accuracy").lower()
                task = "binary" if self.num_classes == 2 else "multiclass"
                
                if metric_type == "accuracy":
                    metric = Accuracy(task=task, num_classes=self.num_classes)
                elif metric_type == "f1":
                    metric = F1Score(task=task, num_classes=self.num_classes, average="macro")
                elif metric_type == "precision":
                    metric = Precision(task=task, num_classes=self.num_classes, average="macro")
                elif metric_type == "recall":
                    metric = Recall(task=task, num_classes=self.num_classes, average="macro")
                elif metric_type == "auroc":
                    metric = AUROC(task=task, num_classes=self.num_classes)
                elif metric_type == "specificity":
                    metric = Specificity(task=task, num_classes=self.num_classes, average="macro")
                elif metric_type == "sensitivity" or metric_type == "recall":
                    # Sensitivity is same as Recall
                    metric = Recall(task=task, num_classes=self.num_classes, average="macro")
                elif metric_type == "npv":
                    # NPV can be calculated using specificity and prevalence, but torchmetrics doesn't have direct NPV
                    # We'll use a custom implementation through our metrics module
                    from ..metrics import NPVMetric
                    metric = NPVMetric(num_classes=self.num_classes)
                elif metric_type == "ppv":
                    # PPV is same as Precision
                    metric = Precision(task=task, num_classes=self.num_classes, average="macro")
                else:
                    print(f"⚠️  Unknown metric type: {metric_type}, skipping")
                    continue
                
                train_metrics[metric_name] = metric.clone()
                val_metrics[metric_name] = metric.clone()
                test_metrics[metric_name] = metric.clone()
                print(f"✅ Created and cloned {metric_name}")
                    
            except Exception as e:
                print(f"❌ Could not create metric {metric_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create ModuleDict collections
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)
        self.test_metrics = nn.ModuleDict(test_metrics)
        
        print(f"Final train_metrics: {list(self.train_metrics.keys())}")
        print(f"Final val_metrics: {list(self.val_metrics.keys())}")
        print(f"Final test_metrics: {list(self.test_metrics.keys())}")
    
    def _update_metrics(self, metrics: nn.ModuleDict, logits: torch.Tensor, labels: torch.Tensor):
        """Update metrics with predictions and labels"""
        preds = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(preds, dim=1)
        
        for metric_name, metric in metrics.items():
            if metric_name == "auc":
                # AUC needs probabilities - for binary: preds[:, 1], for multiclass: preds
                if self.num_classes == 2:
                    metric.update(preds[:, 1], labels)
                else:
                    metric.update(preds, labels)
            else:
                # Other metrics need predicted class indices
                metric.update(pred_classes, labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]

        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        # 计算预测结果
        preds = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(preds, dim=1)
        
        batch_size = images.size(0)
        
        # Log loss with sync_dist for multi-GPU
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)
        
        # 计算并记录训练指标
        for metric_name, metric in self.train_metrics.items():
            try:
                if metric_name == "auc":
                    if self.num_classes == 2:
                        metric_value = metric(preds[:, 1], labels)
                    else:
                        metric_value = metric(preds, labels)
                else:
                    metric_value = metric(pred_classes, labels)
                
                self.log(f"train/{metric_name}", metric_value, on_step=False, on_epoch=True,
                         prog_bar=False, batch_size=batch_size, sync_dist=True)
                         
            except Exception as e:
                print(f"Warning: Failed to compute training metric {metric_name}: {e}")
        
        return loss
    
    def on_train_epoch_end(self):
        # 只重置指标，不进行日志记录
        for metric in self.train_metrics.values():
            metric.reset()
        
        if hasattr(self, 'train_step_outputs'):
            self.train_step_outputs.clear()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]

        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        # 计算预测结果
        preds = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(preds, dim=1)
        
        batch_size = images.size(0)
        
        # Log loss with sync_dist for multi-GPU
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=batch_size, sync_dist=True)
        
        # 计算并记录验证指标
        for metric_name, metric in self.val_metrics.items():
            try:
                if metric_name == "auc":
                    if self.num_classes == 2:
                        metric_value = metric(preds[:, 1], labels)
                    else:
                        metric_value = metric(preds, labels)
                else:
                    metric_value = metric(pred_classes, labels)
                
                self.log(f"val/{metric_name}", metric_value, prog_bar=True, on_step=False, on_epoch=True,
                         batch_size=batch_size, sync_dist=True)
                         
            except Exception as e:
                print(f"Warning: Failed to compute validation metric {metric_name}: {e}")
        
        return loss
    
    def on_validation_epoch_end(self):
        # 只重置指标，不进行日志记录以避免死锁
        for metric in self.val_metrics.values():
            metric.reset()
        
        if hasattr(self, 'val_step_outputs'):
            self.val_step_outputs.clear()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self._update_metrics(self.test_metrics, logits, labels)
        
        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        self.test_step_outputs.append({
            "loss": loss,
            "preds": torch.softmax(logits, dim=1),
            "labels": labels
        })
        return loss
    
    def on_test_epoch_end(self):
        # Compute metrics without logging to avoid deadlock
        for metric_name, metric in self.test_metrics.items():
            value = metric.compute()
            # Store metric values for later use if needed
            self.log(f"test/{metric_name}", value)  # Removed to avoid deadlock
            metric.reset()
        
        # Plot ROC curves if we have test step outputs
        if self.test_step_outputs and self.num_classes == 2:
            self._plot_roc_curve()
        
        self.test_step_outputs.clear()
    
    def _plot_roc_curve(self):
        """Plot ROC curve for test data"""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
            import numpy as np
            import os
            
            # Collect all predictions and labels
            all_probs = []
            all_labels = []
            
            for output in self.test_step_outputs:
                probs = output["preds"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                # For binary classification, use probability of positive class
                if self.num_classes == 2:
                    all_probs.extend(probs[:, 1])  # Probability of class 1
                else:
                    all_probs.extend(probs)
                all_labels.extend(labels)
            
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            
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
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
                save_dir = self.logger.log_dir
            else:
                save_dir = "outputs"
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "roc_curve.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ ROC curve saved to: {save_path}")
            print(f"📊 Test AUC: {roc_auc:.4f}")
            
            # Find optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_sensitivity = tpr[optimal_idx]
            optimal_specificity = 1 - fpr[optimal_idx]
            
            print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
            print(f"   Sensitivity: {optimal_sensitivity:.4f}")
            print(f"   Specificity: {optimal_specificity:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to plot ROC curve: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        logits = self(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return {
            "predictions": preds,
            "probabilities": probs,
            "logits": logits
        }
    
    def configure_optimizers(self):
        # Setup optimizer
        optimizer_type = self.optimizer_config.get("type", "adam").lower()
        lr = self.optimizer_config.get("lr", 1e-3)
        weight_decay = self.optimizer_config.get("weight_decay", 0)
        
        if optimizer_type == "adam":
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            momentum = self.optimizer_config.get("momentum", 0.9)
            optimizer = SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup scheduler
        scheduler_type = self.scheduler_config.get("type", "cosine").lower()
        
        if scheduler_type == "cosine":
            T_max = self.scheduler_config.get("T_max", 100)
            eta_min = self.scheduler_config.get("eta_min", 0)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            return [optimizer], [scheduler]
        elif scheduler_type == "plateau":
            patience = self.scheduler_config.get("patience", 10)
            factor = self.scheduler_config.get("factor", 0.5)
            monitor = self.scheduler_config.get("monitor", "val/loss")
            scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif scheduler_type == "step":
            step_size = self.scheduler_config.get("step_size", 30)
            gamma = self.scheduler_config.get("gamma", 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return [optimizer], [scheduler]
        else:
            return optimizer