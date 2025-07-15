# MedVision-Classification

MedVision-Classification 是一个基于 PyTorch Lightning 的医学影像分类框架，提供了训练和推理的简单接口。

## 特点

- 基于 PyTorch Lightning 的高级接口
- 支持常见的医学影像格式（NIfTI、DICOM 等）
- 内置多种分类模型架构（ResNet、DenseNet、EfficientNet 等）
- 灵活的数据加载和预处理管道
- 模块化设计，易于扩展
- 命令行界面用于训练和推理
- 支持二分类和多分类任务

## 安装

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 基本安装

最简单的安装方式：

```bash
pip install -e .
```

### 从源码安装

```bash
git clone https://github.com/yourusername/medvision-classification.git
cd medvision-classification
pip install -e .
```

### 使用requirements文件

```bash
# 基本环境
pip install -r requirements.txt

# 开发环境
pip install -r requirements-dev.txt
```

### 使用conda环境

推荐使用 conda 创建独立的虚拟环境：

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate medvision-cls

# 安装项目本身
pip install -e .
```

## 快速入门

### 训练2D模型

```bash
MedVision-cls train configs/train_config.yml
```

### 训练3D模型

```bash
MedVision-cls train configs/train_3d_resnet_config.yml
```

### 测试模型

```bash
MedVision-cls test configs/test_config.yml
```

### 推理

```bash
MedVision-cls predict configs/inference_config.yml --input /path/to/image --output /path/to/output
```

## 配置格式

### 2D分类训练配置示例

```yaml
# 2D ResNet Training Configuration
seed: 42

task_dim: 2d

# Model configuration
model:
  type: "classification"
  network:
    name: "resnet50"
    pretrained: true
  num_classes: 2

  # Metrics to compute
  metrics:
    accuracy:
      type: "accuracy"
    f1:
      type: "f1"
    precision:
      type: "precision"
    recall:
      type: "recall"
    auc:
      type: "auroc"
        
  # Loss configuration
  loss:
    type: "cross_entropy"
    weight: null
    label_smoothing: 0.0
  
  # Optimizer configuration
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001
  
  # Scheduler configuration
  scheduler:
    type: "cosine"
    T_max: 100
    eta_min: 0.00001

# Data configuration
data:
  type: "medical"
  batch_size: 4
  num_workers: 4
  data_dir: "data/classification"
  image_format: "*.png"
  
  # Transform configuration for 2D data
  transforms:
    image_size: [224, 224]
    normalize: true
    augment: true
    
  # Data split configuration
  train_val_split: [0.8, 0.2]
  seed: 42

# Training configuration
training:
  max_epochs: 10
  accelerator: "gpu"
  devices: [0,1,2,3]  # Multi-GPU training
  precision: 16
  save_metrics: true
  
  # Callbacks
  model_checkpoint:
    monitor: "val/accuracy"
    mode: "max"
    save_top_k: 3
    filename: "epoch_{epoch:02d}-val_acc_{val/accuracy:.3f}"

# Validation configuration
validation:
  check_val_every_n_epoch: 1

# Class names
class_names:
  - "Class_0"
  - "Class_1"

# Output paths
outputs:
  output_dir: "outputs"
  checkpoint_dir: "outputs/checkpoints"
  log_dir: "outputs/logs"

# Logging
logging:
  log_every_n_steps: 10
  wandb:
    enabled: false
    project: "medvision-2d-classification"
    entity: null
```

### 3D分类训练配置示例

```yaml
# 3D ResNet Training Configuration
seed: 42

task_dim: 3D

# Model configuration
model:
  type: "classification"
  network:
    name: "resnet3d_18"  # Options: resnet3d_18, resnet3d_34, resnet3d_50
    pretrained: false    # No pretrained weights for 3D models
    in_channels: 3       # Input channels (typically 1 for medical images)
    dropout: 0.1
  num_classes: 2

  # Metrics to compute
  metrics:
    accuracy:
      type: "accuracy"
    f1:
      type: "f1"
    precision:
      type: "precision"
    recall:
      type: "recall"
    auc:
      type: "auroc"

  # Loss configuration
  loss:
    type: "cross_entropy"
    weight: null
    label_smoothing: 0.0
  
  # Optimizer configuration
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001
  
  # Scheduler configuration
  scheduler:
    type: "cosine"
    T_max: 100
    eta_min: 0.00001

# Data configuration
data:
  type: "medical"
  batch_size: 4         # Smaller batch size for 3D data
  num_workers: 4
  data_dir: "data/3D"
  image_format: "*.nii.gz"  # 3D medical image format
  
  # Transform configuration for 3D data
  transforms:
    image_size: [64, 64, 64]  # [D, H, W] for 3D volumes
    normalize: true
    augment: true
    
  # Data split configuration
  train_val_split: [0.8, 0.2]
  seed: 42

# Training configuration
training:
  max_epochs: 5
  accelerator: "gpu"
  devices: 1            # Single GPU for 3D (memory intensive)
  precision: 16         # Use mixed precision to save memory
  
  # Callbacks
  early_stopping:
    monitor: "val/loss"
    patience: 10
    mode: "min"
  
  model_checkpoint:
    monitor: "val/accuracy"
    mode: "max"
    save_top_k: 3
    filename: "epoch_{epoch:02d}-val_acc_{val/accuracy:.3f}"

# Validation configuration
validation:
  check_val_every_n_epoch: 1

# Output paths
outputs:
  output_dir: "outputs"
  checkpoint_dir: "outputs/checkpoints"
  log_dir: "outputs/logs"

# Logging
logging:
  log_every_n_steps: 10
  wandb:
    enabled: false
    project: "medvision-3d-classification"
    entity: null
```

### 推理配置示例

```yaml
# Model configuration
model:
  type: "classification"
  network:
    name: "resnet50"
    pretrained: false
  num_classes: 2
  checkpoint_path: "outputs/checkpoints/best_model.ckpt"

# Inference settings
inference:
  batch_size: 1
  device: "cuda:0"  # 或 "cpu"
  return_probabilities: true
  class_names: ["class0", "class1"]
  confidence_threshold: 0.5

# Preprocessing
preprocessing:
  image_size: [224, 224]
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# Output settings
output:
  save_predictions: true
  include_probabilities: true
  format: "json"  # 或 "csv"
```

## 数据格式

### 文件夹结构

```
data/
├── classification/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── image1.png
│   │   │   └── image2.png
│   │   └── class2/
│   │       ├── image3.png
│   │       └── image4.png
│   ├── val/
│   │   ├── class1/
│   │   └── class2/
│   └── test/
│       ├── class1/
│       └── class2/
```

### CSV格式

```csv
image_path,label
/path/to/image1.png,0
/path/to/image2.png,1
/path/to/image3.png,0
```

## 支持的模型

- **ResNet系列**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **DenseNet系列**: DenseNet121, DenseNet161, DenseNet169, DenseNet201
- **EfficientNet系列**: EfficientNet-B0 到 EfficientNet-B7
- **Vision Transformer**: ViT-Base, ViT-Large
- **ConvNeXt**: ConvNeXt-Tiny, ConvNeXt-Small, ConvNeXt-Base
- **Medical专用**: MedNet, RadImageNet预训练模型

## 许可证

本项目基于 MIT 许可证开源。

## 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 引用

如果您在研究中使用了本框架，请引用：

```bibtex
@software{medvision_classification,
  title={MedVision-Classification: A PyTorch Lightning Framework for Medical Image Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/Hi-Zhipeng/MedVision-classification}
}
```


"""
PyTorch Lightning module for medical image classification
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from typing import Dict, Any
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC

from .model_factory import create_model
from ..losses import create_loss


class ClassificationLightningModule(pl.LightningModule):
    """PyTorch Lightning module for medical image classification"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        loss_config: Dict[str, Any] = None,
        optimizer_config: Dict[str, Any] = None,
        scheduler_config: Dict[str, Any] = None,
        metrics_config: Dict[str, Any] = None,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **model_kwargs
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
                task = "binary" if self.hparams.num_classes == 2 else "multiclass"
                
                if metric_type == "accuracy":
                    metric = Accuracy(task=task, num_classes=self.hparams.num_classes)
                elif metric_type == "f1":
                    metric = F1Score(task=task, num_classes=self.hparams.num_classes, average="macro")
                elif metric_type == "precision":
                    metric = Precision(task=task, num_classes=self.hparams.num_classes, average="macro")
                elif metric_type == "recall":
                    metric = Recall(task=task, num_classes=self.hparams.num_classes, average="macro")
                elif metric_type == "auroc":
                    metric = AUROC(task=task, num_classes=self.hparams.num_classes)
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
                if self.hparams.num_classes == 2:
                    metric.update(preds[:, 1], labels)
                else:
                    metric.update(preds, labels)
            else:
                # Other metrics need predicted class indices
                metric.update(pred_classes, labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    # def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     images, labels = batch["image"], batch["label"]
    #     logits = self(images)
    #     loss = self.loss_fn(logits, labels)
        
    #     # Update metrics
    #     self._update_metrics(self.train_metrics, logits, labels)
        
    #     # Log loss
    #     self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
    #     self.train_step_outputs.append(loss)
    #     return loss
    
    # def on_train_epoch_end(self):
    #     # Compute and log metrics
    #     for metric_name, metric in self.train_metrics.items():
    #         value = metric.compute()
    #         self.log(f"train/{metric_name}", value, prog_bar=True)
    #         metric.reset()
        
    #     self.train_step_outputs.clear()
    
    # def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     images, labels = batch["image"], batch["label"]

    #     logits = self(images)
    #     loss = self.loss_fn(logits, labels)
        
    #     # Update metrics
    #     self._update_metrics(self.val_metrics, logits, labels)
        
    #     # Log loss
    #     self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
    #     self.val_step_outputs.append(loss)
    #     return loss
    
    # def on_validation_epoch_end(self):
    #     # Compute and log metrics
    #     for metric_name, metric in self.val_metrics.items():
    #         value = metric.compute()
    #         self.log(f"val/{metric_name}", value, prog_bar=True, on_epoch=True)
    #         metric.reset()
        
    #     self.val_step_outputs.clear()
    
    # def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     images, labels = batch["image"], batch["label"]
    #     logits = self(images)
    #     loss = self.loss_fn(logits, labels)
        
    #     # Update metrics
    #     self._update_metrics(self.test_metrics, logits, labels)
        
    #     # Log loss
    #     self.log("test/loss", loss, on_step=False, on_epoch=True)
        
    #     self.test_step_outputs.append({
    #         "loss": loss,
    #         "preds": torch.softmax(logits, dim=1),
    #         "labels": labels
    #     })
    #     return loss
    
    # def on_test_epoch_end(self):
    #     # Compute metrics without logging to avoid deadlock
    #     for metric_name, metric in self.test_metrics.items():
    #         value = metric.compute()
    #         # Store metric values for later use if needed
    #         # self.log(f"test/{metric_name}", value)  # Removed to avoid deadlock
    #         metric.reset()
        
    #     self.test_step_outputs.clear()

# ...existing code...

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self._update_metrics(self.train_metrics, logits, labels)
        
        # Log loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log metrics in the step itself to avoid deadlock
        for metric_name, metric in self.train_metrics.items():
            self.log(f"train/{metric_name}", metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        self.train_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        # Only reset metrics, don't log here to avoid deadlock
        for metric in self.train_metrics.values():
            metric.reset()
        
        self.train_step_outputs.clear()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self._update_metrics(self.val_metrics, logits, labels)
        
        # Log loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log metrics in the step itself to avoid deadlock
        for metric_name, metric in self.val_metrics.items():
            self.log(f"val/{metric_name}", metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        # Only reset metrics, don't log here to avoid deadlock
        for metric in self.val_metrics.values():
            metric.reset()
        
        self.val_step_outputs.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self._update_metrics(self.test_metrics, logits, labels)
        
        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log metrics in the step itself to avoid deadlock
        for metric_name, metric in self.test_metrics.items():
            self.log(f"test/{metric_name}", metric, on_step=False, on_epoch=True, sync_dist=True)
        
        self.test_step_outputs.append({
            "loss": loss,
            "preds": torch.softmax(logits, dim=1),
            "labels": labels
        })
        return loss

    def on_test_epoch_end(self):
        # Only reset metrics, don't log here to avoid deadlock
        
        for metric in self.test_metrics.values():
            metric.reset()

        self.test_step_outputs.clear()

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
