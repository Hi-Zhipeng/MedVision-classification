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
medvision-cls train configs/train_config.yml
```

### 训练3D模型

```bash
medvision-cls train configs/train_3d_resnet_config.yml
```

### 测试模型

```bash
medvision-cls test configs/test_config.yml
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
