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

### 训练模型

```bash
MedVision-cls train configs/train_config.yml
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

### 训练配置示例

```yaml
# General settings
seed: 42

# Model configuration
model:
  type: "classification"
  network:
    name: "resnet50"
    pretrained: true
  num_classes: 2
  loss:
    type: "cross_entropy"
    weight: null
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    type: "cosine"
    T_max: 100
  metrics:
    accuracy:
      type: "accuracy"
      task: "binary"
    f1:
      type: "f1"
      task: "binary"
    auc:
      type: "auroc"
      task: "binary"

# Data configuration
data:
  type: "medical"
  batch_size: 16
  num_workers: 4
  data_dir: "data/classification"
  train_val_test_split: [0.7, 0.2, 0.1]
  dataset_args:
    image_size: [224, 224]
    normalize: true
    augment: true

# Training configuration
trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1
  precision: 16
  log_every_n_steps: 10
  val_check_interval: 1.0
  callbacks:
    early_stopping:
      monitor: "val/val_loss"
      patience: 10
      mode: "min"
    model_checkpoint:
      monitor: "val/val_accuracy"
      mode: "max"
      save_top_k: 3
      filename: "epoch_{epoch:02d}-val_acc_{val/val_accuracy:.3f}"
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
