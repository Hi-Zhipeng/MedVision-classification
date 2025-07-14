"""
3D ResNet models for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseClassifier


class BasicBlock3D(nn.Module):
    """3D Basic Block for ResNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """3D Bottleneck Block for ResNet"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """3D ResNet backbone"""
    
    def __init__(self, block, layers, in_channels=1, shortcut_type='B'):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = nn.Sequential(
                    nn.AvgPool3d(kernel_size=1, stride=stride),
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        return x


class ResNet3DClassifier(BaseClassifier):
    """3D ResNet classifier for medical volume data"""
    
    def __init__(
        self,
        model_name: str = "resnet3d_18",
        num_classes: int = 2,
        pretrained: bool = False,  # No pretrained weights for 3D models
        in_channels: int = 1,
        dropout: float = 0.0,
        shortcut_type: str = 'B',
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Create backbone and classifier
        self._create_backbone(
            model_name=model_name,
            in_channels=in_channels,
            shortcut_type=shortcut_type,
            **kwargs
        )
        
        feature_dim = self._get_feature_dim_3d(in_channels)
        self._create_classifier(feature_dim, num_classes, dropout)
    
    def _create_backbone(self, model_name: str, in_channels: int, shortcut_type: str = 'B', **kwargs):
        """Create the 3D ResNet backbone"""
        
        # Define model architectures
        model_configs = {
            "resnet3d_18": (BasicBlock3D, [2, 2, 2, 2]),
            "resnet3d_34": (BasicBlock3D, [3, 4, 6, 3]),
            "resnet3d_50": (Bottleneck3D, [3, 4, 6, 3]),
            "resnet3d_101": (Bottleneck3D, [3, 4, 23, 3]),
            "resnet3d_152": (Bottleneck3D, [3, 8, 36, 3]),
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown 3D ResNet model: {model_name}")
        
        block, layers = model_configs[model_name]
        self.backbone = ResNet3D(
            block=block,
            layers=layers,
            in_channels=in_channels,
            shortcut_type=shortcut_type
        )
    
    def _create_classifier(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        """Create the classifier head"""
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
    
    def _get_feature_dim_3d(self, in_channels: int = 1):
        """Get the feature dimension of the 3D backbone"""
        with torch.no_grad():
            # Use a smaller dummy input for 3D (memory considerations)
            dummy_input = torch.randn(1, in_channels, 32, 64, 64)
            features = self.backbone(dummy_input)
            return features.shape[1]


def resnet3d_18(**kwargs):
    """3D ResNet-18 model"""
    return ResNet3DClassifier(model_name="resnet3d_18", **kwargs)


def resnet3d_34(**kwargs):
    """3D ResNet-34 model"""
    return ResNet3DClassifier(model_name="resnet3d_34", **kwargs)


def resnet3d_50(**kwargs):
    """3D ResNet-50 model"""
    return ResNet3DClassifier(model_name="resnet3d_50", **kwargs)


def resnet3d_101(**kwargs):
    """3D ResNet-101 model"""
    return ResNet3DClassifier(model_name="resnet3d_101", **kwargs)


def resnet3d_152(**kwargs):
    """3D ResNet-152 model"""
    return ResNet3DClassifier(model_name="resnet3d_152", **kwargs)
