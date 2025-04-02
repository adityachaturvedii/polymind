"""
Computer Vision models module for the AI Agent System.

This module provides model implementations and utilities for computer vision tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class CVModelFactory:
    """
    Factory class for creating computer vision models.
    """

    @staticmethod
    def create_model(
        model_type: str,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a computer vision model.

        Args:
            model_type: Type of model to create
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters

        Returns:
            nn.Module: The created model
        """
        model_type = model_type.lower()
        
        if model_type == "resnet18":
            return CVModelFactory._create_resnet(18, num_classes, pretrained, **kwargs)
        elif model_type == "resnet34":
            return CVModelFactory._create_resnet(34, num_classes, pretrained, **kwargs)
        elif model_type == "resnet50":
            return CVModelFactory._create_resnet(50, num_classes, pretrained, **kwargs)
        elif model_type == "resnet101":
            return CVModelFactory._create_resnet(101, num_classes, pretrained, **kwargs)
        elif model_type == "efficientnet_b0":
            return CVModelFactory._create_efficientnet("b0", num_classes, pretrained, **kwargs)
        elif model_type == "efficientnet_b1":
            return CVModelFactory._create_efficientnet("b1", num_classes, pretrained, **kwargs)
        elif model_type == "efficientnet_b2":
            return CVModelFactory._create_efficientnet("b2", num_classes, pretrained, **kwargs)
        elif model_type == "mobilenet_v2":
            return CVModelFactory._create_mobilenet_v2(num_classes, pretrained, **kwargs)
        elif model_type == "custom_cnn":
            return CVModelFactory._create_custom_cnn(num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _create_resnet(
        depth: int,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a ResNet model.

        Args:
            depth: Depth of the ResNet (18, 34, 50, 101)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters

        Returns:
            nn.Module: The created ResNet model
        """
        if depth == 18:
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        elif depth == 34:
            model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        elif depth == 50:
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        elif depth == 101:
            model = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Modify the final layer for the number of classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model

    @staticmethod
    def _create_efficientnet(
        version: str,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create an EfficientNet model.

        Args:
            version: Version of EfficientNet (b0, b1, b2, etc.)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters

        Returns:
            nn.Module: The created EfficientNet model
        """
        if version == "b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        elif version == "b1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V1" if pretrained else None)
        elif version == "b2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unsupported EfficientNet version: {version}")
        
        # Modify the classifier for the number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        return model

    @staticmethod
    def _create_mobilenet_v2(
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a MobileNetV2 model.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters

        Returns:
            nn.Module: The created MobileNetV2 model
        """
        model = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        
        # Modify the classifier for the number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        return model

    @staticmethod
    def _create_custom_cnn(
        num_classes: int,
        input_channels: int = 3,
        **kwargs
    ) -> nn.Module:
        """
        Create a custom CNN model.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            **kwargs: Additional model-specific parameters

        Returns:
            nn.Module: The created custom CNN model
        """
        return CustomCNN(input_channels, num_classes, **kwargs)


class CustomCNN(nn.Module):
    """
    Custom CNN model for computer vision tasks.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        hidden_channels: List[int] = [32, 64, 128],
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the CustomCNN model.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            hidden_channels: List of hidden channel dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First convolutional layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )
        
        # Additional convolutional layers
        for i in range(len(hidden_channels) - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x


class ObjectDetectionModel(nn.Module):
    """
    Object detection model based on Faster R-CNN.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ):
        """
        Initialize the ObjectDetectionModel.

        Args:
            num_classes: Number of object classes (including background)
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        
        # Load Faster R-CNN model
        self.model = models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            **kwargs
        )
        
        # Replace the classifier with a new one for the specified number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(self, x: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass of the model.

        Args:
            x: List of input tensors
            targets: Optional list of target dictionaries

        Returns:
            Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]: Model output
        """
        return self.model(x, targets)


class SegmentationModel(nn.Module):
    """
    Segmentation model based on DeepLabV3.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ):
        """
        Initialize the SegmentationModel.

        Args:
            num_classes: Number of segmentation classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        
        # Load DeepLabV3 model
        self.model = models.segmentation.deeplabv3_resnet50(
            weights="DEFAULT" if pretrained else None,
            **kwargs
        )
        
        # Replace the classifier with a new one for the specified number of classes
        self.model.classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            Dict[str, torch.Tensor]: Model output
        """
        return self.model(x)


def get_cv_transforms(
    task_type: str = "classification",
    input_size: Tuple[int, int] = (224, 224),
    augmentation: bool = True,
) -> Dict[str, transforms.Compose]:
    """
    Get standard transforms for computer vision tasks.

    Args:
        task_type: Type of CV task (classification, detection, segmentation)
        input_size: Input image size (height, width)
        augmentation: Whether to include data augmentation

    Returns:
        Dict[str, transforms.Compose]: Dictionary of transforms for train and val
    """
    # Basic normalization for ImageNet pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Training transforms
    if augmentation:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transforms = val_transforms
    
    return {
        "train": train_transforms,
        "val": val_transforms,
    }
