"""
ResNet architecture for Fashion-MNIST classification.

Modified ResNet-18 implementation adapted for Fashion-MNIST:
    - Single-channel grayscale input (not RGB)
    - Modified first conv: 3x3 kernel with stride=1 (not 7x7, stride=2)
    - No initial max pooling (preserves spatial information for 28x28 images)
    - 10-class output for Fashion-MNIST categories

Components:
    - BasicBlock: Residual block with skip connections
    - ResNetFashionMNIST: Complete ResNet-18 architecture
    - create_resnet_fashion_mnist(): Factory function for model creation

Author: Michael Gratius - PhD Application - AutoML Research Group
Year: 2025
"""

# Standard library
from typing import List, Optional

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules
import config

# =============================================================================
# BASIC RESIDUAL BLOCK
# =============================================================================

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet.
    
    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+skip) -> ReLU -> out
        |                                                   |
        +-------------------skip connection-----------------+
    
    The skip connection is identity if dimensions match, otherwise uses
    1x1 Conv projection for downsampling.
    """
    
    expansion = 1  
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer 
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False 
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer 
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through basic block.
        
        Args:
            x (torch.Tensor): Input of shape (N, in_channels, H, W).
        
        Returns:
            torch.Tensor: Output of shape 
                (N, out_channels, H/stride, W/stride).
        """
        identity = x
        
        # Main path: Conv -> BN -> ReLU -> Conv -> BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection and apply ReLU
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


# =============================================================================
# RESNET FOR FASHION-MNIST
# =============================================================================

class ResNetFashionMNIST(nn.Module):
    """
    Modified ResNet-18 for Fashion-MNIST classification.
    
    Architecture differences from standard ResNet-18:
        1. Input: 1 channel (grayscale) 
        2. First conv: 3x3 kernel, stride=1 
        3. No initial max pooling 
        4. Output: 10 classes (Fashion-MNIST categories)
    
    Layer structure:
        - conv1: 1 -> 64, 3x3, stride=1
        - layer1: 2x BasicBlock (64 -> 64), no downsampling
        - layer2: 2x BasicBlock (64 -> 128), stride=2 
        - layer3: 2x BasicBlock (128 -> 256), stride=2 
        - layer4: 2x BasicBlock (256 -> 512), stride=2 
        - avgpool: Adaptive average pooling to 1x1
        - fc: 512 -> 10
    
    Spatial dimensions:
        - Input: (N, 1, 28, 28)
        - After conv1: (N, 64, 28, 28)
        - After layer1: (N, 64, 28, 28)
        - After layer2: (N, 128, 14, 14)
        - After layer3: (N, 256, 7, 7)
        - After layer4: (N, 512, 3, 3) 
        - After avgpool: (N, 512, 1, 1)
        - After fc: (N, 10)
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        layers: List[int] = config.RESNET_LAYERS
    ):
        """
        Initialize ResNet for Fashion-MNIST.
        
        Args:
            num_classes (int, optional): Number of output classes. 
                Defaults to config.NUM_CLASSES (10).
            layers (List[int], optional): Number of blocks per layer. 
                Defaults to config.RESNET_LAYERS ([2,2,2,2] for ResNet-18).
        """
        super(ResNetFashionMNIST, self).__init__()
        
        self.in_channels = 64  # Track current number of channels
        
        # Initial convolutional layer (modified for 28x28 input)
        self.conv1 = nn.Conv2d(
            in_channels=1,      
            out_channels=64,
            kernel_size=3,      
            stride=1,           
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Global average pooling (adaptive to any spatial size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple BasicBlocks.
        
        Args:
            out_channels (int): Output channels for this layer.
            num_blocks (int): Number of BasicBlocks in layer.
            stride (int, optional): Stride for first block (1 or 2). 
                Defaults to 1.
        
        Returns:
            nn.Sequential: Sequential module containing all blocks.
        """
        downsample = None
        
        # If dimensions change, need projection for skip connection
        needs_projection = (stride != 1 or 
                   self.in_channels != out_channels * BasicBlock.expansion)
        if needs_projection:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(
            BasicBlock(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels * BasicBlock.expansion
        
        # Remaining blocks (no downsampling)
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(self.in_channels, out_channels)
            )
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using Kaiming initialization.

        Uses Kaiming initialization for Conv2d and Linear layers,
        which is the standard for ReLU-based networks and prevents 
        vanishing/exploding gradients.
        
        Initialization scheme:
            - Conv2d: Kaiming normal (mode='fan_out', nonlinearity='relu')
            - BatchNorm2d: weight=1, bias=0 
            - Linear: Kaiming normal (mode='fan_out', nonlinearity='relu')
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal initialization for conv layers
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm weights to 1, bias to 0
                # This makes BatchNorm initially act as identity transform
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Kaiming normal for classification head (consistent with conv)
                # Uses fan_out mode for better gradient flow during backprop
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, 28, 28).
        
        Returns:
            torch.Tensor: Output logits of shape (N, 10).
        """
        # Initial conv layer
        x = self.conv1(x)       # (batch, 64, 28, 28)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual layers
        x = self.layer1(x)      # (batch, 64, 28, 28)
        x = self.layer2(x)      # (batch, 128, 14, 14)
        x = self.layer3(x)      # (batch, 256, 7, 7)
        x = self.layer4(x)      # (batch, 512, 3, 3) 
        
        # Global average pooling
        x = self.avgpool(x)     # (batch, 512, 1, 1)
        x = torch.flatten(x, 1) # (batch, 512)
        
        # Classification layer
        x = self.fc(x)          # (batch, 10)
        
        return x


# =============================================================================
# MODEL CREATION HELPER
# =============================================================================

def create_resnet_fashion_mnist(
    num_classes: int = config.NUM_CLASSES,
    layers: List[int] = config.RESNET_LAYERS,
    device: torch.device = config.DEVICE
) -> ResNetFashionMNIST:
    """
    Create and initialize ResNet model for Fashion-MNIST.
    
    Args:
        num_classes (int, optional): Number of output classes. 
            Defaults to config.NUM_CLASSES (10).
        layers (List[int], optional): Blocks per layer. 
            Defaults to config.RESNET_LAYERS ([2,2,2,2]).
        device (torch.device, optional): Device to place model on. 
            Defaults to config.DEVICE.
    
    Returns:
        ResNetFashionMNIST: Initialized model on specified device.
    """
    model = ResNetFashionMNIST(num_classes=num_classes, layers=layers)
    model = model.to(device)
       
    return model

