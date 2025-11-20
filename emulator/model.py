"""PGA Emulator model architecture.

ResNet-18 based architecture for scalar PGA prediction:
- Input: 150x150 Vs field (HF resolution)
- Output: Scalar PGA value
- Architecture: ResNet-18 (modified) → Global Average Pooling → MLP → 1 output
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class PGAEmulator(nn.Module):
    """PGA Emulator model based on ResNet-18 architecture.

    Takes 150x150 Vs field as input and predicts scalar PGA.
    Architecture:
    - Modified ResNet-18 encoder (1 input channel instead of 3)
    - Global Average Pooling
    - MLP head → 1 output (PGA)
    """

    def __init__(self, input_size: tuple[int, int] = (150, 150), num_classes: int = 1):
        """Initialize PGA emulator.

        Args:
            input_size: Size of input Vs field (H, W)
            num_classes: Number of output classes (1 for scalar PGA)
        """
        super().__init__()
        self.input_size = input_size

        # Initial convolution layer (1 channel input)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks (ResNet-18: [2, 2, 2, 2] blocks)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP head for regression
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of residual blocks in the layer
            stride: Stride for the first block

        Returns:
            Sequential container with residual blocks
        """
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input Vs field of shape (B, 1, H, W) where H=W=150

        Returns:
            Predicted PGA of shape (B, 1)
        """
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)

        # MLP head
        x = self.fc(x)  # (B, 1)

        return x
