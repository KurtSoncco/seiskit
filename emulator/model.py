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
        
        ============================================================================
        MODEL ARCHITECTURE CONFIGURATION
        ============================================================================
        To modify the model architecture, edit the sections below:
        
        1. Input size: Change 'input_size' parameter (line 68)
        2. Initial conv layer: Line 79 (channels, kernel_size, stride)
        3. Residual layers: Lines 85-88 (number of blocks, channels, stride)
        4. MLP head: Lines 94-102 (hidden dimensions, dropout rates, activation)
        5. ResidualBlock: Lines 13-55 (conv layers, batch norm, activation)
        ============================================================================
        """
        super().__init__()
        self.input_size = input_size

        # Initial convolution layer (1 channel input)
        # MODIFY: Change channels (64), kernel_size (7), stride (1) here
        # Changed stride from 2 to 1 to preserve high-frequency information (Option B)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks (ResNet-18: [2, 2, 2, 2] blocks)
        # MODIFY: Change number of blocks, channels, stride here
        # Format: _make_layer(in_channels, out_channels, num_blocks, stride)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP head for regression
        # MODIFY: Change hidden dimensions (256, 128), dropout (0.5), activation here
        self.fc = nn.Sequential(
            nn.Linear(512, 256),      # First hidden layer
            nn.ReLU(),                # Activation
            nn.Dropout(0.5),          # Dropout rate
            nn.Linear(256, 128),      # Second hidden layer
            nn.ReLU(),                # Activation
            nn.Dropout(0.5),          # Dropout rate
            nn.Linear(128, num_classes),  # Output layer
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


# Alias for backward compatibility (TransferFunctionEmulator not yet implemented)
# TODO: Implement TransferFunctionEmulator for time-series prediction
TransferFunctionEmulator = PGAEmulator
