import torch
import torch.nn as nn
import torch.nn.functional as func

from .base import BaseModel


class SimpleCNN(BaseModel):
    """
    This is our simple CNN that we use as a starting point.
    """

    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)

        # First conv block: input RGB (3) -> 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # First downsampling

        # Second conv block: 16 -> 32 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Second downsampling

        # 128x128 images are 32x32 at this point
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.pool1(x)

        x = func.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BatchNCNN(BaseModel):
    """
    This is the first improvement upon our simple CNN.
    Here we introduce BatchNorm after each convolutional layer.
    """

    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)

        # First conv block: input RGB (3) -> 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv block: 16 -> 32 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool1(func.relu(self.bn1(self.conv1(x))))
        x = self.pool2(func.relu(self.bn2(self.conv2(x))))

        x = torch.flatten(x, 1)
        x = func.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x


class AdvancedBatchNCNN(BaseModel):
    """
    This is an advanced version of the BatchCNN.
    We use this model to experiment with different numbers of layers.
    """

    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)

        # Block 1: input 3 -> 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64

        # Block 2: 16 -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32

        # Block 3: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16

        # Block 4: 64 -> 128 channels
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool1(func.relu(self.bn1(self.conv1(x))))
        x = self.pool2(func.relu(self.bn2(self.conv2(x))))
        x = self.pool3(func.relu(self.bn3(self.conv3(x))))
        x = self.pool4(func.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = func.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResidualCNN(BaseModel):
    """
    This is the implementation of the AdvancedBatchCNN with residual blocks.
    We omit the implementation of the BatchCnn with the residual blocks
    because it showed no improvement in the accuracy and because it looks
    very similar to this implementation.
    """

    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)

        # Input: [3, 128, 128] → Conv → [16, 128, 128]
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # → [16, 64, 64]

        # Conv to expand channels from 16 → 32, then residual block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # → [32, 32, 32]

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.res3 = ResidualBlock(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # → [64, 16, 16]

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res4 = ResidualBlock(128)
        self.pool4 = nn.MaxPool2d(2, 2)  # → [128, 8, 8]

        # Flattened shape: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool1(self.block1(x))  # [16, 64, 64]
        x = self.pool2(self.res2(self.conv2(x)))  # [32, 32, 32]
        x = self.pool3(self.res3(self.conv3(x)))  # [64, 16, 16]
        x = self.pool4(self.res4(self.conv4(x)))  # [128, 8, 8]

        x = torch.flatten(x, 1)
        x = func.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out
