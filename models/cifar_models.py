import torch.nn as nn
import torch.nn.functional as F

class GNLeNet(nn.Module):
    def __init__(self, num_classes: int = 10, gn_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(gn_groups, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(gn_groups, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.gn3 = nn.GroupNorm(gn_groups, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.gn3(self.fc1(x).unsqueeze(-1))).squeeze(-1)
        return self.fc2(x)

class CifarConvNet(nn.Module):
    """Lightweight 3-layer CNN for CIFAR-10 (from p2pfl)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Conv block 2: 32 -> 64, with maxpool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        # Conv block 3: 64 -> 128, with maxpool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        # Classifier: 128*8*8 -> 256 -> 10
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
