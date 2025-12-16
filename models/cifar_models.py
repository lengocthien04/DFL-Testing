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
