import torch.nn as nn
import torch.nn.functional as F

class LogisticMNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class MnistLinear(nn.Module):
    """Single linear layer for MNIST (from p2pfl)."""
    def __init__(self, input_shape=(1, 28, 28), num_classes: int = 10):
        super().__init__()
        features = 1
        for dim in input_shape:
            features *= dim
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
