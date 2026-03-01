import torch.nn as nn
import torch.nn.functional as F

class LogisticMNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class MnistLinear(nn.Module):
    """Single linear layer for MNIST (from p2pfl) - equivalent to LogisticMNIST."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)
