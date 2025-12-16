import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_models(models: list[nn.Module], test_loader: DataLoader, device: torch.device) -> dict:
    accs = []
    for m in models:
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = m(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accs.append(correct / max(total, 1))

    a = np.array(accs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std())
    }
