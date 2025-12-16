import numpy as np
import torch

def fully_connected(n: int, device: torch.device):
    W = torch.ones((n, n), dtype=torch.float32, device=device) / float(n)
    A = np.ones((n, n), dtype=np.int32)
    np.fill_diagonal(A, 0)
    return A, W
