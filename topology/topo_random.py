import numpy as np
import torch

def random_dmax_graph(n: int, d_max: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = 1

    deg = A.sum(axis=1)
    tries = 0
    while np.any(deg < d_max) and tries < n * n * 20:
        i, j = rng.randint(0, n, 2)
        if i == j or A[i, j] == 1:
            tries += 1
            continue
        if deg[i] >= d_max or deg[j] >= d_max:
            tries += 1
            continue
        A[i, j] = A[j, i] = 1
        deg[i] += 1
        deg[j] += 1
        tries += 1
    return A

def metropolis_from_adj(A: np.ndarray, device: torch.device) -> torch.Tensor:
    n = A.shape[0]
    deg = A.sum(axis=1)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return torch.tensor(W, dtype=torch.float32, device=device)

def build(n: int, dmax: int, seed: int, device: torch.device):
    A = random_dmax_graph(n, dmax, seed)
    W = metropolis_from_adj(A, device)
    return A, W
