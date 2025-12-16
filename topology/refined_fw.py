import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.communication import adjacency_from_W

def class_proportions(labels: np.ndarray, node_indices: list[np.ndarray], n_classes: int) -> np.ndarray:
    n = len(node_indices)
    Pi = np.zeros((n, n_classes), dtype=np.float64)
    for i, idx in enumerate(node_indices):
        counts = np.bincount(labels[idx], minlength=n_classes).astype(np.float64)
        Pi[i] = counts / max(counts.sum(), 1.0)
    return Pi

def grad_g(W: np.ndarray, Pi: np.ndarray, lam: float) -> np.ndarray:
    n, _ = Pi.shape
    ones = np.ones((n, 1), dtype=np.float64)
    J = (ones @ ones.T) / n
    diff = (W @ Pi) - (J @ Pi)
    term1 = (2.0 / n) * (diff @ Pi.T)
    term2 = (2.0 * lam / n) * (W - J)
    return term1 + term2

def fw_atom_from_gradient(grad: np.ndarray) -> np.ndarray:
    r, c = linear_sum_assignment(grad)
    n = grad.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    P[r, c] = 1.0
    return P

def line_search_gamma(W: np.ndarray, P: np.ndarray, Pi: np.ndarray, lam: float) -> float:
    n, K = Pi.shape
    ones = np.ones((n, 1), dtype=np.float64)
    J = (ones @ ones.T) / n

    D = P - W
    WPi = W @ Pi
    JPi = J @ Pi

    numer = 0.0
    for k in range(K):
        a = (JPi[:, k] - WPi[:, k])
        b = D @ Pi[:, k]
        numer += float(a.T @ b)
    numer -= lam * float(np.trace((W - J).T @ D))

    denom = float(np.linalg.norm(D @ Pi, ord="fro") ** 2 + lam * (np.linalg.norm(D, ord="fro") ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(numer / denom, 0.0, 1.0))

def stl_fw(Pi: np.ndarray, lam: float, iters: int) -> np.ndarray:
    n = Pi.shape[0]
    W = np.eye(n, dtype=np.float64)
    for _ in range(iters):
        G = grad_g(W, Pi, lam)
        P = fw_atom_from_gradient(G)
        gamma = line_search_gamma(W, P, Pi, lam)
        W = (1.0 - gamma) * W + gamma * P
    return W

def build(labels: np.ndarray, node_indices: list[np.ndarray], n_classes: int, lam: float, iters: int, device: torch.device):
    Pi = class_proportions(labels, node_indices, n_classes)
    W_np = stl_fw(Pi, lam=lam, iters=iters)
    A = adjacency_from_W(W_np, eps=1e-12)
    W = torch.tensor(W_np, dtype=torch.float32, device=device)
    return A, W
