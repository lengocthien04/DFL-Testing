import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.communication import adjacency_from_W


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def class_proportions(
    labels: np.ndarray,
    node_indices: list[np.ndarray],
    n_classes: int,
) -> np.ndarray:
    """
    Pi[i, k] = proportion of class k on node i
    """
    n = len(node_indices)
    Pi = np.zeros((n, n_classes), dtype=np.float64)

    for i, idx in enumerate(node_indices):
        counts = np.bincount(labels[idx], minlength=n_classes).astype(np.float64)
        s = counts.sum()
        if s > 0:
            Pi[i] = counts / s

    return Pi


# ------------------------------------------------------------
# STL-FW objective gradient
# ------------------------------------------------------------

def grad_g(W: np.ndarray, Pi: np.ndarray, lam: float) -> np.ndarray:
    """
    Gradient of the refined (STL-FW) objective
    """
    n = Pi.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    J = (ones @ ones.T) / n

    diff = (W @ Pi) - (J @ Pi)
    term1 = (2.0 / n) * (diff @ Pi.T)
    term2 = (2.0 * lam / n) * (W - J)

    return term1 + term2


# ------------------------------------------------------------
# Frank–Wolfe atom (permutation matrix)
# ------------------------------------------------------------

def fw_atom_from_gradient(G: np.ndarray) -> np.ndarray:
    """
    Linear minimization oracle over permutation matrices
    """
    r, c = linear_sum_assignment(G)
    n = G.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    P[r, c] = 1.0
    return P


# ------------------------------------------------------------
# Line search
# ------------------------------------------------------------

def line_search_gamma(
    W: np.ndarray,
    P: np.ndarray,
    Pi: np.ndarray,
    lam: float,
) -> float:
    """
    Closed-form line search for quadratic STL-FW objective
    """
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

    denom = (
        np.linalg.norm(D @ Pi, ord="fro") ** 2
        + lam * np.linalg.norm(D, ord="fro") ** 2
    )

    if denom <= 1e-12:
        return 0.0

    return float(np.clip(numer / denom, 0.0, 1.0))


# ------------------------------------------------------------
# Core STL-FW solver
# ------------------------------------------------------------

def stl_fw(
    Pi: np.ndarray,
    lam: float,
    iters: int,
) -> np.ndarray:
    """
    Frank–Wolfe optimization over the Birkhoff polytope
    """
    n = Pi.shape[0]
    W = np.eye(n, dtype=np.float64)

    for _ in range(iters):
        G = grad_g(W, Pi, lam)
        P = fw_atom_from_gradient(G)
        gamma = line_search_gamma(W, P, Pi, lam)
        W = (1.0 - gamma) * W + gamma * P

    return W


# ------------------------------------------------------------
# Degree-constrained projection
# ------------------------------------------------------------

def sparsify_topk(
    W: np.ndarray,
    dmax: int,
) -> np.ndarray:
    """
    Enforce degree constraint by keeping top-dmax neighbors
    per node (excluding self), then symmetrize and normalize.
    """
    n = W.shape[0]
    W_new = np.zeros_like(W)

    for i in range(n):
        # sort descending, skip self-loop
        order = np.argsort(W[i])[::-1]
        neighbors = [j for j in order if j != i][:dmax]
        for j in neighbors:
            W_new[i, j] = W[i, j]

    # enforce symmetry
    W_new = 0.5 * (W_new + W_new.T)

    # non-negativity
    W_new = np.maximum(W_new, 0.0)

    # row normalization
    row_sum = W_new.sum(axis=1, keepdims=True)
    W_new = W_new / np.maximum(row_sum, 1e-12)

    return W_new


# ------------------------------------------------------------
# Public builder
# ------------------------------------------------------------

def build(
    labels: np.ndarray,
    node_indices: list[np.ndarray],
    n_classes: int,
    lam: float,
    iters: int,
    device: torch.device,
    dmax: int,
):
    """
    Build refined topology with explicit degree bound.
    """
    Pi = class_proportions(labels, node_indices, n_classes)

    # 1) Dense refined FW optimization
    W_np = stl_fw(Pi, lam=lam, iters=iters)

    # 2) Degree constraint (THIS IS THE FIX)
    W_np = sparsify_topk(W_np, dmax)

    # 3) Adjacency for communication stats
    A = adjacency_from_W(W_np, eps=1e-12)

    # 4) Torch tensor
    W = torch.tensor(W_np, dtype=torch.float32, device=device)

    return A, W
