import numpy as np

def adjacency_from_W(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = W.shape[0]
    A = np.zeros((n, n), dtype=np.int32)
    M = (W > eps).astype(np.int32)
    M = np.maximum(M, M.T)
    np.fill_diagonal(M, 0)
    A[M == 1] = 1
    return A

def communication_stats_from_adj(A: np.ndarray) -> dict:
    deg = A.sum(axis=1).astype(np.float64)
    return {
        "avg_degree": float(deg.mean()),
        "msgs_per_node_per_step": float(deg.mean()),
        "total_msgs_per_step": float(deg.sum())
    }
