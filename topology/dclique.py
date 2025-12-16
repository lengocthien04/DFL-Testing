import numpy as np
import torch

from .topo_random import metropolis_from_adj

def compute_Pi(labels: np.ndarray, node_indices: list[np.ndarray], n_classes: int) -> np.ndarray:
    n = len(node_indices)
    Pi = np.zeros((n, n_classes), dtype=np.float64)
    for i, idx in enumerate(node_indices):
        counts = np.bincount(labels[idx], minlength=n_classes).astype(np.float64)
        Pi[i] = counts / max(counts.sum(), 1.0)
    return Pi

def clique_cost(clique: list[int], Pi: np.ndarray, global_pi: np.ndarray) -> float:
    if len(clique) == 0:
        return 0.0
    mean = Pi[clique].mean(axis=0)
    return float(np.linalg.norm(mean - global_pi, ord=2))

def build_cliques_random(n_nodes: int, clique_size: int, seed: int) -> list[list[int]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_nodes).tolist()
    return [perm[i:i+clique_size] for i in range(0, n_nodes, clique_size)]

def greedy_swap_refine(cliques: list[list[int]], Pi: np.ndarray, n_swaps: int, seed: int) -> list[list[int]]:
    rng = np.random.RandomState(seed)
    global_pi = Pi.mean(axis=0)
    C = len(cliques)

    for _ in range(n_swaps):
        c1, c2 = rng.randint(0, C, 2)
        if c1 == c2 or len(cliques[c1]) == 0 or len(cliques[c2]) == 0:
            continue
        i = rng.choice(cliques[c1])
        j = rng.choice(cliques[c2])

        before = clique_cost(cliques[c1], Pi, global_pi) + clique_cost(cliques[c2], Pi, global_pi)

        i_pos = cliques[c1].index(i)
        j_pos = cliques[c2].index(j)
        cliques[c1][i_pos], cliques[c2][j_pos] = j, i

        after = clique_cost(cliques[c1], Pi, global_pi) + clique_cost(cliques[c2], Pi, global_pi)

        if after > before:
            cliques[c1][i_pos], cliques[c2][j_pos] = i, j

    return cliques

def build_adjacency(n_nodes: int, cliques: list[list[int]]) -> np.ndarray:
    A = np.zeros((n_nodes, n_nodes), dtype=np.int32)
    for clique in cliques:
        for u in clique:
            for v in clique:
                if u != v:
                    A[u, v] = 1
    reps = [c[0] for c in cliques]
    C = len(reps)
    for c in range(C):
        a = reps[c]
        b = reps[(c + 1) % C]
        A[a, b] = 1
        A[b, a] = 1
    return A

def clique_averaging_matrix(n_nodes: int, cliques: list[list[int]], device: torch.device) -> torch.Tensor:
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for clique in cliques:
        m = len(clique)
        for i in clique:
            for j in clique:
                W[i, j] = 1.0 / m
    return torch.tensor(W, dtype=torch.float32, device=device)

def build(labels: np.ndarray, node_indices: list[np.ndarray], n_classes: int, clique_size: int,
          n_swaps: int, seed: int, device: torch.device):
    Pi = compute_Pi(labels, node_indices, n_classes)
    cliques = build_cliques_random(len(node_indices), clique_size, seed)
    cliques = greedy_swap_refine(cliques, Pi, n_swaps=n_swaps, seed=seed+1)

    A = build_adjacency(len(node_indices), cliques)
    W_param = metropolis_from_adj(A, device)
    W_clique = clique_averaging_matrix(len(node_indices), cliques, device)
    return cliques, A, W_clique, W_param
