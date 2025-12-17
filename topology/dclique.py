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
    # Paper (Bellet et al., ICASSP'21) uses L1 skew:
    #   skew(C)=sum_l |p_C(l) - p(l)|
    # Keep the same objective to match Greedy Swap in Algorithm 2.
    return float(np.abs(mean - global_pi).sum())

def build_cliques_random(n_nodes: int, clique_size: int, seed: int) -> list[list[int]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_nodes).tolist()
    return [perm[i:i+clique_size] for i in range(0, n_nodes, clique_size)]

def greedy_swap_refine(cliques: list[list[int]], Pi: np.ndarray, n_swaps: int, seed: int) -> list[list[int]]:
    rng = np.random.RandomState(seed)
    global_pi = Pi.mean(axis=0)
    C = len(cliques)

    # Faithful Greedy Swap (Alg. 2): pick two cliques, enumerate all (i,j) pairs,
    # collect improving swaps, then apply a random improving swap.
    for _ in range(n_swaps):
        c1, c2 = rng.randint(0, C, 2)
        if c1 == c2 or len(cliques[c1]) == 0 or len(cliques[c2]) == 0:
            continue

        base = clique_cost(cliques[c1], Pi, global_pi) + clique_cost(cliques[c2], Pi, global_pi)
        improving: list[tuple[int, int]] = []

        # Evaluate all possible cross-clique swaps.
        for i in cliques[c1]:
            for j in cliques[c2]:
                if i == j:
                    continue
                # Compute cost after swap using temporary lists.
                c1_new = [x if x != i else j for x in cliques[c1]]
                c2_new = [x if x != j else i for x in cliques[c2]]
                new_cost = clique_cost(c1_new, Pi, global_pi) + clique_cost(c2_new, Pi, global_pi)
                if new_cost < base:
                    improving.append((i, j))

        if len(improving) == 0:
            continue

        i, j = improving[rng.randint(0, len(improving))]
        i_pos = cliques[c1].index(i)
        j_pos = cliques[c2].index(j)
        cliques[c1][i_pos], cliques[c2][j_pos] = j, i

    return cliques

def build_adjacency(n_nodes: int, cliques: list[list[int]], seed: int = 0) -> np.ndarray:
    A = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    # 1) Intra-clique fully connected
    for clique in cliques:
        for u in clique:
            for v in clique:
                if u != v:
                    A[u, v] = 1

    # 2) Inter-clique ring + small-world (Algorithm 4, D-Cliques)
    rng = np.random.RandomState(seed)
    C = len(cliques)

    # power-of-two offsets: 1, 2, 4, 8, ...
    offsets = []
    k = 1
    while k < C:
        offsets.append(k)
        k *= 2

    for c in range(C):
        for off in offsets:
            d = (c + off) % C

            a_clique = cliques[c]
            b_clique = cliques[d]

            # pick random representatives (paper style)
            a = a_clique[rng.randint(0, len(a_clique))]
            b = b_clique[rng.randint(0, len(b_clique))]

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

    A = build_adjacency(len(node_indices), cliques, seed=seed+2)
    W_param = metropolis_from_adj(A, device)
    W_clique = clique_averaging_matrix(len(node_indices), cliques, device)
    return cliques, A, W_clique, W_param


def build_clique_neighbors(cliques: list[list[int]], A: np.ndarray) -> list[list[int]]:
    """
    Build clique-level adjacency from node-level adjacency A.
    clique_neighbors[c] = list of clique indices connected to clique c
    """
    C = len(cliques)

    # map node -> clique index
    node_to_clique = {}
    for c_idx, clique in enumerate(cliques):
        for node in clique:
            node_to_clique[node] = c_idx

    neighbors = [set() for _ in range(C)]

    # scan adjacency matrix
    n = A.shape[0]
    for u in range(n):
        for v in range(n):
            if A[u, v] == 1:
                cu = node_to_clique[u]
                cv = node_to_clique[v]
                if cu != cv:
                    neighbors[cu].add(cv)

    return [sorted(list(nbrs)) for nbrs in neighbors]
