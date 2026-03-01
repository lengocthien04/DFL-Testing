"""
D-Cliques topology builder (synced with p2pfl implementation).
Implements label-aware clique partitioning with greedy swap refinement.
"""
import numpy as np
import torch
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from .topo_random import metropolis_from_adj

LabelDist = Dict[int, float]


# ---------------------------
# Label distributions + skew
# ---------------------------
def _normalize(counts: Dict[int, float]) -> LabelDist:
    """Normalize label counts to probabilities."""
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Label counts must sum to a positive value")
    return {label: value / total for label, value in counts.items()}


def compute_label_distribution(labels: np.ndarray, node_indices: List[np.ndarray]) -> Dict[int, LabelDist]:
    """
    Compute normalized label distribution for each node.
    
    Args:
        labels: Array of all labels
        node_indices: List of arrays, each containing indices for one node's data
        
    Returns:
        Dictionary mapping node_id -> label distribution
    """
    distributions = {}
    for node_id, idx in enumerate(node_indices):
        node_labels = labels[idx]
        counts = Counter(node_labels.tolist())
        distributions[node_id] = _normalize({int(k): float(v) for k, v in counts.items()})
    return distributions


def _aggregate_clique_distribution(clique: List[int], node_distributions: Dict[int, LabelDist]) -> LabelDist:
    """Aggregate label distributions across nodes in a clique."""
    agg: Dict[int, float] = defaultdict(float)
    for node in clique:
        if node not in node_distributions:
            raise ValueError(f"Node {node} missing label distribution")
        for label, prob in node_distributions[node].items():
            agg[label] += prob
    return _normalize(agg)


def compute_skew(clique: List[int], node_distributions: Dict[int, LabelDist], global_distribution: LabelDist) -> float:
    """
    Compute L1 distance between clique label distribution and global distribution.
    Lower is better (more balanced).
    """
    clique_dist = _aggregate_clique_distribution(clique, node_distributions)
    labels = set(clique_dist) | set(global_distribution)
    return sum(abs(clique_dist.get(label, 0.0) - global_distribution.get(label, 0.0)) for label in labels)


# ---------------------------
# Greedy-Swap clique builder
# ---------------------------
def build_cliques_random(n_nodes: int, clique_size: int, seed: int) -> List[List[int]]:
    """Initialize cliques with random partitioning."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_nodes).tolist()
    return [perm[i:i+clique_size] for i in range(0, n_nodes, clique_size)]


def merge_singleton_cliques(
    cliques: List[List[int]],
    node_distributions: Dict[int, LabelDist],
    global_distribution: LabelDist,
) -> List[List[int]]:
    """
    Merge singleton cliques into larger cliques to avoid degenerate cases.
    """
    if len(cliques) <= 1:
        return cliques
    
    idx = 0
    while idx < len(cliques):
        clique = cliques[idx]
        if len(clique) > 1:
            idx += 1
            continue
        
        node = clique[0]
        best_target = None
        best_delta = float('inf')
        
        for target_idx, target_clique in enumerate(cliques):
            if target_idx == idx:
                continue
            current_skew = compute_skew(target_clique, node_distributions, global_distribution)
            candidate = target_clique + [node]
            new_skew = compute_skew(candidate, node_distributions, global_distribution)
            delta = new_skew - current_skew
            if delta < best_delta:
                best_delta = delta
                best_target = target_idx
        
        if best_target is None:
            idx += 1
            continue
        
        cliques[best_target].append(node)
        cliques.pop(idx)
        
        if len(cliques) <= 1:
            break
    
    return cliques


def build_d_cliques(
    labels: np.ndarray,
    node_indices: List[np.ndarray],
    clique_size: int,
    iterations: int = 1000,
    seed: int = 0,
) -> List[List[int]]:
    """
    D-Cliques-style greedy swap algorithm.
    
    Args:
        labels: Array of all labels
        node_indices: List of arrays, each containing indices for one node's data
        clique_size: Target size for each clique
        iterations: Number of greedy swap iterations
        seed: Random seed
        
    Returns:
        List of cliques (each clique is a list of node IDs)
    """
    if clique_size <= 0:
        raise ValueError("clique_size must be positive")
    
    n_nodes = len(node_indices)
    if n_nodes == 0:
        raise ValueError("node_indices cannot be empty")
    
    rng = np.random.RandomState(seed)
    
    # Compute label distributions
    node_distributions = compute_label_distribution(labels, node_indices)
    
    # Global distribution = normalized sum of per-node distributions
    global_counts: Counter = Counter()
    for dist in node_distributions.values():
        for label, prob in dist.items():
            global_counts[label] += prob
    global_distribution = _normalize({int(k): float(v) for k, v in global_counts.items()})
    
    # Initial random partition
    cliques = build_cliques_random(n_nodes, clique_size, seed)
    
    if len(cliques) <= 1:
        return cliques
    
    # Greedy swap refinement
    for _ in range(iterations):
        idx_a, idx_b = rng.choice(len(cliques), size=2, replace=False)
        clique_a = cliques[idx_a]
        clique_b = cliques[idx_b]
        
        if len(clique_a) == 0 or len(clique_b) == 0:
            continue
        
        base_skew = compute_skew(clique_a, node_distributions, global_distribution) + \
                    compute_skew(clique_b, node_distributions, global_distribution)
        
        improvements: List[Tuple[int, int]] = []
        
        # Evaluate all cross-swaps for strictly improving moves
        for a in clique_a:
            for b in clique_b:
                new_a = [x if x != a else b for x in clique_a]
                new_b = [x if x != b else a for x in clique_b]
                
                new_skew = compute_skew(new_a, node_distributions, global_distribution) + \
                          compute_skew(new_b, node_distributions, global_distribution)
                
                if new_skew < base_skew:
                    improvements.append((a, b))
        
        if improvements:
            swap_a, swap_b = improvements[rng.randint(0, len(improvements))]
            # Perform swap
            a_pos = cliques[idx_a].index(swap_a)
            b_pos = cliques[idx_b].index(swap_b)
            cliques[idx_a][a_pos] = swap_b
            cliques[idx_b][b_pos] = swap_a
    
    # Merge singleton cliques
    cliques = merge_singleton_cliques(cliques, node_distributions, global_distribution)
    
    return cliques


# ---------------------------
# Inter-clique edges
# ---------------------------
def build_interclique_edges(
    num_cliques: int,
    mode: str = "small_world",
    small_world_c: int = 2,
) -> List[Tuple[int, int]]:
    """
    Build inter-clique edges (between clique indices).
    
    Args:
        num_cliques: Number of cliques
        mode: Connectivity mode - "ring", "fractal", "small_world", "fully_connected"
        small_world_c: Parameter for small_world mode (number of power-of-two hops)
        
    Returns:
        List of edges as (clique_idx_a, clique_idx_b) tuples
    """
    if num_cliques <= 1:
        return []
    
    if mode not in {"ring", "fractal", "small_world", "fully_connected"}:
        raise ValueError(f"Unknown mode '{mode}'")
    
    edges: Set[Tuple[int, int]] = set()
    
    def add_edge(a: int, b: int) -> None:
        if a != b:
            edges.add((min(a, b), max(a, b)))
    
    # Base ring connectivity (keeps graph connected)
    for i in range(num_cliques):
        add_edge(i, (i + 1) % num_cliques)
    
    if mode == "ring":
        return sorted(edges)
    
    if mode == "fully_connected":
        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                add_edge(i, j)
        return sorted(edges)
    
    if mode == "fractal":
        stride = max(2, num_cliques // 2)
        for i in range(num_cliques):
            add_edge(i, (i + stride) % num_cliques)
        return sorted(edges)
    
    # small_world: power-of-two offsets starting from 2^1=2
    for k in range(1, small_world_c + 1):
        offset = 2**k
        for i in range(num_cliques):
            add_edge(i, (i + offset) % num_cliques)
    
    return sorted(edges)


# ---------------------------
# Assign clique-edges to node-edges
# ---------------------------
def assign_node_edges_balanced(
    cliques: List[List[int]],
    interclique_edges: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Convert clique-index edges into node-id edges by greedy load balancing.
    
    For each (clique_a, clique_b):
        - Pick node in clique_a with lowest current degree
        - Pick node in clique_b with lowest current degree
        - Connect them
    
    Returns:
        Tuple of (node_edges, node_edge_count)
    """
    node_edge_count: Dict[int, int] = {}
    
    # Initialize with intra-clique degree
    for clique in cliques:
        base = len(clique) - 1  # degree in complete clique
        for node in clique:
            node_edge_count[node] = base
    
    node_edges: List[Tuple[int, int]] = []
    
    for ca, cb in interclique_edges:
        # Select lowest degree nodes (deterministic tie-break by node id)
        a = min(cliques[ca], key=lambda n: (node_edge_count[n], n))
        b = min(cliques[cb], key=lambda n: (node_edge_count[n], n))
        
        node_edges.append((a, b))
        node_edge_count[a] += 1
        node_edge_count[b] += 1
    
    return node_edges, node_edge_count


# ---------------------------
# Build adjacency matrix
# ---------------------------
def build_adjacency_matrix(
    n_nodes: int,
    cliques: List[List[int]],
    seed: int = 0,
    inter_mode: str = "small_world",
    small_world_c: int = 2,
    load_balanced: bool = True,
) -> np.ndarray:
    """
    Build adjacency matrix for D-Cliques topology.
    
    Args:
        n_nodes: Total number of nodes
        cliques: List of cliques (each clique is a list of node IDs)
        seed: Random seed for random bridge selection
        inter_mode: Inter-clique connectivity mode
        small_world_c: Parameter for small_world mode
        load_balanced: Whether to use load-balanced bridge selection
        
    Returns:
        Adjacency matrix (n_nodes x n_nodes)
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.int32)
    
    # Intra-clique edges (complete graph within each clique)
    for clique in cliques:
        for u in clique:
            for v in clique:
                if u != v:
                    A[u, v] = 1
    
    # Inter-clique edges
    interclique_edges = build_interclique_edges(
        num_cliques=len(cliques),
        mode=inter_mode,
        small_world_c=small_world_c,
    )
    
    if load_balanced:
        # Load-balanced bridge selection
        node_edges, _ = assign_node_edges_balanced(cliques, interclique_edges)
        for u, v in node_edges:
            A[u, v] = 1
            A[v, u] = 1
    else:
        # Random bridge selection (original paper style)
        rng = np.random.RandomState(seed)
        for ca, cb in interclique_edges:
            a = cliques[ca][rng.randint(0, len(cliques[ca]))]
            b = cliques[cb][rng.randint(0, len(cliques[cb]))]
            A[a, b] = 1
            A[b, a] = 1
    
    return A


# ---------------------------
# Helper matrices
# ---------------------------
def clique_averaging_matrix(n_nodes: int, cliques: List[List[int]], device: torch.device) -> torch.Tensor:
    """Build uniform averaging matrix within cliques."""
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for clique in cliques:
        m = len(clique)
        for i in clique:
            for j in clique:
                W[i, j] = 1.0 / m
    return torch.tensor(W, dtype=torch.float32, device=device)


def build_clique_neighbors(cliques: List[List[int]], A: np.ndarray) -> List[List[int]]:
    """
    Build clique-level adjacency from node-level adjacency A.
    
    Returns:
        List where clique_neighbors[c] = list of clique indices connected to clique c
    """
    C = len(cliques)
    
    # Map node -> clique index
    node_to_clique = {}
    for c_idx, clique in enumerate(cliques):
        for node in clique:
            node_to_clique[node] = c_idx
    
    neighbors = [set() for _ in range(C)]
    
    # Scan adjacency matrix
    n = A.shape[0]
    for u in range(n):
        for v in range(n):
            if A[u, v] == 1:
                cu = node_to_clique[u]
                cv = node_to_clique[v]
                if cu != cv:
                    neighbors[cu].add(cv)
    
    return [sorted(list(nbrs)) for nbrs in neighbors]


# ---------------------------
# Main build function
# ---------------------------
def build(
    labels: np.ndarray,
    node_indices: List[np.ndarray],
    n_classes: int,
    clique_size: int,
    n_swaps: int,
    seed: int,
    device: torch.device,
    inter_mode: str = "small_world",
    load_balanced: bool = True,
):
    """
    Build D-Cliques topology with label-aware clique partitioning.
    
    Args:
        labels: Array of all labels
        node_indices: List of arrays, each containing indices for one node's data
        n_classes: Number of classes (not used, kept for compatibility)
        clique_size: Target size for each clique
        n_swaps: Number of greedy swap iterations
        seed: Random seed
        device: PyTorch device
        inter_mode: Inter-clique connectivity mode
        load_balanced: Whether to use load-balanced bridge selection
        
    Returns:
        Tuple of (cliques, adjacency_matrix, W_clique, W_param)
    """
    # Build label-aware cliques
    cliques = build_d_cliques(labels, node_indices, clique_size, iterations=n_swaps, seed=seed)
    
    # Build adjacency matrix
    A = build_adjacency_matrix(
        n_nodes=len(node_indices),
        cliques=cliques,
        seed=seed + 2,
        inter_mode=inter_mode,
        load_balanced=load_balanced,
    )
    
    # Build weight matrices
    W_param = metropolis_from_adj(A, device)
    W_clique = clique_averaging_matrix(len(node_indices), cliques, device)
    
    return cliques, A, W_clique, W_param
