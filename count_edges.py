#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from topology.fully import fully_connected
from topology.topo_random import build as build_random
from topology.dclique import build as build_dclique
from utils.hierarchy import build_state_cliques, load_hierarchy_levels, load_scope_instances
from pathlib import Path

def count_edges(A: np.ndarray) -> int:
    """Count undirected edges in adjacency matrix"""
    return int(np.sum(A) // 2)

def main():
    n = 100
    clique_size = 10
    dmax = 4
    seed = 42
    device = torch.device("cpu")
    
    print(f"Counting edges for n={n} nodes\n")
    
    # 1. Fully connected
    A_fully, _ = fully_connected(n, device)
    edges_fully = count_edges(A_fully)
    print(f"Fully Connected: {edges_fully} edges")
    
    # 2. Random
    A_random, _ = build_random(n, dmax, seed, device)
    edges_random = count_edges(A_random)
    avg_degree = np.sum(A_random) / n
    print(f"Random (dmax={dmax}): {edges_random} edges, avg degree={avg_degree:.2f}")
    
    # 3. D-Clique (need dummy data)
    from data.cifar10 import load_cifar10, make_cifar10_loaders
    train, _ = load_cifar10("./data")
    loaders, node_idx = make_cifar10_loaders(train, n, 0.5, 32, seed)
    labels = np.array(train.targets, dtype=np.int64)
    
    cliques, A_dclique, Wc, Wp = build_dclique(
        labels, node_idx, 10, clique_size, 5000, seed, device
    )
    edges_dclique = count_edges(A_dclique)
    print(f"D-Clique (clique_size={clique_size}): {edges_dclique} edges")
    print(f"  - Number of cliques: {len(cliques)}")
    print(f"  - Clique sizes: {[len(c) for c in cliques]}")
    
    # 4. Hierarchy
    hier_cfg_path = Path("config/hierarchy_config.json")
    nodes_map_path = Path("config/nodes_map.json")
    
    if hier_cfg_path.exists() and nodes_map_path.exists():
        levels = load_hierarchy_levels(hier_cfg_path)
        scope_instances = load_scope_instances(nodes_map_path, levels)
        lowest_index = min(cfg.scope_index for cfg in levels)
        states = list(scope_instances[lowest_index].values())
        
        clique_assignments, state_to_cliques = build_state_cliques(
            labels=labels,
            node_indices=node_idx,
            states=states,
            clique_size=clique_size,
            n_swaps=5000,
            seed=seed,
        )
        
        A_hierarchy = np.zeros((n, n), dtype=np.int32)
        for assignment in clique_assignments:
            nodes = assignment["nodes"]
            for u in nodes:
                for v in nodes:
                    if u != v:
                        A_hierarchy[u, v] = 1
        
        edges_hierarchy = count_edges(A_hierarchy)
        print(f"Hierarchy (clique_size={clique_size}): {edges_hierarchy} edges")
        print(f"  - Number of cliques: {len(clique_assignments)}")
        print(f"  - Clique sizes: {[len(a['nodes']) for a in clique_assignments]}")
    else:
        print("Hierarchy: Config files not found")
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"Fully:     {edges_fully:3d} edges (100.0%)")
    print(f"Random:    {edges_random:3d} edges ({edges_random/edges_fully*100:5.1f}%)")
    print(f"D-Clique:  {edges_dclique:3d} edges ({edges_dclique/edges_fully*100:5.1f}%)")
    if hier_cfg_path.exists() and nodes_map_path.exists():
        print(f"Hierarchy: {edges_hierarchy:3d} edges ({edges_hierarchy/edges_fully*100:5.1f}%)")

if __name__ == "__main__":
    main()
