"""
DSGD with random node dropout.
Each epoch, randomly select nodes to drop offline.
"""
import torch
import torch.nn as nn
import numpy as np

from .dsgd import local_sgd_step
from .vector import get_param_matrix, set_param_matrix


def run_steps_dsgd_dropout(
    models, 
    optims, 
    loaders, 
    W: torch.Tensor, 
    device: torch.device, 
    steps: int,
    dropout_count: int = 3,
    seed: int = None
):
    """
    Run DSGD with random node dropout each epoch.
    
    Args:
        models: List of models for each node
        optims: List of optimizers
        loaders: List of data loaders
        W: Mixing matrix (adjacency-based)
        device: PyTorch device
        steps: Number of training steps per epoch
        dropout_count: Number of nodes to drop each epoch
        seed: Random seed for reproducibility (optional)
    """
    iters = [iter(ld) for ld in loaders]
    n = len(models)
    
    # Randomly select nodes to drop this epoch
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    dropped_nodes = set(rng.choice(n, size=dropout_count, replace=False))
    active_nodes = [i for i in range(n) if i not in dropped_nodes]
    
    print(f"  Dropped nodes: {sorted(dropped_nodes)}, Active: {len(active_nodes)}/{n}")
    
    # Training steps (only active nodes train)
    for _ in range(steps):
        for i in active_nodes:
            try:
                batch = next(iters[i])
            except StopIteration:
                iters[i] = iter(loaders[i])
                batch = next(iters[i])
            local_sgd_step(models[i], optims[i], batch, device)
    
    # Gossip averaging at end of epoch
    # Only active nodes participate in averaging
    with torch.no_grad():
        X = get_param_matrix(models).to(device)
        
        # Create modified mixing matrix: zero out dropped nodes
        W_active = W.clone()
        for dropped in dropped_nodes:
            W_active[dropped, :] = 0.0
            W_active[:, dropped] = 0.0
            W_active[dropped, dropped] = 1.0  # Dropped nodes keep their own model
        
        # Renormalize rows for active nodes
        for i in active_nodes:
            row_sum = W_active[i, :].sum()
            if row_sum > 0:
                W_active[i, :] = W_active[i, :] / row_sum
        
        X = W_active @ X
        set_param_matrix(models, X)
