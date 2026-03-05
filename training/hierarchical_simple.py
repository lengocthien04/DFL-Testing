"""
Simplified hierarchical training:
1. Local SGD in each node
2. Clique averaging (nodes in same clique share models)
3. State aggregation (aggregate cliques within each state)
4. Nation aggregation (aggregate states)
"""

from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn as nn
from .dsgd import local_sgd_step
from .vector import get_param_matrix, set_param_matrix


def run_steps_hierarchical_simple(
    models: List[nn.Module],
    optims: List[torch.optim.Optimizer],
    loaders: List,
    cliques: List[List[int]],
    states: List[List[int]],
    device: torch.device,
    steps: int,
    current_epoch: int = 0,
    state_interval: int = 1,
    nation_interval: int = 1,
) -> None:
    """
    Execute hierarchical training with clique → state → nation aggregation.
    
    Args:
        models: List of model instances for each node
        optims: List of optimizers for each node
        loaders: List of data loaders for each node
        cliques: List of cliques, where each clique is a list of node IDs
        states: List of states, where each state is a list of node IDs
        device: PyTorch device
        steps: Number of training steps per epoch
        current_epoch: Current epoch number
        state_interval: Perform state aggregation every N epochs
        nation_interval: Perform nation aggregation every N epochs
    """
    iters = [iter(ld) for ld in loaders]
    
    # Training steps
    for _ in range(steps):
        # 1. Local SGD in each node
        for idx, model in enumerate(models):
            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(loaders[idx])
                batch = next(iters[idx])
            local_sgd_step(model, optims[idx], batch, device)
        
        # 2. Clique averaging (after every step)
        with torch.no_grad():
            X = get_param_matrix(models).to(device)
            
            for clique in cliques:
                if len(clique) == 0:
                    continue
                idx = torch.tensor(clique, device=device, dtype=torch.long)
                mean_vec = X.index_select(0, idx).mean(dim=0)
                X[idx] = mean_vec
            
            set_param_matrix(models, X)
    
    # 3. State aggregation (at end of epoch if due)
    if current_epoch > 0 and state_interval > 0 and current_epoch % state_interval == 0:
        with torch.no_grad():
            X = get_param_matrix(models).to(device)
            
            # Build clique-to-state mapping
            clique_to_state = {}
            for state_idx, state_nodes in enumerate(states):
                state_node_set = set(state_nodes)
                for clique_idx, clique in enumerate(cliques):
                    if len(clique) > 0 and clique[0] in state_node_set:
                        clique_to_state[clique_idx] = state_idx
            
            # Aggregate within each state
            for state_idx, state_nodes in enumerate(states):
                if len(state_nodes) == 0:
                    continue
                
                # Get all cliques in this state
                state_cliques = [cliques[i] for i, s in clique_to_state.items() if s == state_idx]
                
                if len(state_cliques) == 0:
                    continue
                
                # Aggregate: take one representative from each clique
                representatives = []
                for clique in state_cliques:
                    if len(clique) > 0:
                        representatives.append(clique[0])
                
                if len(representatives) == 0:
                    continue
                
                # Average the representatives
                rep_idx = torch.tensor(representatives, device=device, dtype=torch.long)
                state_avg = X.index_select(0, rep_idx).mean(dim=0)
                
                # Broadcast to all nodes in the state
                state_idx_tensor = torch.tensor(state_nodes, device=device, dtype=torch.long)
                X[state_idx_tensor] = state_avg.unsqueeze(0).expand(len(state_nodes), -1)
            
            set_param_matrix(models, X)
    
    # 4. Nation aggregation (at end of epoch if due)
    if current_epoch > 0 and nation_interval > 0 and current_epoch % nation_interval == 0:
        with torch.no_grad():
            X = get_param_matrix(models).to(device)
            
            # Take one representative from each state
            state_representatives = []
            for state_nodes in states:
                if len(state_nodes) > 0:
                    state_representatives.append(state_nodes[0])
            
            if len(state_representatives) > 0:
                # Average across states
                rep_idx = torch.tensor(state_representatives, device=device, dtype=torch.long)
                nation_avg = X.index_select(0, rep_idx).mean(dim=0)
                
                # Broadcast to all nodes
                n_nodes = len(models)
                X[:] = nation_avg.unsqueeze(0).expand(n_nodes, -1)
                
                set_param_matrix(models, X)
