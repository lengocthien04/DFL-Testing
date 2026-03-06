"""
Hierarchical training with bridge-based aggregation:
1. Local SGD in each node
2. Clique averaging (nodes in same clique share models)
3. State aggregation (only aggregate cliques connected by bridges within state)
4. Nation aggregation (aggregate states)
"""

from __future__ import annotations
from typing import List, Dict, Set, Any
import torch
import torch.nn as nn
import numpy as np
from .dsgd import local_sgd_step
from .vector import get_param_matrix, set_param_matrix


def build_hierarchy_runtime(
    level_configs,
    scope_instances,
    clique_assignments: List[Dict[str, Any]],
    state_to_cliques: Dict[str, List[str]],
):
    """
    Simplified: Just return the data structures needed for bridge-based aggregation.
    """
    return {
        'clique_assignments': clique_assignments,
        'state_to_cliques': state_to_cliques,
        'scope_instances': scope_instances,
    }


def find_bridge_connections(cliques: List[List[int]], adjacency: np.ndarray) -> Dict[int, Set[int]]:
    """
    Find which cliques are connected via bridges (inter-clique edges).
    
    Args:
        cliques: List of cliques (each is a list of node IDs)
        adjacency: Adjacency matrix (n_nodes x n_nodes)
        
    Returns:
        Dict mapping clique_idx to set of connected clique indices
    """
    connections = {i: set() for i in range(len(cliques))}
    
    for i, clique_a in enumerate(cliques):
        for j, clique_b in enumerate(cliques):
            if i >= j:
                continue
            
            # Check if there's any edge between clique_a and clique_b
            has_bridge = False
            for node_a in clique_a:
                for node_b in clique_b:
                    if adjacency[node_a, node_b] > 0 or adjacency[node_b, node_a] > 0:
                        has_bridge = True
                        break
                if has_bridge:
                    break
            
            if has_bridge:
                connections[i].add(j)
                connections[j].add(i)
    
    return connections


def run_steps_hierarchical_mydclique(
    models: List[nn.Module],
    optims: List[torch.optim.Optimizer],
    loaders: List,
    hierarchy_state: Dict,
    device: torch.device,
    steps: int,
    current_epoch: int = 0,
    adjacency: np.ndarray = None,
    state_interval: int = 1,
    nation_interval: int = 1,
) -> None:
    """
    Execute hierarchical training with bridge-based state aggregation.
    
    Args:
        models: List of model instances for each node
        optims: List of optimizers for each node
        loaders: List of data loaders for each node
        hierarchy_state: Dict containing clique_assignments, state_to_cliques, scope_instances
        device: PyTorch device
        steps: Number of training steps per epoch
        current_epoch: Current epoch number
        adjacency: Adjacency matrix to determine bridge connections
        state_interval: Perform state aggregation every N epochs
        nation_interval: Perform nation aggregation every N epochs
    """
    iters = [iter(ld) for ld in loaders]
    
    # Training steps (local SGD only, no averaging during steps)
    for _ in range(steps):
        for idx, model in enumerate(models):
            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(loaders[idx])
                batch = next(iters[idx])
            local_sgd_step(model, optims[idx], batch, device)
    
    # Extract hierarchy info
    clique_assignments = hierarchy_state['clique_assignments']
    state_to_cliques = hierarchy_state['state_to_cliques']
    scope_instances = hierarchy_state['scope_instances']
    
    # Build clique list (list of node lists)
    cliques_list = [assignment['nodes'] for assignment in clique_assignments]
    
    # Build state list
    states_list = [scope.nodes for scope in scope_instances[1].values()]
    
    # All aggregations happen at END of epoch
    with torch.no_grad():
        X = get_param_matrix(models).to(device)
        
        # 1. Clique averaging (always)
        for clique in cliques_list:
            if len(clique) == 0:
                continue
            idx = torch.tensor(clique, device=device, dtype=torch.long)
            mean_vec = X.index_select(0, idx).mean(dim=0)
            X[idx] = mean_vec
        
        # 2. State aggregation with bridge-based connectivity (if due)
        if current_epoch > 0 and state_interval > 0 and current_epoch % state_interval == 0:
            if adjacency is not None:
                # Find bridge connections between cliques
                bridge_connections = find_bridge_connections(cliques_list, adjacency)
                
                # Build clique-to-state mapping
                clique_to_state = {}
                for state_idx, state_id in enumerate(scope_instances[1].keys()):
                    state_clique_ids = state_to_cliques[state_id]
                    for clique_idx, assignment in enumerate(clique_assignments):
                        if assignment['clique_id'] in state_clique_ids:
                            clique_to_state[clique_idx] = state_idx
                
                # For each state, aggregate only connected cliques
                for state_idx, state_nodes in enumerate(states_list):
                    if len(state_nodes) == 0:
                        continue
                    
                    # Get cliques in this state
                    state_clique_indices = [i for i, s in clique_to_state.items() if s == state_idx]
                    
                    if len(state_clique_indices) == 0:
                        continue
                    
                    # Find connected components within this state using bridge connections
                    visited = set()
                    components = []
                    
                    for start_idx in state_clique_indices:
                        if start_idx in visited:
                            continue
                        
                        # BFS to find all cliques connected to start_idx
                        component = []
                        queue = [start_idx]
                        visited.add(start_idx)
                        
                        while queue:
                            curr_idx = queue.pop(0)
                            component.append(curr_idx)
                            
                            # Add connected cliques that are in the same state
                            for neighbor_idx in bridge_connections[curr_idx]:
                                if neighbor_idx in state_clique_indices and neighbor_idx not in visited:
                                    visited.add(neighbor_idx)
                                    queue.append(neighbor_idx)
                        
                        components.append(component)
                    
                    # Aggregate each connected component separately
                    for component in components:
                        if len(component) == 0:
                            continue
                        
                        # Get all nodes in this connected component
                        component_nodes = []
                        for clique_idx in component:
                            component_nodes.extend(cliques_list[clique_idx])
                        
                        if len(component_nodes) == 0:
                            continue
                        
                        # Average models in this component
                        comp_idx = torch.tensor(component_nodes, device=device, dtype=torch.long)
                        comp_avg = X.index_select(0, comp_idx).mean(dim=0)
                        
                        # Broadcast to all nodes in the component
                        X[comp_idx] = comp_avg.unsqueeze(0).expand(len(component_nodes), -1)
        
        # 3. Nation aggregation (if due)
        if current_epoch > 0 and nation_interval > 0 and current_epoch % nation_interval == 0:
            # Take one representative from each state
            state_representatives = []
            for state_nodes in states_list:
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
