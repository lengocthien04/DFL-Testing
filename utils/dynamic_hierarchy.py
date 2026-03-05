"""
Dynamic hierarchy configuration generator.
Automatically divides nodes into 2 states based on --n argument.
"""

from typing import List, Dict
from utils.hierarchy import ScopeInstance, HierarchyLevelConfig


def generate_two_state_hierarchy(n_nodes: int) -> Dict[int, Dict[str, ScopeInstance]]:
    """
    Generate a 2-state hierarchy by dividing n_nodes in half.
    
    Args:
        n_nodes: Total number of nodes
        
    Returns:
        scope_instances: Dict mapping scope_index to scope instances
    """
    if n_nodes < 2:
        raise ValueError("Need at least 2 nodes for 2-state hierarchy")
    
    # Divide nodes into 2 states
    half = n_nodes // 2
    
    state_alpha_nodes = list(range(0, half))
    state_beta_nodes = list(range(half, n_nodes))
    
    # Create state instances (scope_index=1 for state level)
    state_alpha = ScopeInstance(
        scope_id="state_alpha",
        nodes=state_alpha_nodes,
        child_ids=[],
    )
    
    state_beta = ScopeInstance(
        scope_id="state_beta",
        nodes=state_beta_nodes,
        child_ids=[],
    )
    
    # Create nation instance (scope_index=2 for nation level)
    nation = ScopeInstance(
        scope_id="nation_0",
        nodes=list(range(n_nodes)),
        child_ids=["state_alpha", "state_beta"],
    )
    
    # Return scope instances organized by level
    scope_instances = {
        1: {  # State level
            "state_alpha": state_alpha,
            "state_beta": state_beta,
        },
        2: {  # Nation level
            "nation_0": nation,
        }
    }
    
    return scope_instances
