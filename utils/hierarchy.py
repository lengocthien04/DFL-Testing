from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from topology.dclique import build_d_cliques


@dataclass
class HierarchyLevelConfig:
    scope_index: int
    scope_name: str
    scope_id: str
    enabled: bool
    approach: str
    rounds_per_scope: int
    interval_seconds: int
    interval_epochs: int = 1  # New field for epoch-based intervals
    wait_seconds: int = 0
    max_aggregators: int = 1
    fanout_count: int = 1
    apply_policy: str = "interpolate"
    apply_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.apply_policy = self.apply_policy.lower()
        if self.apply_policy not in {"replace", "interpolate"}:
            raise ValueError(f"Unsupported apply_policy '{self.apply_policy}'")
        self.scope_name = self.scope_name.strip()
        if not self.scope_name:
            raise ValueError("scope_name cannot be empty")


@dataclass
class ScopeInstance:
    level_index: int
    level_name: str
    scope_id: str
    nodes: List[int]
    child_ids: List[str] = field(default_factory=list)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hierarchy_levels(path: str | Path) -> List[HierarchyLevelConfig]:
    """Load hierarchy level configuration."""
    data = _load_json(path)
    entries = data.get("hierarchy_levels", [])
    if not entries:
        raise ValueError("hierarchy_levels is empty")

    levels = [HierarchyLevelConfig(**entry) for entry in entries]
    levels.sort(key=lambda cfg: cfg.scope_index)
    return levels


def _plural_key(name: str) -> str:
    """Best-effort pluralization for config keys."""
    return f"{name}s"


def load_scope_instances(
    path: str | Path,
    level_configs: List[HierarchyLevelConfig],
) -> Dict[int, Dict[str, ScopeInstance]]:
    """
    Parse nodes-map.json into per-level scope instances.
    """
    if not level_configs:
        return {}

    levels_by_index = {cfg.scope_index: cfg for cfg in level_configs}
    min_index = min(levels_by_index)
    max_index = max(levels_by_index)

    data = _load_json(path)
    top_key = levels_by_index[max_index].scope_name
    if top_key not in data:
        raise ValueError(f"nodes map missing top-level key '{top_key}'")

    scopes_by_level: Dict[int, Dict[str, ScopeInstance]] = {idx: {} for idx in levels_by_index}

    def parse_entries(entries: List[Dict[str, Any]], level_idx: int) -> List[ScopeInstance]:
        cfg = levels_by_index[level_idx]
        scope_id_key = f"{cfg.scope_name}_id"
        parsed: List[ScopeInstance] = []
        for entry in entries:
            if scope_id_key not in entry:
                raise ValueError(f"Entry missing key '{scope_id_key}' for level '{cfg.scope_name}'")
            scope_id = entry[scope_id_key]
            if level_idx == min_index:
                nodes = entry.get("nodes", [])
                if not isinstance(nodes, list) or not nodes:
                    raise ValueError(f"Scope '{scope_id}' must list member node ids")
                node_ids = [int(n) for n in nodes]
                child_ids: List[str] = []
            else:
                child_cfg = levels_by_index[level_idx - 1]
                child_key = _plural_key(child_cfg.scope_name)
                child_entries = entry.get(child_key, [])
                if not child_entries:
                    raise ValueError(f"Scope '{scope_id}' missing child list '{child_key}'")
                children = parse_entries(child_entries, level_idx - 1)
                node_ids = sorted({node for child in children for node in child.nodes})
                child_ids = [child.scope_id for child in children]

            scope = ScopeInstance(
                level_index=level_idx,
                level_name=cfg.scope_name,
                scope_id=scope_id,
                nodes=node_ids,
                child_ids=child_ids,
            )
            scopes_by_level[level_idx][scope_id] = scope
            parsed.append(scope)

        return parsed

    parse_entries(data[top_key], max_index)
    return scopes_by_level


def build_state_cliques(
    labels: np.ndarray,
    node_indices: List[np.ndarray],
    states: List[ScopeInstance],
    clique_size: int,
    n_swaps: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Run D-Clique partitioning independently inside each state scope.
    """
    assignments: List[Dict[str, Any]] = []
    state_to_cliques: Dict[str, List[str]] = {state.scope_id: [] for state in states}

    for offset, state in enumerate(states):
        if not state.nodes:
            continue
        subset_indices = [node_indices[idx] for idx in state.nodes]
        local_seed = seed + offset
        local_cliques = build_d_cliques(
            labels=labels,
            node_indices=subset_indices,
            clique_size=clique_size,
            iterations=n_swaps,
            seed=local_seed,
        )
        for c_idx, clique in enumerate(local_cliques):
            mapped = [state.nodes[pos] for pos in clique]
            clique_id = f"{state.scope_id}_clique_{c_idx}"
            assignments.append(
                {
                    "clique_id": clique_id,
                    "nodes": mapped,
                    "state_id": state.scope_id,
                }
            )
            state_to_cliques[state.scope_id].append(clique_id)

    return assignments, state_to_cliques
