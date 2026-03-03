from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .dsgd import local_sgd_step
from .vector import get_param_matrix, set_param_matrix
from utils.hierarchy import HierarchyLevelConfig, ScopeInstance


@dataclass
class CliqueRuntime:
    clique_id: str
    nodes: List[int]
    parent_state_id: str
    fanout_cursor: int = 0

    def select_fanout(self) -> int:
        node = self.nodes[self.fanout_cursor % len(self.nodes)]
        self.fanout_cursor += 1
        return node


@dataclass
class ScopeRuntime:
    instance: ScopeInstance
    config: HierarchyLevelConfig
    aggregator_cursor: int = 0
    fanout_cursor: int = 0
    latest_model: Optional[torch.Tensor] = None
    child_cliques: List[str] = field(default_factory=list)

    def select_aggregator(self) -> int:
        node = self.instance.nodes[self.aggregator_cursor % len(self.instance.nodes)]
        self.aggregator_cursor += 1
        return node

    def select_fanout_node(self) -> int:
        node = self.instance.nodes[self.fanout_cursor % len(self.instance.nodes)]
        self.fanout_cursor += 1
        return node


@dataclass
class LevelRuntime:
    config: HierarchyLevelConfig
    scope_index: int
    scopes: Dict[str, ScopeRuntime]
    child_level_index: Optional[int]
    next_round_time: Optional[float] = None
    initialized: bool = False


@dataclass
class HierarchyRuntimeState:
    cliques: Dict[str, CliqueRuntime]
    level_runtimes: List[LevelRuntime]
    scopes_by_level: Dict[int, Dict[str, ScopeRuntime]]
    lowest_level_index: int


def build_hierarchy_runtime(
    level_configs: List[HierarchyLevelConfig],
    scope_instances: Dict[int, Dict[str, ScopeInstance]],
    clique_assignments: List[Dict[str, Any]],
    state_to_cliques: Dict[str, List[str]],
) -> HierarchyRuntimeState:
    if not level_configs:
        raise ValueError("Hierarchy requires at least one high-level scope configuration")

    levels_sorted = sorted(level_configs, key=lambda cfg: cfg.scope_index)
    lowest_index = levels_sorted[0].scope_index

    # Build clique runtimes
    cliques: Dict[str, CliqueRuntime] = {}
    for assignment in clique_assignments:
        clique_id = assignment["clique_id"]
        nodes = assignment["nodes"]
        state_id = assignment["state_id"]
        if not nodes:
            continue
        cliques[clique_id] = CliqueRuntime(
            clique_id=clique_id,
            nodes=nodes,
            parent_state_id=state_id,
        )

    scopes_by_level: Dict[int, Dict[str, ScopeRuntime]] = {}
    for cfg in levels_sorted:
        instances = scope_instances.get(cfg.scope_index)
        if instances is None:
            raise ValueError(f"No nodes-map entries for level '{cfg.scope_name}'")
        scope_runtime_map: Dict[str, ScopeRuntime] = {}
        for scope_id, instance in instances.items():
            runtime = ScopeRuntime(instance=instance, config=cfg)
            if cfg.scope_index == lowest_index:
                runtime.child_cliques = state_to_cliques.get(scope_id, [])
            scope_runtime_map[scope_id] = runtime
        scopes_by_level[cfg.scope_index] = scope_runtime_map

    child_indices: Dict[int, Optional[int]] = {}
    for idx, cfg in enumerate(levels_sorted):
        child_indices[cfg.scope_index] = (
            levels_sorted[idx - 1].scope_index if idx > 0 else None
        )

    level_runtimes = [
        LevelRuntime(
            config=cfg,
            scope_index=cfg.scope_index,
            scopes=scopes_by_level[cfg.scope_index],
            child_level_index=child_indices[cfg.scope_index],
        )
        for cfg in levels_sorted
    ]

    return HierarchyRuntimeState(
        cliques=cliques,
        level_runtimes=level_runtimes,
        scopes_by_level=scopes_by_level,
        lowest_level_index=lowest_index,
    )


def run_steps_hierarchical_mydclique(
    models,
    optims,
    loaders,
    hierarchy_state: HierarchyRuntimeState,
    device,
    steps: int,
) -> None:
    """
    Execute local clique rounds interleaved with scheduled high-level aggregations.
    """
    iters = [iter(ld) for ld in loaders]
    start_time = time.monotonic()
    for level in hierarchy_state.level_runtimes:
        cfg = level.config
        if not cfg.enabled or cfg.interval_seconds <= 0:
            level.next_round_time = None
            level.initialized = True
            continue
        if not level.initialized:
            level.next_round_time = start_time + cfg.interval_seconds
            level.initialized = True

    with torch.no_grad():
        X_bootstrap = get_param_matrix(models).to(device)
        X_bootstrap, changed = _process_high_levels(
            X=X_bootstrap,
            hierarchy_state=hierarchy_state,
            device=device,
        )
        if changed:
            set_param_matrix(models, X_bootstrap)

    for _ in range(steps):
        # Local SGD inside each node
        for idx, model in enumerate(models):
            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(loaders[idx])
                batch = next(iters[idx])
            local_sgd_step(model, optims[idx], batch, device)

        with torch.no_grad():
            X = get_param_matrix(models).to(device)

            # Intra-clique averaging
            for clique in hierarchy_state.cliques.values():
                idx = torch.tensor(clique.nodes, device=device, dtype=torch.long)
                mean_vec = X.index_select(0, idx).mean(dim=0)
                X[idx] = mean_vec

            set_param_matrix(models, X)

            X, updated = _process_high_levels(
                X=X,
                hierarchy_state=hierarchy_state,
                device=device,
            )
            if updated:
                set_param_matrix(models, X)


def _process_high_levels(
    X: torch.Tensor,
    hierarchy_state: HierarchyRuntimeState,
    device,
) -> Tuple[torch.Tensor, bool]:
    """
    Run all due high-level rounds before the next clique round begins.
    """
    updated = False
    while True:
        now = time.monotonic()
        due_levels = [
            level
            for level in hierarchy_state.level_runtimes
            if level.next_round_time is not None and now >= level.next_round_time
        ]
        if not due_levels:
            break
        due_levels.sort(key=lambda lvl: lvl.scope_index)
        for level in due_levels:
            X = _run_level_round(level, hierarchy_state, X, device)
            updated = True
            _schedule_next_round(level, now)

    return X, updated


def _run_level_round(
    level: LevelRuntime,
    hierarchy_state: HierarchyRuntimeState,
    X: torch.Tensor,
    device,
) -> torch.Tensor:
    cfg = level.config
    child_level_idx = level.child_level_index

    for scope in level.scopes.values():
        if not scope.instance.nodes:
            continue

        scope.select_aggregator()  # round-robin selection
        payloads: List[torch.Tensor] = []

        if child_level_idx is None:
            # Lowest high level: gather models from cliques
            for clique_id in scope.child_cliques:
                clique = hierarchy_state.cliques.get(clique_id)
                if clique is None or not clique.nodes:
                    continue
                fanout_node = clique.select_fanout()
                payloads.append(X[fanout_node].clone())
        else:
            child_scopes = hierarchy_state.scopes_by_level[child_level_idx]
            for child_id in scope.instance.child_ids:
                child_scope = child_scopes.get(child_id)
                if child_scope is None or not child_scope.instance.nodes:
                    continue
                fanout_node = child_scope.select_fanout_node()
                if child_scope.latest_model is None:
                    payloads.append(X[fanout_node].clone())
                else:
                    payloads.append(child_scope.latest_model.clone())

        if not payloads:
            continue

        aggregated = torch.stack(payloads, dim=0).mean(dim=0)
        scope.latest_model = aggregated.clone()

        idx = torch.tensor(scope.instance.nodes, device=device, dtype=torch.long)
        expanded = aggregated.unsqueeze(0).expand(len(scope.instance.nodes), -1)

        if cfg.apply_policy == "replace":
            X[idx] = expanded
        elif cfg.apply_policy == "interpolate":
            alpha = cfg.apply_alpha
            X[idx] = (1 - alpha) * X[idx] + alpha * expanded
        else:
            raise ValueError(f"Unsupported apply_policy '{cfg.apply_policy}'")

    return X


def _schedule_next_round(level: LevelRuntime, reference_time: float) -> None:
    cfg = level.config
    if not cfg.enabled or cfg.interval_seconds <= 0:
        level.next_round_time = None
        return

    next_time = (level.next_round_time or reference_time) + cfg.interval_seconds
    while next_time <= reference_time:
        next_time += cfg.interval_seconds
    level.next_round_time = next_time
