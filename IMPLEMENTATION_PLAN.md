## Hierarchical Aggregation – Implementation Plan

### 1. Configuration & Metadata Loading
- Create a `config/` folder that holds `hierarchy_config.json` (level definitions) and `nodes_map.json` (scope membership).
- Implement a helper module (e.g., `utils/hierarchy_config.py`) that
  - Loads both JSON files.
  - Validates required keys (`scope_index`, `scope_name`, `apply_policy`, etc.).
  - Normalizes node identifiers by mapping integer IDs in the JSON to internal node indices.
  - Builds lookup tables: node → state, state → nation, and enumerations for any future higher levels.

### 2. Clique Construction per Lowest High-Level Scope
- Extend the topology builder so it can accept “scope → node indices” partitions.
- For each state (the level immediately above cliques), run the current D-Clique partitioner using only the nodes in that state. This keeps cliques local to their parent state.
- Consolidate all per-state cliques into a global `cliques` list while retaining metadata that maps each clique to its parent state and nation. Store this mapping for later use in aggregation rounds.

### 3. Hierarchical Runtime State
- Introduce data classes/containers describing each level instance:
  - Static config (intervals, wait times, apply policy, etc.).
  - Dynamic round-robin pointers for aggregator and fan-out selection.
  - Latest aggregated model tensors held per level so any node can forward models if selected as fan-out.
- Build scheduling logic that tracks elapsed training steps/time and determines when a high-level round must run. Ensure ordering constraints: if multiple levels are due, execute them from lowest high level (state) upward before starting the next clique round.

### 4. Hierarchical Training Runner
- Implement `run_steps_hierarchical_mydclique` inside a new module (e.g., `training/hierarchical_mydclique.py`):
  1. Perform the usual local SGD updates.
  2. Execute intra-state clique rounds (fan-out selection within each clique, aggregator selection within the parent scope, waiting logic, averaging).
  3. Check scheduled high-level rounds and run them sequentially:
     - Each lower scope uploads its latest model via its selected fan-out nodes.
     - Aggregator performs uniform averaging and stores the resulting model as the latest level model.
     - All member nodes pull+apply the new model using the configured `apply_policy` (`replace` or `interpolate` with `apply_alpha`).
  4. After high levels finish, resume clique rounds.
- Ensure every node maintains:
  - Local model tensor.
  - Cached copy of the latest models for each high level it participates in (needed when acting as fan-out).

### 5. Entry Points & Testing Hooks
- Add a new CLI method (e.g., `--method hierarchy`) in `run_mnist.py` / `run_cifar10.py` that:
  - Loads hierarchy config + node map paths from CLI arguments.
  - Builds per-state cliques and adjacency just like the existing D-Clique path.
  - Calls the new hierarchical runner.
- Provide sample JSON config files in `config/` using the structures from the prompt for quick testing.
- Validate via a dry-run script or unit-style test to ensure:
  - Cliques are partitioned per state correctly.
  - Round-robin roles rotate deterministically.
  - High-level rounds honor ordering constraints when multiple rounds are queued.
