import torch
from .vector import get_param_matrix, set_param_matrix
from .dsgd import local_sgd_step


def build_agg_selector(cliques: list[list[int]], mode: str = "first") -> list[int]:
    """
    Select one representative (aggregator) per clique.
    """
    if mode == "first":
        return [c[0] for c in cliques]
    raise ValueError("Unsupported mode")


def run_steps_mydclique(
    models,
    optims,
    loaders,
    cliques,
    agg_nodes,
    clique_neighbors,   # <<< REQUIRED
    device,
    steps: int,
):
    """
    MyD-Clique training step:
    1. Local SGD on all nodes
    2. Intra-clique averaging
    3. Inter-clique aggregation (aggregators only, topology-aware)
    4. Broadcast aggregator model back to clique
    """

    iters = [iter(ld) for ld in loaders]
    n = len(models)
    agg_nodes = list(agg_nodes)

    for _ in range(steps):

        # --------------------------------------------------
        # (1) Local SGD
        # --------------------------------------------------
        for i in range(n):
            try:
                batch = next(iters[i])
            except StopIteration:
                iters[i] = iter(loaders[i])
                batch = next(iters[i])
            local_sgd_step(models[i], optims[i], batch, device)

        with torch.no_grad():
            X = get_param_matrix(models).to(device)

            # --------------------------------------------------
            # (2) Intra-clique averaging
            # --------------------------------------------------
            for clique in cliques:
                idx = torch.tensor(clique, device=device, dtype=torch.long)
                mean_vec = X.index_select(0, idx).mean(dim=0)
                X[idx] = mean_vec

            # --------------------------------------------------
            # (3) Inter-clique aggregation (aggregators ONLY)
            #     Each aggregator averages with neighbor cliques
            # --------------------------------------------------
            new_agg_values = {}

            for c_idx, agg in enumerate(agg_nodes):
                neighbors = clique_neighbors[c_idx]

                # self + neighbor aggregators
                src_nodes = [agg] + [agg_nodes[nc] for nc in neighbors]
                idx = torch.tensor(src_nodes, device=device, dtype=torch.long)

                new_agg_values[agg] = X.index_select(0, idx).mean(dim=0)

            # apply aggregator updates
            for agg, vec in new_agg_values.items():
                X[agg] = vec

            # --------------------------------------------------
            # (4) Broadcast aggregator model back to clique
            # --------------------------------------------------
            for c_idx, clique in enumerate(cliques):
                agg = agg_nodes[c_idx]
                idx = torch.tensor(clique, device=device, dtype=torch.long)
                X[idx] = X[agg].clone()

            set_param_matrix(models, X)
