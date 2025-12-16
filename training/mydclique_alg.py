import torch
from .vector import get_param_matrix, set_param_matrix
from .dsgd import local_sgd_step

def build_agg_selector(cliques: list[list[int]], mode: str = "first") -> list[int]:
    if mode == "first":
        return [c[0] for c in cliques]
    raise ValueError("Unsupported mode")

def run_steps_mydclique(models, optims, loaders, cliques, agg_nodes, W_inter_agg: torch.Tensor, device, steps: int):
    iters = [iter(ld) for ld in loaders]
    n = len(models)
    agg_nodes = list(agg_nodes)

    for _ in range(steps):
        for i in range(n):
            try:
                batch = next(iters[i])
            except StopIteration:
                iters[i] = iter(loaders[i])
                batch = next(iters[i])
            local_sgd_step(models[i], optims[i], batch, device)

        with torch.no_grad():
            X = get_param_matrix(models).to(device)

            # clique average
            for clique in cliques:
                idx = torch.tensor(clique, device=device, dtype=torch.long)
                mean_vec = X.index_select(0, idx).mean(dim=0)
                X[idx] = mean_vec

            # aggregator-only inter-clique mixing
            Xa = X[agg_nodes]
            Xa = W_inter_agg @ Xa
            X[agg_nodes] = Xa

            # broadcast agg back to clique
            for c_idx, clique in enumerate(cliques):
                agg = agg_nodes[c_idx]
                idx = torch.tensor(clique, device=device, dtype=torch.long)
                X[idx] = X[agg]

            set_param_matrix(models, X)
