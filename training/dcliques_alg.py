import torch
from .vector import get_param_matrix, set_param_matrix
from .dsgd import local_sgd_step

def run_steps_dcliques_two_stage(models, optims, loaders, W_clique: torch.Tensor, W_param: torch.Tensor, device, steps: int):
    iters = [iter(ld) for ld in loaders]
    n = len(models)
    for _ in range(steps):
        # Snapshot parameters before local steps so we can approximate
        # clique-averaged gradients via clique-averaged parameter deltas.
        with torch.no_grad():
            X0 = get_param_matrix(models).to(device)

        for i in range(n):
            try:
                batch = next(iters[i])
            except StopIteration:
                iters[i] = iter(loaders[i])
                batch = next(iters[i])
            local_sgd_step(models[i], optims[i], batch, device)

        with torch.no_grad():
            X1 = get_param_matrix(models).to(device)

            # --- Clique Averaging (Alg. 3 spirit) ---
            # The paper averages *gradients* within each clique, then applies
            # the SGD step, then mixes models over the full topology.
            # We implement a practical surrogate without needing explicit
            # gradients: average the *local deltas* (X1 - X0) within cliques.
            dX = X1 - X0
            dX = W_clique @ dX
            X = X0 + dX

            # --- Model Mixing over full topology (Metropolis W) ---
            X = W_param @ X
            set_param_matrix(models, X)
