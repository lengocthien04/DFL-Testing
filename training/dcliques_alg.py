import torch
from .vector import get_param_matrix, set_param_matrix
from .dsgd import local_sgd_step

def run_steps_dcliques_two_stage(models, optims, loaders, W_clique: torch.Tensor, W_param: torch.Tensor, device, steps: int):
    iters = [iter(ld) for ld in loaders]
    n = len(models)
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
            X = W_clique @ X
            X = W_param @ X
            set_param_matrix(models, X)
