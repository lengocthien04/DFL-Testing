import torch
import torch.nn as nn
import torch.nn.functional as F

from .vector import get_param_matrix, set_param_matrix

def local_sgd_step(model: nn.Module, optim: torch.optim.Optimizer, batch, device: torch.device):
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optim.step()

def run_steps_plain_dsgd(models, optims, loaders, W: torch.Tensor, device: torch.device, steps: int):
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
            X = W @ X
            set_param_matrix(models, X)
