import torch
import torch.nn as nn

def get_param_matrix(models: list[nn.Module]) -> torch.Tensor:
    vecs = []
    for m in models:
        vecs.append(torch.cat([p.data.view(-1) for p in m.parameters()]))
    return torch.stack(vecs, dim=0)

def set_param_matrix(models: list[nn.Module], X: torch.Tensor):
    with torch.no_grad():
        for i, m in enumerate(models):
            k = 0
            for p in m.parameters():
                n = p.numel()
                p.data.copy_(X[i, k:k+n].view_as(p.data))
                k += n
