#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baselines for 100-node D-SGD on MNIST:
- topology = "random" : sparse random graph (max degree d_max), Metropolis mixing
- topology = "fully"  : fully-connected averaging (W = 1/n)

Outputs:
- random_output.txt + random_accuracy.png
- fully_output.txt  + fully_accuracy.png
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


class CFG:
    n_nodes = 100
    batch_size = 128
    epochs = 100
    lr = 0.1
    seed = 42
    alpha = 0.1

    topology = "fully"   # "random" or "fully"
    d_max = 10            # only used for random

torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


class LogisticMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def load_mnist():
    tf = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return train, test


def noniid_dirichlet_partition(labels, n_nodes, alpha, seed):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    K = len(np.unique(labels))
    idx_by_class = {c: np.where(labels == c)[0] for c in range(K)}
    node_indices = [[] for _ in range(n_nodes)]

    for c in range(K):
        idx = idx_by_class[c]
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(n_nodes))
        cut = (np.cumsum(props) * len(idx)).astype(int)
        splits = np.split(idx, cut[:-1])
        for i in range(n_nodes):
            node_indices[i].extend(splits[i].tolist())

    return [np.array(sorted(lst)) for lst in node_indices]


def make_loaders(train_set, node_indices, batch_size):
    loaders = []
    for idx in node_indices:
        subset = Subset(train_set, idx)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True))
    return loaders


def random_dmax_graph(n, d_max, rng):
    A = np.zeros((n, n), dtype=np.int32)
    # ring for connectivity
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = 1
    deg = A.sum(axis=1)

    tries = 0
    while np.any(deg < d_max) and tries < n * n * 10:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j or A[i, j] == 1:
            tries += 1
            continue
        if deg[i] >= d_max or deg[j] >= d_max:
            tries += 1
            continue
        A[i, j] = A[j, i] = 1
        deg[i] += 1
        deg[j] += 1
        tries += 1
    return A


def metropolis_from_adj(A, device):
    n = A.shape[0]
    deg = A.sum(axis=1)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return torch.tensor(W, dtype=torch.float32, device=device)


def fully_connected_W(n, device):
    return torch.ones((n, n), dtype=torch.float32, device=device) / float(n)


def get_vec(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_vec(model, vec):
    k = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[k:k+n].view_as(p.data))
        k += n

def local_sgd_step(model, optim, loader, device):
    model.train()
    try:
        x, y = next(iter(loader))
    except StopIteration:
        return
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optim.step()

def evaluate(models, test_loader, device):
    accs = []
    for m in models:
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = m(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accs.append(correct / max(total, 1))

    arr = np.array(accs)
    return arr.mean(), np.median(arr), arr.min(), arr.max()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Topology:", CFG.topology)

    train_set, test_set = load_mnist()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    labels = np.array(train_set.targets)

    node_indices = noniid_dirichlet_partition(labels, CFG.n_nodes, CFG.alpha, CFG.seed)
    loaders = make_loaders(train_set, node_indices, CFG.batch_size)

    # build W
    if CFG.topology == "fully":
        W = fully_connected_W(CFG.n_nodes, device)
        out_path = "fully_output.txt"
        png_path = "fully_accuracy.png"
    elif CFG.topology == "random":
        rng = np.random.RandomState(CFG.seed)
        A = random_dmax_graph(CFG.n_nodes, CFG.d_max, rng)
        W = metropolis_from_adj(A, device)
        out_path = "random_output.txt"
        png_path = "random_accuracy.png"
    else:
        raise ValueError("CFG.topology must be 'random' or 'fully'")

    models = [LogisticMNIST().to(device) for _ in range(CFG.n_nodes)]
    optims = [torch.optim.SGD(m.parameters(), lr=CFG.lr) for m in models]

    with torch.no_grad():
        X = torch.stack([get_vec(m) for m in models], dim=0).to(device)

    f = open(out_path, "w")
    mean_curve = []
    median_curve = []


    for epoch in range(1, CFG.epochs + 1):
        for i in range(CFG.n_nodes):
            local_sgd_step(models[i], optims[i], loaders[i], device)

        with torch.no_grad():
            for i in range(CFG.n_nodes):
                X[i] = get_vec(models[i])

            X = W @ X

            for i in range(CFG.n_nodes):
                set_vec(models[i], X[i])

        mean_acc, median_acc, min_acc, max_acc = evaluate(models, test_loader, device)

        mean_curve.append(mean_acc)
        median_curve.append(median_acc)

        print(
            f"Epoch {epoch:03d}/{CFG.epochs} - "
            f"mean {mean_acc*100:.2f}% | "
            f"median {median_acc*100:.2f}% | "
            f"min {min_acc*100:.2f}% | "
            f"max {max_acc*100:.2f}%"
        )

        f.write(f"{epoch},{mean_acc},{median_acc},{min_acc},{max_acc}\n")


    f.close()
    print(f"Saved: {out_path}")

    plt.figure()
    plt.plot(range(1, CFG.epochs + 1), [m * 100 for m in mean_curve])
    plt.xlabel("Epoch")
    plt.ylabel("Mean accuracy (%)")
    plt.title(f"{CFG.topology.upper()} Topology - MNIST")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
