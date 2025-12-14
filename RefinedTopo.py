#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refined Topology (clean, practical):
- Dirichlet non-IID
- Learn sparse topology by degree-constrained edge swaps to reduce heterogeneity proxy
  H(W) = ||W Pi - Pi_global||^2
- Use Metropolis mixing for D-SGD: X <- W @ X
- Logs every epoch to refined_output.txt
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

    d_max = 10               # max degree
    refine_steps = 2000     # number of edge-swap attempts
    candidates_per_step = 5 # tries per step (keeps it robust)

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


def compute_label_distributions(labels, node_indices):
    labels = np.array(labels)
    K = len(np.unique(labels))
    Pi = np.zeros((len(node_indices), K), dtype=np.float64)
    for i, idx in enumerate(node_indices):
        c = np.bincount(labels[idx], minlength=K)
        Pi[i] = c / (c.sum() + 1e-12)
    return Pi


# ---------- topology utils ----------

def random_dmax_graph(n, d_max, rng):
    """Build a connected-ish undirected graph with max degree d_max (simple heuristic)."""
    A = np.zeros((n, n), dtype=np.int32)
    deg = np.zeros(n, dtype=np.int32)

    # Start with a ring to guarantee connectivity
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = 1
    deg = A.sum(axis=1)

    # Add random edges until degrees approach d_max
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


def metropolis_from_adj(A):
    n = A.shape[0]
    deg = A.sum(axis=1)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def heterogeneity_cost(W, Pi):
    Pi_global = Pi.mean(axis=0)
    mixed = W @ Pi
    diff = mixed - Pi_global
    return float(np.sum(diff * diff))


def try_edge_swap(A, rng, d_max):
    """
    Propose a swap:
    remove one existing edge (u,v), add one non-edge (a,b),
    while respecting degree <= d_max and keeping symmetry.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)

    # pick an existing edge
    edges = np.argwhere(np.triu(A, 1) == 1)
    if len(edges) == 0:
        return None
    u, v = edges[rng.randint(0, len(edges))]

    # pick a non-edge
    non_edges = np.argwhere(np.triu(A, 1) == 0)
    # avoid self-loops via triu; still includes diagonal zeros? triu excludes diag; OK.
    if len(non_edges) == 0:
        return None

    # attempt a few random non-edges
    for _ in range(50):
        a, b = non_edges[rng.randint(0, len(non_edges))]
        if a == b:
            continue
        if A[a, b] == 1:
            continue

        # simulate degrees after swap
        deg2 = deg.copy()
        deg2[u] -= 1; deg2[v] -= 1
        deg2[a] += 1; deg2[b] += 1

        if np.any(deg2 < 1):      # keep at least ring-like connectivity pressure
            continue
        if deg2[a] > d_max or deg2[b] > d_max:
            continue

        return (u, v, a, b)

    return None


def refine_topology(Pi, d_max, steps, candidates_per_step, seed):
    rng = np.random.RandomState(seed)
    n = Pi.shape[0]

    A = random_dmax_graph(n, d_max, rng)
    W = metropolis_from_adj(A)
    best_cost = heterogeneity_cost(W, Pi)

    for _ in range(steps):
        best_local = None
        best_local_cost = best_cost

        for __ in range(candidates_per_step):
            prop = try_edge_swap(A, rng, d_max)
            if prop is None:
                continue
            u, v, a, b = prop

            A2 = A.copy()
            # remove (u,v)
            A2[u, v] = A2[v, u] = 0
            # add (a,b)
            A2[a, b] = A2[b, a] = 1

            W2 = metropolis_from_adj(A2)
            c2 = heterogeneity_cost(W2, Pi)
            if c2 < best_local_cost:
                best_local_cost = c2
                best_local = (A2, W2)

        if best_local is not None:
            A, W = best_local
            best_cost = best_local_cost

    return A, W


# ---------- training utils ----------

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

    train_set, test_set = load_mnist()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    labels = np.array(train_set.targets)

    node_indices = noniid_dirichlet_partition(labels, CFG.n_nodes, CFG.alpha, CFG.seed)
    loaders = make_loaders(train_set, node_indices, CFG.batch_size)

    Pi = compute_label_distributions(labels, node_indices)

    print("Learning refined topology...")
    A, W_np = refine_topology(
        Pi,
        d_max=CFG.d_max,
        steps=CFG.refine_steps,
        candidates_per_step=CFG.candidates_per_step,
        seed=CFG.seed
    )
    W = torch.tensor(W_np, dtype=torch.float32, device=device)
    print("Refined W built. shape:", W.shape)

    models = [LogisticMNIST().to(device) for _ in range(CFG.n_nodes)]
    optims = [torch.optim.SGD(m.parameters(), lr=CFG.lr) for m in models]

    with torch.no_grad():
        X = torch.stack([get_vec(m) for m in models], dim=0).to(device)

    out_path = "refined_output.txt"
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
    plt.title("Refined Topology - MNIST")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("refined_accuracy.png")
    print("Saved: refined_accuracy.png")


if __name__ == "__main__":
    main()
