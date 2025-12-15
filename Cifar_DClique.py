#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
D-Cliques on CIFAR-10 (GN-LeNet) with paper-style clique construction:
- Random init cliques
- Greedy node swaps to balance label distributions (minimize clique cost)

Training update (per epoch):
1) Compute local stochastic gradients (one batch per node)
2) Clique gradient averaging: G <- W_clique @ G
3) SGD update on parameter vectors: X <- X - lr * G
4) Graph parameter mixing over D-Cliques adjacency: X <- W_param @ X (Metropolis)
5) Write back to each node model

Logs per epoch to dclique_output.txt:
epoch,mean,median,min,max
Saves plot to dclique_accuracy.png
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

class CFG:
    n_nodes = 100
    clique_size = 10

    batch_size = 20
    epochs = 100
    lr = 0.002

    seed = 42
    alpha = 0.1

    # clique construction swaps
    max_swaps = 20000


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


# =====================
# MODEL: GN-LeNet (CIFAR-10)
# =====================

class GNLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(8, 64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.gn3 = nn.GroupNorm(8, 256)

        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.gn3(self.fc1(x).unsqueeze(-1))).squeeze(-1)
        return self.fc2(x)



# =====================
# DATA
# =====================

def load_cifar10():
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    return train, test


def dirichlet_partition_min(labels, n_nodes, alpha, min_size, seed):
    """
    Dirichlet non-IID split with HARD minimum samples per node (>= min_size).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    K = len(np.unique(labels))

    idx_by_class = {k: np.where(labels == k)[0] for k in range(K)}
    for k in idx_by_class:
        rng.shuffle(idx_by_class[k])

    node_indices = [[] for _ in range(n_nodes)]

    # Phase 1: guarantee minimum
    all_idx = np.concatenate(list(idx_by_class.values()))
    rng.shuffle(all_idx)

    need = n_nodes * min_size
    if need > len(all_idx):
        raise ValueError("min_size too large: not enough total samples.")

    for i in range(n_nodes):
        node_indices[i].extend(all_idx[i * min_size:(i + 1) * min_size].tolist())

    used = set(all_idx[:need])
    remaining = np.array([i for i in all_idx if i not in used], dtype=np.int64)

    # Phase 2: Dirichlet on remaining
    rem_labels = labels[remaining]
    idx_by_class_rem = {k: remaining[rem_labels == k] for k in range(K)}

    for k in range(K):
        idx = idx_by_class_rem[k]
        if len(idx) == 0:
            continue
        props = rng.dirichlet(alpha * np.ones(n_nodes))
        cut = (np.cumsum(props) * len(idx)).astype(int)
        splits = np.split(idx, cut[:-1])
        for i in range(n_nodes):
            node_indices[i].extend(splits[i].tolist())

    return [np.array(v, dtype=np.int64) for v in node_indices]


def make_loaders(train_set, node_indices, batch_size):
    loaders = []
    for idx in node_indices:
        subset = Subset(train_set, idx)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True))
    return loaders


# =====================
# D-CLIQUES CONSTRUCTION (random init + greedy node swaps)
# =====================

def compute_label_distributions(labels, node_indices):
    labels = np.asarray(labels)
    K = len(np.unique(labels))
    Pi = np.zeros((len(node_indices), K), dtype=np.float64)

    for i, idx in enumerate(node_indices):
        counts = np.bincount(labels[idx], minlength=K)
        Pi[i] = counts / (counts.sum() + 1e-12)
    return Pi


def random_init_cliques(n_nodes, clique_size, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_nodes)
    return [perm[i:i + clique_size].tolist() for i in range(0, n_nodes, clique_size)]


def clique_cost(cliques, Pi, global_pi):
    cost = 0.0
    for C in cliques:
        pi_c = Pi[C].mean(axis=0)
        diff = pi_c - global_pi
        cost += float(np.dot(diff, diff))
    return cost


def greedy_swap_dcliques(Pi, clique_size, max_swaps, seed):
    rng = np.random.RandomState(seed)
    n_nodes = Pi.shape[0]
    global_pi = Pi.mean(axis=0)

    cliques = random_init_cliques(n_nodes, clique_size, seed)
    best = clique_cost(cliques, Pi, global_pi)

    for _ in range(max_swaps):
        a, b = rng.choice(len(cliques), size=2, replace=False)
        Ca, Cb = cliques[a], cliques[b]

        i_pos = rng.randint(0, len(Ca))
        j_pos = rng.randint(0, len(Cb))

        Ca2 = Ca.copy()
        Cb2 = Cb.copy()
        Ca2[i_pos], Cb2[j_pos] = Cb2[j_pos], Ca2[i_pos]

        new_cliques = cliques.copy()
        new_cliques[a] = Ca2
        new_cliques[b] = Cb2

        new_cost = clique_cost(new_cliques, Pi, global_pi)
        if new_cost < best:
            cliques = new_cliques
            best = new_cost

    return cliques


# =====================
# TOPOLOGY + MIXING
# =====================

def build_dcliques_adjacency(n_nodes, cliques):
    """
    Node adjacency:
    - complete graph inside each clique
    - ring between clique representatives cliques[c][0]
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    for C in cliques:
        for i in C:
            for j in C:
                if i != j:
                    A[i, j] = 1

    K = len(cliques)
    for c in range(K):
        a = cliques[c][0]
        b = cliques[(c + 1) % K][0]
        A[a, b] = 1
        A[b, a] = 1

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


def clique_averaging_matrix(n_nodes, cliques, device):
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for C in cliques:
        m = len(C)
        for i in C:
            for j in C:
                W[i, j] = 1.0 / m
    return torch.tensor(W, dtype=torch.float32, device=device)


# =====================
# PARAM / GRAD VECTORS
# =====================

def get_param_vec(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_param_vec(model, vec):
    k = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[k:k + n].view_as(p.data))
        k += n


def get_grad_vec(model):
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p.data).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


# =====================
# TRAIN / EVAL
# =====================

def local_grad_one_batch(model, loader, device):
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    model.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return get_grad_vec(model)


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
        accs.append(correct / total)
    arr = np.array(accs)
    return float(arr.mean()), float(np.median(arr)), float(arr.min()), float(arr.max())


# =====================
# MAIN
# =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set, test_set = load_cifar10()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    labels = np.asarray(train_set.targets, dtype=np.int64)

    node_indices = dirichlet_partition_min(
        labels=labels,
        n_nodes=CFG.n_nodes,
        alpha=CFG.alpha,
        min_size=CFG.batch_size,
        seed=CFG.seed
    )
    loaders = make_loaders(train_set, node_indices, CFG.batch_size)

    Pi = compute_label_distributions(labels, node_indices)

    print("Building D-Cliques via random init + greedy node swaps...")
    cliques = greedy_swap_dcliques(Pi, CFG.clique_size, CFG.max_swaps, CFG.seed)
    print("Number of cliques:", len(cliques))

    A = build_dcliques_adjacency(CFG.n_nodes, cliques)
    W_param = metropolis_from_adj(A, device)
    W_clique = clique_averaging_matrix(CFG.n_nodes, cliques, device)

    models = [GNLeNet().to(device) for _ in range(CFG.n_nodes)]

    d = get_param_vec(models[0]).numel()
    X = torch.stack([get_param_vec(m) for m in models], dim=0).to(device)

    out_path = "dclique_output.txt"
    png_path = "dclique_accuracy.png"

    f = open(out_path, "w")
    mean_curve, median_curve = [], []

    for epoch in range(1, CFG.epochs + 1):
        # 1) local gradients
        G = torch.zeros((CFG.n_nodes, d), dtype=X.dtype, device=device)
        for i in range(CFG.n_nodes):
            G[i] = local_grad_one_batch(models[i], loaders[i], device)

        # 2) clique gradient averaging
        G = W_clique @ G

        # 3) SGD update
        X = X - CFG.lr * G

        # 4) Metropolis parameter mixing on D-Cliques graph
        X = W_param @ X

        # 5) write back
        for i in range(CFG.n_nodes):
            set_param_vec(models[i], X[i])

        mean_acc, median_acc, min_acc, max_acc = evaluate(models, test_loader, device)
        mean_curve.append(mean_acc)
        median_curve.append(median_acc)

        print(
            f"Epoch {epoch:03d}/{CFG.epochs} - "
            f"mean {mean_acc*100:.2f}% | median {median_acc*100:.2f}% | "
            f"min {min_acc*100:.2f}% | max {max_acc*100:.2f}%"
        )
        f.write(f"{epoch},{mean_acc},{median_acc},{min_acc},{max_acc}\n")

    f.close()
    print("Saved:", out_path)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, CFG.epochs + 1), [m * 100 for m in mean_curve], label="Mean", linewidth=2)
    plt.plot(range(1, CFG.epochs + 1), [m * 100 for m in median_curve], label="Median", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("D-Cliques - CIFAR-10 (GN-LeNet)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
