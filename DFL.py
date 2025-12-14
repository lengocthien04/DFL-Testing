#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Your D-Clique Aggregator variant (swap-based clique construction).

Pipeline:
1) Dirichlet non-IID data split (alpha)
2) Compute per-node label distributions Pi
3) Build cliques using random init + greedy node swaps (same as paper-style D-Cliques)
4) Training each epoch:
   - local SGD step at each node (one batch)
   - clique model averaging (hard consensus within clique)
   - aggregator-only inter-clique mixing (clique ring, Metropolis)
   - broadcast mixed clique model back to all nodes in the clique
5) Log each epoch to my_agg_output.txt as: epoch,mean,min,max
6) Plot mean accuracy to my_agg_accuracy.png
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
    batch_size = 128
    epochs = 100
    lr = 0.1
    alpha = 0.1
    seed = 42

    max_swaps = 20000  # clique swap improvement steps


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


# =====================
# MODEL
# =====================

class LogisticMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


# =====================
# DATA
# =====================

def load_mnist():
    tf = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return train, test


def dirichlet_partition_min(
    labels,
    n_nodes,
    alpha,
    min_size,
    seed
):
    """
    Dirichlet non-IID partition with HARD minimum samples per node.

    Guarantees:
    - every node has at least `min_size` samples
    - no empty DataLoaders
    - still non-IID
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    K = len(np.unique(labels))

    idx_by_class = {k: np.where(labels == k)[0] for k in range(K)}
    for k in idx_by_class:
        rng.shuffle(idx_by_class[k])

    node_indices = [[] for _ in range(n_nodes)]

    # ---------- Phase 1: guarantee minimum ----------
    all_indices = np.concatenate(list(idx_by_class.values()))
    rng.shuffle(all_indices)

    for i in range(n_nodes):
        node_indices[i].extend(
            all_indices[i * min_size:(i + 1) * min_size].tolist()
        )

    # remove assigned samples
    used = set(all_indices[:n_nodes * min_size])
    remaining_idx = np.array(
        [i for i in all_indices if i not in used],
        dtype=np.int64
    )

    # ---------- Phase 2: Dirichlet on remaining ----------
    rem_labels = labels[remaining_idx]
    idx_by_class_rem = {
        k: remaining_idx[rem_labels == k]
        for k in range(K)
    }

    for k in range(K):
        idx = idx_by_class_rem[k]
        if len(idx) == 0:
            continue

        props = rng.dirichlet(alpha * np.ones(n_nodes))
        cuts = (np.cumsum(props) * len(idx)).astype(int)
        splits = np.split(idx, cuts[:-1])

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
# D-CLIQUES SWAP CONSTRUCTION
# =====================

def compute_label_distributions(labels, node_indices):
    labels = np.asarray(labels)
    K = len(np.unique(labels))
    Pi = np.zeros((len(node_indices), K), dtype=np.float64)

    for i, idx in enumerate(node_indices):
        if len(idx) == 0:
            continue
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
        i = Ca[i_pos]
        j = Cb[j_pos]

        Ca2 = Ca.copy()
        Cb2 = Cb.copy()
        Ca2[i_pos] = j
        Cb2[j_pos] = i

        new_cliques = cliques.copy()
        new_cliques[a] = Ca2
        new_cliques[b] = Cb2

        new_cost = clique_cost(new_cliques, Pi, global_pi)
        if new_cost < best:
            cliques = new_cliques
            best = new_cost

    return cliques


# =====================
# MIXING FOR YOUR VARIANT
# =====================

def clique_averaging_matrix(n_nodes, cliques, device):
    """Hard model consensus within each clique."""
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for C in cliques:
        m = len(C)
        for i in C:
            for j in C:
                W[i, j] = 1.0 / m
    return torch.tensor(W, dtype=torch.float32, device=device)


def clique_ring_metropolis(num_cliques, device):
    """Metropolis weights over clique-ring graph."""
    A = np.zeros((num_cliques, num_cliques), dtype=np.int32)
    for c in range(num_cliques):
        a = c
        b = (c + 1) % num_cliques
        A[a, b] = 1
        A[b, a] = 1

    deg = A.sum(axis=1)
    W = np.zeros((num_cliques, num_cliques), dtype=np.float32)

    for i in range(num_cliques):
        for j in range(num_cliques):
            if A[i, j] == 1:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()

    return torch.tensor(W, dtype=torch.float32, device=device)


# =====================
# PARAM HELPERS
# =====================

def get_param_vec(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vec(model, vec):
    k = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[k:k+n].view_as(p.data))
        k += n


# =====================
# TRAIN / EVAL
# =====================

def local_sgd_one_batch(model, optim, loader, device):
    model.train()
    x, y = next(iter(loader))
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



# =====================
# MAIN
# =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set, test_set = load_mnist()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    labels = np.array(train_set.targets)

    node_indices = dirichlet_partition_min(
        labels,
        n_nodes=CFG.n_nodes,
        alpha=CFG.alpha,
        min_size=CFG.batch_size,   # <-- key line
        seed=CFG.seed
    )
    loaders = make_loaders(train_set, node_indices, CFG.batch_size)

    Pi = compute_label_distributions(labels, node_indices)

    print("Building D-Cliques via random init + greedy node swaps (for your variant)...")
    cliques = greedy_swap_dcliques(
        Pi,
        clique_size=CFG.clique_size,
        max_swaps=CFG.max_swaps,
        seed=CFG.seed
    )
    num_cliques = len(cliques)
    print("Number of cliques:", num_cliques)

    W_intra = clique_averaging_matrix(CFG.n_nodes, cliques, device)
    W_inter = clique_ring_metropolis(num_cliques, device)

    # Aggregator per clique: pick first node in clique (deterministic)
    aggregators = [C[0] for C in cliques]

    models = [LogisticMNIST().to(device) for _ in range(CFG.n_nodes)]
    optims = [torch.optim.SGD(m.parameters(), lr=CFG.lr) for m in models]

    d = get_param_vec(models[0]).numel()
    X = torch.stack([get_param_vec(m) for m in models], dim=0).to(device)

    out_path = "my_agg_output.txt"
    f = open(out_path, "w")

    mean_curve = []
    median_curve = []


    for epoch in range(1, CFG.epochs + 1):
        # 1) local SGD on each node (one batch)
        for i in range(CFG.n_nodes):
            local_sgd_one_batch(models[i], optims[i], loaders[i], device)

        # 2) collect params into X
        for i in range(CFG.n_nodes):
            X[i] = get_param_vec(models[i])

        # 3) intra-clique model averaging (hard consensus)
        X = W_intra @ X

        # 4) take one model per clique (aggregator)
        Z = torch.zeros((num_cliques, d), dtype=X.dtype, device=device)
        for c in range(num_cliques):
            Z[c] = X[aggregators[c]]

        # 5) inter-clique mixing among aggregators
        Z = W_inter @ Z

        # 6) broadcast back to each clique
        for c, C in enumerate(cliques):
            for node_id in C:
                X[node_id] = Z[c]

        # 7) write back
        for i in range(CFG.n_nodes):
            set_param_vec(models[i], X[i])

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
    print("Saved:", out_path)

    # Plot (same simple style)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, CFG.epochs + 1), [m * 100 for m in mean_curve], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean accuracy (%)")
    plt.title("Your D-Clique Aggregator Variant - MNIST")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("my_agg_accuracy.png", dpi=200)
    plt.show()
    print("Saved: my_agg_accuracy.png")


if __name__ == "__main__":
    main()
