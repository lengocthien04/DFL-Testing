#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
D-SGD baseline on CIFAR-10 with Group-Normalized LeNet
(Hsieh et al., 2020 style)

Topology:
- random : sparse random graph + Metropolis mixing
- fully  : fully connected averaging

Hyperparameters:
- batch size = 20
- epochs = 100
- lr = 0.002

Logs per epoch:
epoch, mean, median, min, max
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
    batch_size = 20
    epochs = 100
    lr = 0.002
    seed = 42
    alpha = 0.1

    topology = "random"   # "random" or "fully"
    d_max = 10            # only for random topology


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


# =====================
# MODEL: GN-LeNet
# =====================

class GNLeNet(nn.Module):
    """
    Group-Normalized LeNet (CIFAR-10)
    Following Hsieh et al., 2020
    """
    def __init__(self, num_groups=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.gn1 = nn.GroupNorm(num_groups, 6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gn2 = nn.GroupNorm(num_groups, 16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.gn3 = nn.GroupNorm(num_groups, 120)

        self.fc2 = nn.Linear(120, 84)
        self.gn4 = nn.GroupNorm(num_groups, 84)

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.gn3(self.fc1(x)))
        x = F.relu(self.gn4(self.fc2(x)))
        return self.fc3(x)


# =====================
# DATA
# =====================

def load_cifar10():
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    return train, test


def dirichlet_partition_min(labels, n_nodes, alpha, min_size, seed):
    """
    Dirichlet non-IID with HARD minimum samples per node.
    Guarantees each node has >= min_size samples.
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

    for i in range(n_nodes):
        node_indices[i].extend(
            all_idx[i * min_size:(i + 1) * min_size].tolist()
        )

    used = set(all_idx[:n_nodes * min_size])
    remaining = np.array([i for i in all_idx if i not in used], dtype=np.int64)

    # Phase 2: Dirichlet on remaining
    rem_labels = labels[remaining]
    idx_by_class_rem = {
        k: remaining[rem_labels == k] for k in range(K)
    }

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


def make_loaders(train_set, node_indices):
    loaders = []
    for idx in node_indices:
        subset = Subset(train_set, idx)
        loaders.append(
            DataLoader(subset, batch_size=CFG.batch_size,
                       shuffle=True, drop_last=True)
        )
    return loaders


# =====================
# TOPOLOGY + MIXING
# =====================

def random_dmax_graph(n, d_max, rng):
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = 1
    deg = A.sum(axis=1)

    tries = 0
    while np.any(deg < d_max) and tries < n * n * 10:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i != j and A[i, j] == 0 and deg[i] < d_max and deg[j] < d_max:
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
            if A[i, j]:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return torch.tensor(W, dtype=torch.float32, device=device)


def fully_connected_W(n, device):
    return torch.ones((n, n), dtype=torch.float32, device=device) / float(n)


# =====================
# UTIL
# =====================

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
        accs.append(correct / total)
    arr = np.array(accs)
    return arr.mean(), np.median(arr), arr.min(), arr.max()


# =====================
# MAIN
# =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Topology:", CFG.topology)

    train_set, test_set = load_cifar10()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    labels = train_set.targets

    node_indices = dirichlet_partition_min(
        labels,
        CFG.n_nodes,
        CFG.alpha,
        CFG.batch_size,
        CFG.seed
    )

    loaders = make_loaders(train_set, node_indices)

    if CFG.topology == "fully":
        W = fully_connected_W(CFG.n_nodes, device)
        out_path = "fully_output.txt"
        png_path = "fully_accuracy.png"
    else:
        rng = np.random.RandomState(CFG.seed)
        A = random_dmax_graph(CFG.n_nodes, CFG.d_max, rng)
        W = metropolis_from_adj(A, device)
        out_path = "random_output.txt"
        png_path = "random_accuracy.png"

    models = [GNLeNet().to(device) for _ in range(CFG.n_nodes)]
    optims = [torch.optim.SGD(m.parameters(), lr=CFG.lr) for m in models]

    with torch.no_grad():
        X = torch.stack([get_vec(m) for m in models]).to(device)

    f = open(out_path, "w")
    mean_curve, median_curve = [], []

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
    print("Saved:", out_path)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, CFG.epochs + 1), [m*100 for m in mean_curve], label="Mean", linewidth=2)
    plt.plot(range(1, CFG.epochs + 1), [m*100 for m in median_curve], label="Median", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"D-SGD CIFAR-10 ({CFG.topology.upper()})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
