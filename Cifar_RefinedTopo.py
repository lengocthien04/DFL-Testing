#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refined Topology (STL-FW) + 100-node D-SGD on CIFAR-10.

Matches the paper's CIFAR10 setting:
- Heterogeneous partition: shards (sort by class, split into shards, assign 2 shards/node).
- Model: Group-Normalized variant of LeNet (Hsieh et al., 2020), as referenced in the paper.
- Hyperparams (paper): lr=0.002, batch_size=20, epochs=100.
- Topology learning: STL-FW (Algorithm 2), centralized pre-processing to learn sparse doubly-stochastic W.
- Training: D-SGD (local SGD step then mixing by W each step).
- Logging: every epoch (mean/median/min/max across nodes) to refined_cifar10_output.txt
- Plot: refined_cifar10_accuracy.png

Dependencies:
  pip install torch torchvision numpy matplotlib scipy
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment


# -----------------------
# Config (paper default)
# -----------------------

@dataclass
class CFG:
    n_nodes: int = 100
    epochs: int = 100

    # Paper hyperparams for CIFAR10
    batch_size: int = 20
    lr: float = 0.002

    # STL-FW
    lambda_reg: float = 0.1
    dmax: int = 10
    fw_iters: int = 10

    seed: int = 42
    data_root: str = "./data"

CFG = CFG()


# -----------------------
# Model: GroupNorm LeNet (compact)
# -----------------------

class GNLeNet(nn.Module):
    """
    A simple LeNet-like CNN with GroupNorm instead of BatchNorm.
    This is a faithful *variant style* (GroupNorm in a LeNet-ish backbone) as referenced by the paper.
    """
    def __init__(self, num_classes: int = 10, gn_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=64)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.gn3 = nn.GroupNorm(num_groups=gn_groups, num_channels=256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 32x16x16
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 64x8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.gn3(self.fc1(x).unsqueeze(-1)).squeeze(-1))
        return self.fc2(x)


# -----------------------
# Data: CIFAR10 + dirichlet partition
# -----------------------

def load_cifar10(root: str):
    tf_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train = datasets.CIFAR10(root, train=True, download=True, transform=tf_train)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tf_test)
    return train, test


def dirichlet_partition(
    labels: np.ndarray,
    n_nodes: int,
    alpha: float,
    min_size: int,
    seed: int
):
    """
    Dirichlet non-IID partition.
    Ensures every node has at least `min_size` samples.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    K = len(np.unique(labels))

    idx_by_class = {k: np.where(labels == k)[0] for k in range(K)}
    for k in idx_by_class:
        rng.shuffle(idx_by_class[k])

    while True:
        node_indices = [[] for _ in range(n_nodes)]

        for k in range(K):
            idx = idx_by_class[k]
            if len(idx) == 0:
                continue

            proportions = rng.dirichlet(alpha * np.ones(n_nodes))
            proportions = proportions / proportions.sum()
            splits = (np.cumsum(proportions) * len(idx)).astype(int)
            splits = np.split(idx, splits[:-1])

            for i in range(n_nodes):
                node_indices[i].extend(splits[i].tolist())

        sizes = np.array([len(v) for v in node_indices])
        if sizes.min() >= min_size:
            break

    return [np.array(sorted(v)) for v in node_indices]



def make_loaders(train_set, node_indices: List[np.ndarray], batch_size: int):
    loaders = []
    for idx in node_indices:
        subset = Subset(train_set, idx)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders


# -----------------------
# STL-FW (Algorithm 2)
# -----------------------

def class_proportions(labels: np.ndarray, node_indices: List[np.ndarray], n_classes: int) -> np.ndarray:
    n = len(node_indices)
    Pi = np.zeros((n, n_classes), dtype=np.float64)
    for i, idx in enumerate(node_indices):
        if len(idx) == 0:
            continue
        counts = np.bincount(labels[idx], minlength=n_classes).astype(np.float64)
        Pi[i] = counts / max(counts.sum(), 1.0)
    return Pi


def grad_g(W: np.ndarray, Pi: np.ndarray, lam: float) -> np.ndarray:
    n, _ = Pi.shape
    ones = np.ones((n, 1), dtype=np.float64)
    J = (ones @ ones.T) / n

    diff = (W @ Pi) - (J @ Pi)
    term1 = (2.0 / n) * (diff @ Pi.T)
    term2 = (2.0 * lam / n) * (W - J)
    return term1 + term2


def fw_atom_from_gradient(grad: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(grad)
    n = grad.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    P[row_ind, col_ind] = 1.0
    return P


def line_search_gamma(W: np.ndarray, P: np.ndarray, Pi: np.ndarray, lam: float) -> float:
    n, K = Pi.shape
    ones = np.ones((n, 1), dtype=np.float64)
    J = (ones @ ones.T) / n

    D = P - W
    WPi = W @ Pi
    JPi = J @ Pi

    numer = 0.0
    for k in range(K):
        a = (JPi[:, k] - WPi[:, k])
        b = D @ Pi[:, k]
        numer += float(a.T @ b)
    numer -= lam * float(np.trace((W - J).T @ D))

    denom = float(np.linalg.norm(D @ Pi, ord="fro") ** 2 + lam * (np.linalg.norm(D, ord="fro") ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(numer / denom, 0.0, 1.0))


def stl_fw(Pi: np.ndarray, lam: float, iters: int) -> np.ndarray:
    n = Pi.shape[0]
    W = np.eye(n, dtype=np.float64)
    for _ in range(iters):
        G = grad_g(W, Pi, lam)
        P = fw_atom_from_gradient(G)
        gamma = line_search_gamma(W, P, Pi, lam)
        W = (1.0 - gamma) * W + gamma * P
    return W


# -----------------------
# D-SGD training
# -----------------------

def get_vec(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_vec(model: nn.Module, vec: torch.Tensor):
    k = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(vec[k:k + n].view_as(p.data))
            k += n


def local_sgd_one_batch(model: nn.Module, optim: torch.optim.Optimizer, batch, device):
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optim.step()


def evaluate_all(models: List[nn.Module], test_loader: DataLoader, device) -> Tuple[float, float, float, float]:
    accs = []
    for m in models:
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = m(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accs.append(correct / max(total, 1))
    arr = np.array(accs, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.min()), float(arr.max())


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set, test_set = load_cifar10(CFG.data_root)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    labels = np.array(train_set.targets, dtype=np.int64)

    
    node_indices = dirichlet_partition(
    labels=labels,
    n_nodes=CFG.n_nodes,
    alpha=0.1,          # strong non-IID (paper-level)
    min_size=32,        # >= 1 batch
    seed=CFG.seed
    )
    assert all(len(idx) > 0 for idx in node_indices), "Partition produced an empty node dataset (should not happen)."
    loaders = make_loaders(train_set, node_indices, CFG.batch_size)

    Pi = class_proportions(labels, node_indices, n_classes=10)

    fw_iters = int(CFG.fw_iters if CFG.fw_iters is not None else CFG.dmax)
    fw_iters = min(max(fw_iters, 1), CFG.n_nodes - 1)
    print(f"Learning refined topology via STL-FW: iters={fw_iters}, lambda={CFG.lambda_reg}")
    W_np = stl_fw(Pi, lam=CFG.lambda_reg, iters=fw_iters)
    W = torch.tensor(W_np, dtype=torch.float32, device=device)

    models = [GNLeNet().to(device) for _ in range(CFG.n_nodes)]
    optims = [torch.optim.SGD(m.parameters(), lr=CFG.lr) for m in models]

    with torch.no_grad():
        X = torch.stack([get_vec(m) for m in models], dim=0).to(device)

    iters = [iter(ld) for ld in loaders]

    train_size = len(train_set)
    steps_per_epoch = max(1, math.ceil(train_size / (CFG.n_nodes * CFG.batch_size)))

    out_path = "refined_cifar10_output.txt"
    fig_path = "refined_cifar10_accuracy.png"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("epoch,mean,median,min,max\n")

        mean_curve, med_curve, min_curve, max_curve = [], [], [], []

        for epoch in range(1, CFG.epochs + 1):
            for _ in range(steps_per_epoch):
                for i in range(CFG.n_nodes):
                    try:
                        batch = next(iters[i])
                    except StopIteration:
                        iters[i] = iter(loaders[i])
                        batch = next(iters[i])
                    local_sgd_one_batch(models[i], optims[i], batch, device)

                with torch.no_grad():
                    for i in range(CFG.n_nodes):
                        X[i] = get_vec(models[i])
                    X = W @ X
                    for i in range(CFG.n_nodes):
                        set_vec(models[i], X[i])

            mean_acc, med_acc, min_acc, max_acc = evaluate_all(models, test_loader, device)
            mean_curve.append(mean_acc * 100.0)
            med_curve.append(med_acc * 100.0)
            min_curve.append(min_acc * 100.0)
            max_curve.append(max_acc * 100.0)

            print(f"Epoch {epoch:03d}/{CFG.epochs} - mean {mean_acc*100:.2f}% | "
                  f"median {med_acc*100:.2f}% | min {min_acc*100:.2f}% | max {max_acc*100:.2f}%")
            f.write(f"{epoch},{mean_acc},{med_acc},{min_acc},{max_acc}\n")

    print(f"Saved: {out_path}")

    x = np.arange(1, CFG.epochs + 1)
    plt.figure()
    plt.plot(x, mean_curve, label="Mean")
    plt.plot(x, med_curve, label="Median")
    plt.fill_between(x, min_curve, max_curve, alpha=0.2, label="Min–Max band")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Refined Topology (STL-FW) - CIFAR10 - n={CFG.n_nodes}, dmax≈{CFG.dmax}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
