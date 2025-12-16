import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from .partition import dirichlet_partition

def load_mnist(root: str = "./data"):
    tf = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return train, test

def make_mnist_loaders(train_set, n_nodes: int, alpha: float, batch_size: int, seed: int):
    labels = np.array(train_set.targets, dtype=np.int64)
    node_idx = dirichlet_partition(labels, n_nodes, alpha, min_size=batch_size, seed=seed)
    loaders = []
    for idx in node_idx:
        subset = Subset(train_set, idx)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders, node_idx
