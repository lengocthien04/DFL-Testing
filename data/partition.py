import numpy as np

def dirichlet_partition(labels: np.ndarray, n_nodes: int, alpha: float, min_size: int, seed: int):
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
            props = rng.dirichlet(alpha * np.ones(n_nodes))
            cuts = (np.cumsum(props) * len(idx)).astype(int)
            splits = np.split(idx, cuts[:-1])
            for i in range(n_nodes):
                node_indices[i].extend(splits[i].tolist())

        if min(len(v) for v in node_indices) >= min_size:
            break

    return [np.array(sorted(v), dtype=np.int64) for v in node_indices]
