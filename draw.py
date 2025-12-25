import random
import math
import networkx as nx
import matplotlib.pyplot as plt


def block_positions(block_sizes, centers, radius=0.45, seed=42):
    """
    Place each SBM block in a separate region (circle around a block center),
    so the drawing visually shows blocks.
    """
    rng = random.Random(seed)
    pos = {}
    node = 0
    for b, size in enumerate(block_sizes):
        cx, cy = centers[b]
        for _ in range(size):
            # random point in a disk around (cx, cy)
            ang = rng.random() * 2 * math.pi
            r = radius * math.sqrt(rng.random())
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            pos[node] = (x, y)
            node += 1
    return pos


def main():
    # -------------------------------
    # SBM configuration
    # -------------------------------
    block_sizes = [10, 10, 10, 10, 10, 10]  # 6 blocks
    p_intra = 0.65                          # dense inside blocks
    p_inter = 0.03                          # sparse between blocks
    seed = 42

    n_blocks = len(block_sizes)

    probs = [
        [p_intra if i == j else p_inter for j in range(n_blocks)]
        for i in range(n_blocks)
    ]

    G = nx.stochastic_block_model(block_sizes, probs, seed=seed)

    # -------------------------------
    # Color nodes by block
    # -------------------------------
    node_colors = []
    for b, size in enumerate(block_sizes):
        node_colors.extend([b] * size)

    # -------------------------------
    # Create "block-looking" layout
    # -------------------------------
    # Put block centers on a circle
    centers = []
    for b in range(n_blocks):
        ang = 2 * math.pi * b / n_blocks
        centers.append((2.0 * math.cos(ang), 2.0 * math.sin(ang)))

    pos = block_positions(block_sizes, centers, radius=0.55, seed=seed)

    # -------------------------------
    # Draw
    # -------------------------------
    plt.figure(figsize=(10, 7))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        node_size=160,
        edge_color="gray",
        width=0.8,
        alpha=0.85,
    )
    plt.title("SBM Graph (Blocks Shown Explicitly)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
