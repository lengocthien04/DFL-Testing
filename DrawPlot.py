import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

FILES = {
    "D-Cliques": {
        "path": "dclique_output.txt",
        "color": "tab:blue",
        "linestyle": "-"
    },
    "D-Clique Aggregator": {
        "path": "my_agg_output.txt",
        "color": "tab:orange",
        "linestyle": "--"
    },
    "Refined Topology": {
        "path": "refined_output.txt",
        "color": "tab:green",
        "linestyle": "-"
    },
    "Random Topology": {
        "path": "random_output.txt",
        "color": "tab:red",
        "linestyle": ":"
    },
    "Fully Connected": {
        "path": "fully_output.txt",
        "color": "black",
        "linestyle": "-"
    },
}

OUTPUT_PNG = "benchmark_compare.png"


# -----------------------------
# Helper
# -----------------------------

def load_output(path):
    """
    Load output.txt file.
    Returns:
        epoch (N,)
        mean  (N,)
        min   (N,)
        max   (N,)
    """
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


# -----------------------------
# Main plotting
# -----------------------------

plt.figure(figsize=(10, 6))

for name, cfg in FILES.items():
    epoch, mean, min_acc, max_acc = load_output(cfg["path"])
    plt.plot(
        epoch,
        mean * 100.0,
        label=name,
        color=cfg["color"],
        linestyle=cfg["linestyle"],
        linewidth=2
    )

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mean Accuracy (%)", fontsize=12)
plt.title("100-Node D-SGD Benchmark Comparison (MNIST)", fontsize=14)

plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.show()

print(f"Saved comparison figure to: {OUTPUT_PNG}")
