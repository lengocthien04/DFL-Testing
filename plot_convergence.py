import csv
import os
import matplotlib.pyplot as plt

METHODS = {
    "fully": "Fully-connected",
    "random": "Random",
    "dclique": "D-Clique",
    "mydclique": "SBDFL",
    "refined": "Refined",
}

REFERENCE_LINES = []

MNIST_LOGS = {
    "fully": "mnist_fully_output.txt",
    "random": "mnist_random_output.txt",
    "dclique": "mnist_dclique_output.txt",
    "mydclique": "mnist_mydclique_output.txt",
    "refined": "mnist_refined_output.txt",
}

CIFAR_LOGS = {
    "fully": "cifar10_fully_output.txt",
    "random": "cifar10_random_output.txt",
    "dclique": "cifar10_dclique_output.txt",
    "mydclique": "cifar10_mydclique_output.txt",
    "refined": "cifar10_refined_output.txt",
}


def load_log_csv(path):
    epochs, means = [], []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            first = row[0].strip()

            # skip comments and headers
            if first.startswith("#"):
                continue
            if first == "epoch":
                continue

            epoch = int(first)
            mean = float(row[1]) * 100.0  # convert to %

            epochs.append(epoch)
            means.append(mean)

    return epochs, means



def plot_convergence(logs, title, outfile):
    plt.figure(figsize=(8, 6))

    for key, label in METHODS.items():
        if key not in logs:
            continue

        path = logs[key]
        if not os.path.exists(path):
            print(f"[WARN] Missing {path}")
            continue

        x, y = load_log_csv(path)
        plt.plot(x, y, linewidth=2, label=label)

    # reference accuracy lines
    for acc in REFERENCE_LINES:
        plt.axhline(acc, linestyle="--", linewidth=1, alpha=0.4)
        plt.text(
            0.99,
            acc,
            f"{acc}%",
            fontsize=9,
            alpha=0.7,
            ha="right",
            va="bottom",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_convergence(
        MNIST_LOGS,
        "MNIST Convergence Speed Comparison",
        "mnist_convergence.png",
    )

    plot_convergence(
        CIFAR_LOGS,
        "CIFAR-10 Convergence Speed Comparison",
        "cifar_convergence.png",
    )

    print("Saved mnist_convergence.png and cifar_convergence.png")
