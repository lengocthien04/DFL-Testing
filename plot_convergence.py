import csv
import os
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 6))

    for label, path in logs.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing {path}")
            continue

        x, y = load_log_csv(path)
        plt.plot(x, y, linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


if __name__ == "__main__":
    # CIFAR-10 with dmax=4 / clique_size=4
    cifar_4_logs = {
        "Fully-connected": "outputs/cifar10_fully_n24_output.txt",
        "Random (dmax=4)": "outputs/cifar10_random_n24_dmax4_output.txt",
        "D-Clique (c=4)": "outputs/cifar10_dclique_n24_c4_output.txt",
        "Hierarchy": "outputs/cifar-4_metrics.csv",
    }
    plot_convergence(
        cifar_4_logs,
        "CIFAR-10 Convergence (n=24, dmax=4, clique_size=4)",
        "outputs/cifar10_convergence_dmax4.png",
    )

    # CIFAR-10 with dmax=3 / clique_size=3
    cifar_3_logs = {
        "Fully-connected": "outputs/cifar10_fully_n24_output.txt",
        "Random (dmax=3)": "outputs/cifar10_random_n24_dmax3_output.txt",
        "D-Clique (c=3)": "outputs/cifar10_dclique_n24_c3_output.txt",
        "Hierarchy": "outputs/cifar-3_metrics.csv",
    }
    plot_convergence(
        cifar_3_logs,
        "CIFAR-10 Convergence (n=24, dmax=3, clique_size=3)",
        "outputs/cifar10_convergence_dmax3.png",
    )

    # CIFAR-10 with dmax=6 / clique_size=6
    cifar_6_logs = {
        "Fully-connected": "outputs/cifar10_fully_n24_output.txt",
        "Random (dmax=6)": "outputs/cifar10_random_n24_dmax6_output.txt",
        "D-Clique (c=6)": "outputs/cifar10_dclique_n24_c6_output.txt",
        "Hierarchy": "outputs/cifar-6_metrics.csv",
    }
    plot_convergence(
        cifar_6_logs,
        "CIFAR-10 Convergence (n=24, dmax=6, clique_size=6)",
        "outputs/cifar10_convergence_dmax6.png",
    )

    # MNIST with dmax=4 / clique_size=4
    mnist_4_logs = {
        "Fully-connected": "outputs/mnist_fully_n24_output.txt",
        "Random (dmax=4)": "outputs/mnist_random_n24_dmax4_output.txt",
        "D-Clique (c=4)": "outputs/mnist_dclique_n24_c4_output.txt",
        "Hierarchy": "outputs/mnist-4_metrics.csv",
    }
    plot_convergence(
        mnist_4_logs,
        "MNIST Convergence (n=24, dmax=4, clique_size=4)",
        "outputs/mnist_convergence_dmax4.png",
    )

    # MNIST with dmax=3 / clique_size=3
    mnist_3_logs = {
        "Fully-connected": "outputs/mnist_fully_n24_output.txt",
        "Random (dmax=3)": "outputs/mnist_random_n24_dmax3_output.txt",
        "D-Clique (c=3)": "outputs/mnist_dclique_n24_c3_output.txt",
        "Hierarchy": "outputs/mnist-3_metrics.csv",
    }
    plot_convergence(
        mnist_3_logs,
        "MNIST Convergence (n=24, dmax=3, clique_size=3)",
        "outputs/mnist_convergence_dmax3.png",
    )

    # MNIST with dmax=6 / clique_size=6
    mnist_6_logs = {
        "Fully-connected": "outputs/mnist_fully_n24_output.txt",
        "Random (dmax=6)": "outputs/mnist_random_n24_dmax6_output.txt",
        "D-Clique (c=6)": "outputs/mnist_dclique_n24_c6_output.txt",
        "Hierarchy": "outputs/mnist-6_metrics.csv",
    }
    plot_convergence(
        mnist_6_logs,
        "MNIST Convergence (n=24, dmax=6, clique_size=6)",
        "outputs/mnist_convergence_dmax6.png",
    )

    print("\nAll convergence plots generated!")
