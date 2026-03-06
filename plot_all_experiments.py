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
            if first.startswith("#") or first == "epoch":
                continue
            epoch = int(first)
            mean = float(row[1]) * 100.0
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
    os.makedirs("outputs/plots", exist_ok=True)
    
    # For each alpha value
    for alpha in [0.2, 0.4, 0.6]:
        alpha_str = str(alpha)
        alpha_folder = f"outputs/{alpha_str}"
        csv_suffix = alpha_str.replace('0.', '0')  # 0.2 -> 02, 0.4 -> 04, 0.6 -> 06
        
        # 24 nodes, clique size 3 (dmax 3)
        logs_24_c3 = {
            "Fully-connected": f"{alpha_folder}/cifar10_fully_n24_alpha{alpha_str}_output.txt",
            "Random (dmax=3)": f"{alpha_folder}/cifar10_random_n24_dmax3_alpha{alpha_str}_output.txt",
            "D-Clique (c=3)": f"{alpha_folder}/cifar10_dclique_n24_c3_alpha{alpha_str}_output.txt",
            "Hierarchy": f"{alpha_folder}/cifar10_hierarchy_n24_c3_alpha{alpha_str}_output.txt",
            "Hierarchy (fully connected aggregator)": f"{alpha_folder}/cifar10_hierarchy_simple_n24_c3_alpha{alpha_str}_output.txt",
            # "Simulation version": f"{alpha_folder}/c3-d{csv_suffix}-epoch_metrics.csv",
        }
        plot_convergence(
            logs_24_c3,
            f"CIFAR-10 Convergence (n=24, c=3, alpha={alpha})",
            f"outputs/plots/cifar10_n24_c3_alpha{alpha_str}_simulation_version.png"
        )
        
        # 24 nodes, clique size 6 (dmax 6)
        logs_24_c6 = {
            "Fully-connected": f"{alpha_folder}/cifar10_fully_n24_alpha{alpha_str}_output.txt",
            "Random (dmax=6)": f"{alpha_folder}/cifar10_random_n24_dmax6_alpha{alpha_str}_output.txt",
            "D-Clique (c=6)": f"{alpha_folder}/cifar10_dclique_n24_c6_alpha{alpha_str}_output.txt",
            "Hierarchy": f"{alpha_folder}/cifar10_hierarchy_n24_c6_alpha{alpha_str}_output.txt",
            "Hierarchy (fully connected aggregator)": f"{alpha_folder}/cifar10_hierarchy_simple_n24_c6_alpha{alpha_str}_output.txt",
            # "Simulation version": f"{alpha_folder}/c6-d{csv_suffix}-epoch_metrics.csv",
        }
        plot_convergence(
            logs_24_c6,
            f"CIFAR-10 Convergence (n=24, c=6, alpha={alpha})",
            f"outputs/plots/cifar10_n24_c6_alpha{alpha_str}_simulation_version.png"
        )
        
        # 100 nodes, clique size 5
        logs_100_c5 = {
            "Hierarchy": f"{alpha_folder}/cifar10_hierarchy_n100_c5_alpha{alpha_str}_output.txt",
            "Hierarchy (fully connected aggregator)": f"{alpha_folder}/cifar10_hierarchy_simple_n100_c5_alpha{alpha_str}_output.txt",
        }
        plot_convergence(
            logs_100_c5,
            f"CIFAR-10 Convergence (n=100, c=5, alpha={alpha})",
            f"outputs/plots/cifar10_n100_c5_alpha{alpha_str}_simulation_version.png"
        )
    
    print("\nAll plots generated in outputs/plots/")
