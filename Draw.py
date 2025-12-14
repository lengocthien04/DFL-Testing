import numpy as np
import matplotlib.pyplot as plt

# Load D-Cliques output
data = np.loadtxt("dclique_output.txt", delimiter=",")

epochs = data[:, 0]
mean_acc = data[:, 1] * 100.0  # convert to %

# ---- Plot (match aggregator style) ----
plt.figure(figsize=(8, 5))   # same size
plt.plot(epochs, mean_acc, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Mean accuracy (%)")
plt.title("D-Cliques - MNIST")

plt.grid(True)
plt.tight_layout()

plt.savefig("dcliques_accuracy_same_style.png", dpi=200)
plt.show()

print("Saved: dcliques_accuracy_same_style.png")
