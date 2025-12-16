#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.cifar10 import load_cifar10, make_cifar10_loaders
from models.cifar_models import GNLeNet
from topology.fully import fully_connected
from topology.topo_random import build as build_random
from topology.dclique import build as build_dclique
from topology.refined_fw import build as build_refined
from training.dsgd import run_steps_plain_dsgd
from training.dcliques_alg import run_steps_dcliques_two_stage
from training.mydclique_alg import build_agg_selector, run_steps_mydclique
from training.evaluation import evaluate_models
from utils.communication import communication_stats_from_adj
from utils.logging import init_log, log_epoch, write_reach_thresholds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fully","random","dclique","refined","mydclique"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dmax", type=int, default=10)
    ap.add_argument("--clique_size", type=int, default=10)
    ap.add_argument("--swaps", type=int, default=5000)
    ap.add_argument("--fw_iters", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_cifar10("./data")
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    loaders, node_idx = make_cifar10_loaders(train, args.n, args.alpha, 32, args.seed)
    labels = np.array(train.targets, dtype=np.int64)

    if args.method == "fully":
        A, W = fully_connected(args.n, device)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, W, device, steps)
        out, fig = "cifar10_fully_output.txt", "cifar10_fully_accuracy.png"

    elif args.method == "random":
        A, W = build_random(args.n, args.dmax, args.seed, device)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, W, device, steps)
        out, fig = "cifar10_random_output.txt", "cifar10_random_accuracy.png"

    elif args.method == "dclique":
        cliques, A, Wc, Wp = build_dclique(labels, node_idx, 10, args.clique_size, args.swaps, args.seed, device)
        step_runner = lambda models, optims, steps: run_steps_dcliques_two_stage(models, optims, loaders, Wc, Wp, device, steps)
        out, fig = "cifar10_dclique_output.txt", "cifar10_dclique_accuracy.png"

    elif args.method == "mydclique":
        cliques, A, Wc, Wp = build_dclique(labels, node_idx, 10, args.clique_size, args.swaps, args.seed, device)
        agg_nodes = build_agg_selector(cliques, mode="first")
        Winter = Wp[agg_nodes][:, agg_nodes]
        step_runner = lambda models, optims, steps: run_steps_mydclique(models, optims, loaders, cliques, agg_nodes, Winter, device, steps)
        out, fig = "cifar10_mydclique_output.txt", "cifar10_mydclique_accuracy.png"

    else:
        A, W = build_refined(labels, node_idx, 10, args.lam, args.fw_iters, device)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, W, device, steps)
        out, fig = "cifar10_refined_output.txt", "cifar10_refined_accuracy.png"

    comm = communication_stats_from_adj(A)
    steps_per_epoch = max(1, math.ceil(len(train) / (args.n * args.batch)))
    header = dict(comm)
    header["steps_per_epoch"] = float(steps_per_epoch)
    header["total_msgs_per_epoch"] = header["total_msgs_per_step"] * steps_per_epoch

    models = [GNLeNet().to(device) for _ in range(args.n)]
    optims = [torch.optim.SGD(m.parameters(), lr=args.lr) for m in models]

    targets = [0.40, 0.50]
    reached = {t: None for t in targets}

    mean_curve, med_curve, min_curve, max_curve = [], [], [], []
    x = np.arange(1, args.epochs + 1)

    with open(out, "w", encoding="utf-8") as f:
        init_log(f, header)

        for epoch in range(1, args.epochs + 1):
            step_runner(models, optims, steps_per_epoch)
            stats = evaluate_models(models, test_loader, device)
            log_epoch(f, epoch, stats)

            for t in targets:
                if reached[t] is None and stats["mean"] >= t:
                    reached[t] = epoch

            mean_curve.append(stats["mean"] * 100.0)
            med_curve.append(stats["median"] * 100.0)
            min_curve.append(stats["min"] * 100.0)
            max_curve.append(stats["max"] * 100.0)

            print(f"Epoch {epoch:03d}/{args.epochs} | mean {stats['mean']*100:.2f}% | std {stats['std']*100:.2f}%")

        write_reach_thresholds(f, reached)

    plt.figure()
    plt.plot(x, mean_curve, label="Mean")
    plt.plot(x, med_curve, label="Median")
    plt.fill_between(x, min_curve, max_curve, alpha=0.2, label="Min–Max band")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"CIFAR-10 - {args.method}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig, dpi=200)
    print("Saved:", out, fig)

if __name__ == "__main__":
    main()
