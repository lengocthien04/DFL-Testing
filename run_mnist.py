#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, math, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.mnist import load_mnist, make_mnist_loaders, make_mnist_test_loaders
from models.mnist_models import LogisticMNIST
from topology.fully import fully_connected
from topology.topo_random import build as build_random
from topology.dclique import build as build_dclique
from topology.refined_fw import build as build_refined
from training.dsgd import run_steps_plain_dsgd
from training.dcliques_alg import run_steps_dcliques_two_stage
from training.mydclique_alg import build_agg_selector, run_steps_mydclique
from training.hierarchical_mydclique import build_hierarchy_runtime, run_steps_hierarchical_mydclique
from training.hierarchical_simple import run_steps_hierarchical_simple
from training.evaluation import evaluate_models
from utils.communication import communication_stats_from_adj
from utils.logging import init_log, log_epoch, write_reach_thresholds
from topology.dclique import build_clique_neighbors
from utils.hierarchy import build_state_cliques, load_hierarchy_levels, load_scope_instances
from utils.dynamic_hierarchy import generate_two_state_hierarchy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fully","random","dclique","refined","mydclique","hierarchy","hierarchy_simple"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=126)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dmax", type=int, default=10)
    ap.add_argument("--clique_size", type=int, default=10)
    ap.add_argument("--swaps", type=int, default=5000)
    ap.add_argument("--fw_iters", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--hierarchy_config", type=str, default="config/hierarchy_config.json")
    ap.add_argument("--nodes_map", type=str, default="config/nodes_map.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_mnist("./data")
    test_loaders = make_mnist_test_loaders(test, args.n, args.alpha, 256, args.seed)

    loaders, node_idx = make_mnist_loaders(train, args.n, args.alpha, 32, args.seed)
    labels = np.array(train.targets, dtype=np.int64)
    n_classes = len(np.unique(labels))
    if args.method == "fully":
        A, W = fully_connected(args.n, device)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, W, device, steps)
        out, fig = f"outputs/mnist_fully_n{args.n}_alpha{args.alpha}_output.txt", f"outputs/mnist_fully_n{args.n}_alpha{args.alpha}_accuracy.png"

    elif args.method == "random":
        A, W = build_random(args.n, args.dmax, args.seed, device)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, W, device, steps)
        out, fig = f"outputs/mnist_random_n{args.n}_dmax{args.dmax}_alpha{args.alpha}_output.txt", f"outputs/mnist_random_n{args.n}_dmax{args.dmax}_alpha{args.alpha}_accuracy.png"

    elif args.method == "dclique":
        cliques, A, Wc, Wp = build_dclique(labels, node_idx, n_classes, args.clique_size, args.swaps, args.seed, device)
        # Use plain DSGD with dclique topology (not clique averaging)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, Wp, device, steps)
        out, fig = f"outputs/mnist_dclique_n{args.n}_c{args.clique_size}_alpha{args.alpha}_output.txt", f"outputs/mnist_dclique_n{args.n}_c{args.clique_size}_alpha{args.alpha}_accuracy.png"

    elif args.method == "mydclique":
        cliques, A, Wc, Wp = build_dclique(
            labels, node_idx, n_classes,
            args.clique_size, args.swaps, args.seed, device
        )
        agg_nodes = build_agg_selector(cliques, mode="first")
        clique_neighbors = build_clique_neighbors(cliques, A)
        print("Clique neighbors:")
        for i, nbs in enumerate(clique_neighbors):
            print(i, "->", nbs)

        step_runner = lambda models, optims, steps: run_steps_mydclique(
            models, optims, loaders,
            cliques, agg_nodes, clique_neighbors,
            device, steps
        )
        out, fig = f"outputs/mnist_mydclique_n{args.n}_c{args.clique_size}_output.txt", f"outputs/mnist_mydclique_n{args.n}_c{args.clique_size}_accuracy.png"

    elif args.method == "hierarchy":
        hier_cfg_path = Path(args.hierarchy_config)
        if not hier_cfg_path.exists():
            raise FileNotFoundError(f"Hierarchy config not found: {hier_cfg_path}")

        levels = load_hierarchy_levels(hier_cfg_path)
        
        # Generate dynamic 2-state hierarchy based on --n
        scope_instances = generate_two_state_hierarchy(args.n)
        
        lowest_index = min(cfg.scope_index for cfg in levels)
        if lowest_index not in scope_instances:
            raise ValueError("Generated hierarchy does not have the lowest level")

        states = list(scope_instances[lowest_index].values())

        clique_assignments, state_to_cliques = build_state_cliques(
            labels=labels,
            node_indices=node_idx,
            states=states,
            clique_size=args.clique_size,
            n_swaps=args.swaps,
            seed=args.seed,
        )

        hierarchy_state = build_hierarchy_runtime(
            level_configs=levels,
            scope_instances=scope_instances,
            clique_assignments=clique_assignments,
            state_to_cliques=state_to_cliques,
        )

        step_runner = lambda models, optims, steps, epoch=1: run_steps_hierarchical_mydclique(
            models, optims, loaders,
            hierarchy_state, device, steps, current_epoch=epoch
        )

        A = np.zeros((args.n, args.n), dtype=np.int32)
        for assignment in clique_assignments:
            nodes = assignment["nodes"]
            for u in nodes:
                for v in nodes:
                    if u != v:
                        A[u, v] = 1

        out, fig = f"outputs/mnist_hierarchy_n{args.n}_c{args.clique_size}_alpha{args.alpha}_output.txt", f"outputs/mnist_hierarchy_n{args.n}_c{args.clique_size}_alpha{args.alpha}_accuracy.png"

    elif args.method == "hierarchy_simple":
        hier_cfg_path = Path(args.hierarchy_config)
        if not hier_cfg_path.exists():
            raise FileNotFoundError(f"Hierarchy config not found: {hier_cfg_path}")

        levels = load_hierarchy_levels(hier_cfg_path)
        
        # Generate dynamic 2-state hierarchy based on --n
        scope_instances = generate_two_state_hierarchy(args.n)
        
        lowest_index = min(cfg.scope_index for cfg in levels)
        if lowest_index not in scope_instances:
            raise ValueError("Generated hierarchy does not have the lowest level")

        states_list = list(scope_instances[lowest_index].values())

        clique_assignments, state_to_cliques = build_state_cliques(
            labels=labels,
            node_indices=node_idx,
            states=states_list,
            clique_size=args.clique_size,
            n_swaps=args.swaps,
            seed=args.seed,
        )

        # Extract cliques and states as simple lists
        cliques = [assignment["nodes"] for assignment in clique_assignments]
        states = [state.nodes for state in states_list]

        step_runner = lambda models, optims, steps, epoch=1: run_steps_hierarchical_simple(
            models, optims, loaders,
            cliques, states, device, steps,
            current_epoch=epoch,
            state_interval=1,
            nation_interval=1,
        )

        # Build adjacency for communication stats
        A = np.zeros((args.n, args.n), dtype=np.int32)
        for clique in cliques:
            for u in clique:
                for v in clique:
                    if u != v:
                        A[u, v] = 1

        out, fig = f"outputs/mnist_hierarchy_simple_n{args.n}_c{args.clique_size}_alpha{args.alpha}_output.txt", f"outputs/mnist_hierarchy_simple_n{args.n}_c{args.clique_size}_alpha{args.alpha}_accuracy.png"

    else:
        A, W = build_refined(labels, node_idx, n_classes, args.lam, args.fw_iters, device)
        step_runner = lambda models, optims, steps: run_steps_plain_dsgd(models, optims, loaders, W, device, steps)
        out, fig = f"outputs/mnist_refined_n{args.n}_alpha{args.alpha}_output.txt", f"outputs/mnist_refined_n{args.n}_alpha{args.alpha}_accuracy.png"

    comm = communication_stats_from_adj(A)
    steps_per_epoch = max(1, math.ceil(len(train) / (args.n * args.batch)))
    header = dict(comm)
    header["steps_per_epoch"] = float(steps_per_epoch)
    header["total_msgs_per_epoch"] = header["total_msgs_per_step"] * steps_per_epoch
    header["n_nodes"] = float(args.n)
    header["learning_rate"] = float(args.lr)
    header["batch_size"] = float(args.batch)
    header["alpha"] = float(args.alpha)
    if args.method in ["random"]:
        header["dmax"] = float(args.dmax)
    if args.method in ["dclique", "mydclique", "hierarchy"]:
        header["clique_size"] = float(args.clique_size)

    models = [LogisticMNIST().to(device) for _ in range(args.n)]
    optims = [torch.optim.SGD(m.parameters(), lr=args.lr) for m in models]

    targets = [0.70, 0.75]
    reached = {t: None for t in targets}

    mean_curve, med_curve, min_curve, max_curve = [], [], [], []
    x = np.arange(1, args.epochs + 1)

    with open(out, "w", encoding="utf-8") as f:
        init_log(f, header)

        for epoch in range(1, args.epochs + 1):
            if args.method == "hierarchy" or args.method == "hierarchy_simple":
                step_runner(models, optims, steps_per_epoch, epoch)
            else:
                step_runner(models, optims, steps_per_epoch)
            stats = evaluate_models(models, test_loaders, device)
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
    plt.title(f"MNIST - {args.method}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig, dpi=200)
    print("Saved:", out, fig)

if __name__ == "__main__":
    main()
