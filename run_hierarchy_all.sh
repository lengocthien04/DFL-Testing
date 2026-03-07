#!/bin/bash

# Hierarchy experiments for CIFAR-10
# 24 nodes with clique size 3 and 6
# 100 nodes with clique size 5
# Alpha values: 0.2, 0.4, 0.6

echo "Starting hierarchy experiments..."

# 24 nodes, clique size 3
python3 run_cifar10.py --method hierarchy --n 24 --clique_size 3 --lr 0.01 --batch 64 --alpha 0.2 --epochs 100 --swaps 20000 &
python3 run_cifar10.py --method hierarchy --n 24 --clique_size 3 --lr 0.01 --batch 64 --alpha 0.4 --epochs 100 --swaps 20000 &
python3 run_cifar10.py --method hierarchy --n 24 --clique_size 3 --lr 0.01 --batch 64 --alpha 0.6 --epochs 100 --swaps 20000 &

# 24 nodes, clique size 6
python3 run_cifar10.py --method hierarchy --n 24 --clique_size 6 --lr 0.01 --batch 64 --alpha 0.2 --epochs 100 --swaps 20000 &
python3 run_cifar10.py --method hierarchy --n 24 --clique_size 6 --lr 0.01 --batch 64 --alpha 0.4 --epochs 100 --swaps 20000 &
python3 run_cifar10.py --method hierarchy --n 24 --clique_size 6 --lr 0.01 --batch 64 --alpha 0.6 --epochs 100 --swaps 20000 &

# 100 nodes, clique size 5
python3 run_cifar10.py --method hierarchy --n 100 --clique_size 5 --lr 0.01 --batch 64 --alpha 0.2 --epochs 100 --swaps 20000 &
python3 run_cifar10.py --method hierarchy --n 100 --clique_size 5 --lr 0.01 --batch 64 --alpha 0.4 --epochs 100 --swaps 20000 &
python3 run_cifar10.py --method hierarchy --n 100 --clique_size 5 --lr 0.01 --batch 64 --alpha 0.6 --epochs 100 --swaps 20000 &

wait
echo "All hierarchy experiments completed!"
