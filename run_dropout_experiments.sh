#!/bin/bash

# Fully connected with dropout experiments
# 24 nodes, 3 nodes drop off randomly each epoch
# Alpha values: 0.2, 0.4, 0.6

echo "Starting fully connected dropout experiments (3 nodes drop per epoch)..."

python3 run_cifar10.py --method fully_dropout --n 24 --dropout_count 3 --lr 0.01 --batch 64 --alpha 0.2 --epochs 100 &
python3 run_cifar10.py --method fully_dropout --n 24 --dropout_count 3 --lr 0.01 --batch 64 --alpha 0.6 --epochs 100 &

wait
echo "All dropout experiments completed!"
