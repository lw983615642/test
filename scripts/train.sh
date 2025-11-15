#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --input_dir /path/to/dataset/inputs \
  --target_dir /path/to/dataset/targets \
  --reflection_dir /path/to/dataset/reflections \
  --output_dir runs/polar_reflection \
  --epochs 300 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --amp
