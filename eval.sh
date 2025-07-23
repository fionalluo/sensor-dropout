#!/usr/bin/env bash

# Run the evaluation script through IsaacLab
echo "[INFO] Running evaluation through IsaacLab environment..."
"/workspace/isaaclab/isaaclab.sh" -p "baselines_isaac/evaluation.py" "$@" 