#!/bin/bash
set -e

# The directory where the results will be archived.
DEST_DIR="/mnt/kostas-graid/datasets/vlongle/rl_sensors"

echo "Archiving results to $DEST_DIR"
mkdir -p "$DEST_DIR"

# List of directories to move.
DIRS_TO_MOVE=(
    "wandb"
    "models"
    "eval_logs"
    "tb_logs"
    "best_models"
    "slurm_outs"
)

# Move directories that exist at the project root
for dir in "${DIRS_TO_MOVE[@]}"; do
    if [ -d "$dir" ]; then
        echo "Moving $dir to $DEST_DIR/"
        mv "$dir" "$DEST_DIR/"
    else
        echo "Directory $dir not found, skipping."
    fi
done

# Handle plot/figures separately
if [ -d "plot/figures" ]; then
    echo "Moving plot/figures to $DEST_DIR/"
    mkdir -p "$DEST_DIR/plot"
    mv "plot/figures" "$DEST_DIR/plot/"
else
    echo "Directory plot/figures not found, skipping."
fi

echo "Archiving complete."
echo "View results at: $DEST_DIR" 