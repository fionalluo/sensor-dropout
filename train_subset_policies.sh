#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base output directory
BASE_OUTPUT_DIR=./policies

# List of configs to run
CONFIGS=(
  "gymnasium_tigerkeydoor"
  # "gymnasium_tigerkeydoorlarge"
  # "gymnasium_maze"
  # "gymnasium_blindpick"
)

NUM_SEEDS=1
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

export MUJOCO_GL=egl;

# Iterate through configs
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG}_${SEED}"

    echo "Training subset policies for config ${CONFIG} with seed ${SEED}"
    echo "Output directory: ${OUTPUT_DIR}"

    timeout 8h python3 -u subset_policies/train_subset_policies.py \
      --configs ${CONFIG} \
      --seed "$SEED" \
      --output_dir "$OUTPUT_DIR" \
      --cuda \
      --debug

    if [ $? -eq 124 ]; then
      echo "Command timed out for config ${CONFIG} and seed ${SEED}."
    else
      echo "Command completed for config ${CONFIG} and seed ${SEED}."
    fi

    echo "-----------------------"
  done
done

echo "All subset policy training complete." 