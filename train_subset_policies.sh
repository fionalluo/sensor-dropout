#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Set policy type (only ppo supported for SB3)
POLICY_TYPE="ppo"

# Base output directory - now includes policy type
BASE_OUTPUT_DIR=./policies/${POLICY_TYPE}

# List of configs to run
CONFIGS=(
  # "gymnasium_tigerdoorkey"
  # "gymnasium_tigerdoorkeylarge"
  # "gymnasium_maze"
  # "gymnasium_maze11"
  "gymnasium_blindpick"
)

NUM_SEEDS=1
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

export MUJOCO_GL=egl;

echo "Training ${POLICY_TYPE} subset policies with SB3"
echo "Output directory: ${BASE_OUTPUT_DIR}"

# Iterate through configs
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # Extract clean task name (remove any prefix_ and any -vN suffix)
    TASK_NAME=$(echo "$CONFIG" | sed 's/^[^_]*_//' | sed 's/-v[0-9]*$//')
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_NAME}"

    echo "Training subset policies for config ${CONFIG} with seed ${SEED}"
    echo "Output directory: ${OUTPUT_DIR}"

    timeout 8h python3 -u subset_policies/train_subset_policies.py \
      --configs ${CONFIG} \
      --seed "$SEED" \
      --output_dir "$OUTPUT_DIR" \
      --policy_type "$POLICY_TYPE" \
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

echo "All ${POLICY_TYPE} subset policy training complete." 