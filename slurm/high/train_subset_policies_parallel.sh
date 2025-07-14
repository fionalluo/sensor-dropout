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

echo "Training ${POLICY_TYPE} subset policies with SB3 (PARALLEL)"
echo "Output directory: ${BASE_OUTPUT_DIR}"

# Calculate total jobs across all configs and seeds
total_jobs=0
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # Get number of subsets for this config
    NUM_SUBSETS=$(python3 -u subset_policies/get_num_subsets.py --configs "$CONFIG")
    total_jobs=$((total_jobs + NUM_SUBSETS))
  done
done

echo "Submitting $total_jobs subset training jobs (parallel execution)..."

# Loop through all config-seed combinations and submit jobs for each subset
for ((config_idx=0; config_idx<${#CONFIGS[@]}; config_idx++)); do
  CONFIG="${CONFIGS[$config_idx]}"

  for ((seed_idx=0; seed_idx<$NUM_SEEDS; seed_idx++)); do
    SEED="${SEEDS[$seed_idx]}"

    # Extract clean task name (remove any prefix_ and any -vN suffix)
    TASK_NAME=$(echo "$CONFIG" | sed 's/^[^_]*_//' | sed 's/-v[0-9]*$//')
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_NAME}"

    # Get number of subsets for this config
    NUM_SUBSETS=$(python3 -u subset_policies/get_num_subsets.py --configs "$CONFIG")
    
    echo "Config: $CONFIG, Seed: $SEED, Subsets: $NUM_SUBSETS"

    # Submit separate job for each subset
    for ((subset_idx=1; subset_idx<=$NUM_SUBSETS; subset_idx++)); do
      SUBSET_NAME="env${subset_idx}"
      
      # Create a temporary helper script for each subset job
      HELPER_SCRIPT="train_subset_${CONFIG}_${SEED}_${SUBSET_NAME}_${config_idx}_${seed_idx}_${subset_idx}.sh"

      cat << EOF > $HELPER_SCRIPT
#!/usr/bin/env bash
## dj-partition settings
#SBATCH --job-name=train_subset_${SUBSET_NAME}
#SBATCH --output=outputs/train_subset_${SUBSET_NAME}_%A_%a.out
#SBATCH --error=outputs/train_subset_${SUBSET_NAME}_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --mem=40G
#SBATCH --exclude=kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,dj-2080ti-0.grasp.maas
##SBATCH --nodelist=dj-l40-0.grasp.maas  # Uncomment for L40 targeting

export MUJOCO_GL=egl;

# Constants passed from parent script
CONFIG=$CONFIG
SEED=$SEED
POLICY_TYPE=$POLICY_TYPE
OUTPUT_DIR=$OUTPUT_DIR
SUBSET_NAME=$SUBSET_NAME

echo "Training subset policy for config \$CONFIG, seed \$SEED, subset \$SUBSET_NAME"
echo "Output directory: \$OUTPUT_DIR"

timeout 24h python3 -u subset_policies/train_single_subset_policy.py \\
  --configs "\$CONFIG" \\
  --seed "\$SEED" \\
  --output_dir "\$OUTPUT_DIR" \\
  --policy_type "\$POLICY_TYPE" \\
  --subset_name "\$SUBSET_NAME" \\
  --cuda \\
  --debug

if [ \$? -eq 124 ]; then
  echo "Command timed out for config \$CONFIG, seed \$SEED, subset \$SUBSET_NAME."
else
  echo "Command completed for config \$CONFIG, seed \$SEED, subset \$SUBSET_NAME."
fi

echo "-----------------------"
EOF

      # Make the helper script executable
      chmod +x $HELPER_SCRIPT

      # Submit the job
      echo "  Submitting job for subset $SUBSET_NAME..."
      sbatch $HELPER_SCRIPT

      # Clean up
      rm $HELPER_SCRIPT
    done
  done
done

echo "All ${POLICY_TYPE} subset policy training jobs submitted (parallel execution)."
echo "Each subset (env1, env2, env3, ...) is now training in parallel on separate cluster nodes." 