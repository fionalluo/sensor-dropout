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

echo "Training ${POLICY_TYPE} subset policies with SB3"
echo "Output directory: ${BASE_OUTPUT_DIR}"

# Calculate the number of configurations and seeds
NUM_CONFIGS=${#CONFIGS[@]}
NUM_TOTAL_JOBS=$(($NUM_CONFIGS * $NUM_SEEDS))

echo "Submitting $NUM_TOTAL_JOBS jobs..."

# Loop through all config-seed combinations and submit jobs individually
for ((config_idx=0; config_idx<$NUM_CONFIGS; config_idx++)); do
  CONFIG="${CONFIGS[$config_idx]}"

  for ((seed_idx=0; seed_idx<$NUM_SEEDS; seed_idx++)); do
    SEED="${SEEDS[$seed_idx]}"

    # Extract clean task name (remove any prefix_ and any -vN suffix)
    TASK_NAME=$(echo "$CONFIG" | sed 's/^[^_]*_//' | sed 's/-v[0-9]*$//')
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_NAME}"

    # Create a temporary helper script for each job
    HELPER_SCRIPT="train_subset_policies_job_${config_idx}_${seed_idx}.sh"

    cat << EOF > $HELPER_SCRIPT
#!/usr/bin/env bash
## dj-partition settings
#SBATCH --job-name=train_subset_policies
#SBATCH --output=outputs/train_subset_policies_%A_%a.out
#SBATCH --error=outputs/train_subset_policies_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --mem=40G
#SBATCH --exclude=kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,dj-2080ti-0.grasp.maas
##SBATCH --nodelist=dj-l40-0.grasp.maas

export MUJOCO_GL=egl;

# Constants passed from parent script
CONFIG=$CONFIG
SEED=$SEED
POLICY_TYPE=$POLICY_TYPE
OUTPUT_DIR=$OUTPUT_DIR

echo "Training subset policies for config \$CONFIG with seed \$SEED"
echo "Output directory: \$OUTPUT_DIR"

timeout 12h python3 -u subset_policies/train_subset_policies.py \\
  --configs "\$CONFIG" \\
  --seed "\$SEED" \\
  --output_dir "\$OUTPUT_DIR" \\
  --policy_type "\$POLICY_TYPE" \\
  --cuda \\
  --debug

if [ \$? -eq 124 ]; then
  echo "Command timed out for config \$CONFIG and seed \$SEED."
else
  echo "Command completed for config \$CONFIG and seed \$SEED."
fi

echo "-----------------------"
EOF

    # Make the helper script executable
    chmod +x $HELPER_SCRIPT

    # Submit the job
    sbatch $HELPER_SCRIPT

    # Clean up
    rm $HELPER_SCRIPT
  done
done

echo "All ${POLICY_TYPE} subset policy training jobs submitted." 