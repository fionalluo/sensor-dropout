#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base log directory
BASE_LOGDIR=~/logdir/baselines/simple_imitation

# Policy types for distillation
EXPERT_POLICY_TYPE="ppo"      # Type of expert policies to distill from

# Automatically construct base policy directory from expert policy type
BASE_POLICY_DIR="./policies/${EXPERT_POLICY_TYPE}"

# List of configs to run
CONFIGS=(
  "gymnasium_tigerdoorkey"
  # "gymnasium_tigerdoorkeylarge"
  # "gymnasium_maze"
  # "gymnasium_maze11"
  # "gymnasium_blindpick"
)

NUM_SEEDS=1
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

echo "Simple Imitation Learning Configuration:"
echo "  Expert policies: ${EXPERT_POLICY_TYPE} (from ${BASE_POLICY_DIR})"
echo "  Method: Direct PyTorch imitation learning"

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
    EXPERT_POLICY_DIR="${BASE_POLICY_DIR}/${TASK_NAME}"
    LOGDIR="${BASE_LOGDIR}/${CONFIG}_${SEED}"

    # Check if expert policies exist
    if [ ! -d "$EXPERT_POLICY_DIR" ]; then
      echo "Error: Expert policy directory not found: $EXPERT_POLICY_DIR"
      echo "Please run train_subset_policies.sh first to create ${EXPERT_POLICY_TYPE} expert policies for $TASK_NAME."
      echo "Skipping config ${CONFIG} with seed ${SEED}."
      echo "-----------------------"
      continue
    fi

    # Create a temporary helper script for each job
    HELPER_SCRIPT="ppo_distill_job_${config_idx}_${seed_idx}.sh"

    cat << EOF > $HELPER_SCRIPT
#!/usr/bin/env bash
## dj-partition settings
#SBATCH --job-name=ppo_distill
#SBATCH --output=outputs/ppo_distill_%A_%a.out
#SBATCH --error=outputs/ppo_distill_%A_%a.err
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
BASE_LOGDIR=$BASE_LOGDIR
EXPERT_POLICY_DIR=$EXPERT_POLICY_DIR
LOGDIR="\$BASE_LOGDIR/\$CONFIG_\$SEED"

echo "Running Simple Imitation Learning with config \$CONFIG and seed \$SEED"
echo "Expert policy directory: \$EXPERT_POLICY_DIR"
echo "Logging to: \$LOGDIR"

# Use GPU if available, fallback to CPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  DEVICE="cuda"
  echo "Using GPU for training"
else
  DEVICE="cpu"
  echo "Using CPU for training"
fi

timeout 12h python3 -u baselines/ppo_distill/train.py \\
  --configs "\$CONFIG" \\
  --expert_policy_dir "\$EXPERT_POLICY_DIR" \\
  --seed "\$SEED" \\
  --device "\$DEVICE" \\
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

echo "All Simple Imitation Learning jobs submitted." 