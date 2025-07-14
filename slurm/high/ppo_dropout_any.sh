#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base log directory
BASE_LOGDIR=~/logdir/baselines/ppo_dropout_any

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

# Calculate the number of configurations and seeds
NUM_CONFIGS=${#CONFIGS[@]}
NUM_TOTAL_JOBS=$(($NUM_CONFIGS * $NUM_SEEDS))

echo "Submitting $NUM_TOTAL_JOBS jobs..."

# Loop through all config-seed combinations and submit jobs individually
for ((config_idx=0; config_idx<$NUM_CONFIGS; config_idx++)); do
  CONFIG="${CONFIGS[$config_idx]}"

  for ((seed_idx=0; seed_idx<$NUM_SEEDS; seed_idx++)); do
    SEED="${SEEDS[$seed_idx]}"

    # Generate the logdir for each config-seed pair
    LOGDIR="${BASE_LOGDIR}/${CONFIG}_${SEED}"

    # Create a temporary helper script for each job
    HELPER_SCRIPT="ppo_dropout_any_job_${config_idx}_${seed_idx}.sh"

    cat << EOF > $HELPER_SCRIPT
#!/usr/bin/env bash
## dj-partition settings
#SBATCH --job-name=ppo_dropout_any
#SBATCH --output=outputs/ppo_dropout_any_%A_%a.out
#SBATCH --error=outputs/ppo_dropout_any_%A_%a.err
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
BASE_LOGDIR=$BASE_LOGDIR
LOGDIR="\$BASE_LOGDIR/\$CONFIG_\$SEED"

echo "Running PPO Dropout Any baseline with config \$CONFIG and seed \$SEED, logging to \$LOGDIR"

timeout 24h python3 -u baselines/ppo_dropout_any/train.py \\
  --configs "\$CONFIG" \\
  --seed "\$SEED"

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

echo "All PPO Dropout Any jobs submitted." 