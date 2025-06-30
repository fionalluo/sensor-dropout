#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base log directory
BASE_LOGDIR=~/logdir/baselines/ppo_distill

# Policy types for distillation
EXPERT_POLICY_TYPE="ppo"      # Type of expert policies to distill from
STUDENT_POLICY_TYPE="ppo" # Type of student agent to distill into

# Automatically construct base policy directory from expert policy type
BASE_POLICY_DIR="./policies/${EXPERT_POLICY_TYPE}"

# List of configs to run
CONFIGS=(
  "gymnasium_tigerdoorkey"
  # "gymnasium_tigerdoorkeylarge"
  # "gymnasium_maze"
  # "gymnasium_blindpick"
)

NUM_SEEDS=5
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

export MUJOCO_GL=egl;

echo "Distillation Configuration:"
echo "  Expert policies: ${EXPERT_POLICY_TYPE} (from ${BASE_POLICY_DIR})"
echo "  Student agent: ${STUDENT_POLICY_TYPE}"

# Iterate through configs
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
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

    echo "Running PPO Distill baseline with config ${CONFIG} and seed ${SEED}"
    echo "Expert policy directory: ${EXPERT_POLICY_DIR}"
    echo "Student policy type: ${STUDENT_POLICY_TYPE}"
    echo "Logging to: ${LOGDIR}"

    timeout 8h python3 -u baselines/ppo_distill/train.py \
      --configs ${CONFIG} \
      --expert_policy_dir "$EXPERT_POLICY_DIR" \
      --student_policy_type "$STUDENT_POLICY_TYPE" \
      --seed "$SEED" \
      --cuda \
      --track

    if [ $? -eq 124 ]; then
      echo "Command timed out for config ${CONFIG} and seed ${SEED}."
    else
      echo "Command completed for config ${CONFIG} and seed ${SEED}."
    fi

    echo "-----------------------"
  done
done

echo "All PPO Distill tasks complete." 