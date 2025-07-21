#!/bin/bash

# User: specify your list of tasks and wandb project here
TASKS=(
  "Isaac-Ant-v0"
  # Add more tasks here as needed
)
WANDB_PROJECT="isaac-test"
WANDB_ENTITY="fionalluo"

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

NUM_SEEDS=1
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

export MUJOCO_GL=egl;

for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "Running RL-Games PPO baseline with task ${TASK}, seed ${SEED}, wandb_project ${WANDB_PROJECT}"

    /workspace/isaaclab/isaaclab.sh -p /workspace/sensor-dropout/baselines_isaac/ppo/train.py \
      --task "$TASK" \
      --wandb-project-name "$WANDB_PROJECT" \
      --wandb-entity "$WANDB_ENTITY" \
      --track \
      --headless

    echo "-----------------------"
  done
done

echo "All tasks complete." 