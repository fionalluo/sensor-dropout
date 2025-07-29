#!/bin/bash

# User: specify your list of tasks and wandb project here
TASKS=(
  # "Isaac-Ant-v0"
  # "Isaac-Humanoid-v0"
  # "Isaac-Lift-Cube-Franka-v0"
  # "Isaac-Repose-Cube-Shadow-Direct-v0"
  "Isaac-Repose-Cube-Shadow-With-Contact-Sensors-Direct-v0"
)
WANDB_PROJECT="isaac-test-729"
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
export ISAAC_SIM_USE_EGL=1
export DISPLAY=

for ((i=0; i<$NUM_SEEDS; i++)); do
  for TASK in "${TASKS[@]}"; do
    SEED=$((INITIAL_SEED + i))
    echo "Running RL-Games PPO baseline with task ${TASK}, seed ${SEED}, wandb_project ${WANDB_PROJECT}"

    /workspace/isaaclab/isaaclab.sh -p /workspace/sensor-dropout/baselines_isaac/ppo/train.py \
      --task "$TASK" \
      --seed $SEED \
      --wandb-project-name "$WANDB_PROJECT" \
      --wandb-entity "$WANDB_ENTITY" \
      --track \
      --headless

    echo "-----------------------"
  done
done

echo "All tasks complete." 