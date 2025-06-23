#!/bin/bash

# Function to generate a unique seed
generate_unique_seed() {
  date +%s%N | sha256sum | awk '{ print "0x" substr($1, 1, 8) }'
}

# Base log directory
BASE_LOGDIR=~/logdir_teacher_student

# List of configs to run
CONFIGS=(
  # "cartpole_small"
  # "gymnasium_blindpick"
  
  # "gymnasium_lavatrail8"
  # "gymnasium_lavatrail8_unprivileged"

  # "gymnasium_lavatrail8_imitationlatent_0.1"
  # "gymnasium_lavatrail8_imitationlatent_1"
  # "gymnasium_lavatrail8_imitationlatent_2"
  # "gymnasium_lavatrail8_imitationlatent_4"
  # "gymnasium_lavatrail8_imitationlatent_10"

  # "gymnasium_lavatrail8_studentteacherlatent_0.1"
  # "gymnasium_lavatrail8_studentteacherlatent_1"
  # "gymnasium_lavatrail8_studentteacherlatent_10"

  # "gymnasium_lavatrail8_teacherstudentlatent_0.1"
  # "gymnasium_lavatrail8_teacherstudentlatent_1"
  # "gymnasium_lavatrail8_teacherstudentlatent_2"
  # "gymnasium_lavatrail8_teacherstudentlatent_4"
  # "gymnasium_lavatrail8_teacherstudentlatent_10"

  # "gymnasium_bandit5"
  # "gymnasium_bandit5_unprivileged"

  # "gymnasium_bandit5_imitationlatent_0.1"
  # "gymnasium_bandit5_imitationlatent_1"
  # "gymnasium_bandit5_imitationlatent_2"
  # "gymnasium_bandit5_imitationlatent_4"
  # "gymnasium_bandit5_imitationlatent_10"

  # "gymnasium_bandit5_studentteacherlatent_0.1"
  # "gymnasium_bandit5_studentteacherlatent_1"
  # "gymnasium_bandit5_studentteacherlatent_2"
  # "gymnasium_bandit5_studentteacherlatent_4"
  # "gymnasium_bandit5_studentteacherlatent_10"

  # "gymnasium_bandit5_teacherstudentlatent_0.1"
  # "gymnasium_bandit5_teacherstudentlatent_1"
  # "gymnasium_bandit5_teacherstudentlatent_2"
  # "gymnasium_bandit5_teacherstudentlatent_4"
  # "gymnasium_bandit5_teacherstudentlatent_10"

  "robopianist_piano"
  # "robopianist_piano_unprivileged"
  # "robopianist_piano_studentteacherlatent_0.1"
  # "robopianist_piano_studentteacherlatent_1"
  # "robopianist_piano_studentteacherlatent_2"
  # "robopianist_piano_studentteacherlatent_4"
  # "robopianist_piano_studentteacherlatent_10"
  # "robopianist_piano_imitationlatent_0.1"
  # "robopianist_piano_imitationlatent_1"
  # "robopianist_piano_imitationlatent_2"
  # "robopianist_piano_imitationlatent_4"
  # "robopianist_piano_imitationlatent_10"
  # "robopianist_piano_teacherstudentlatent_0.1"
  # "robopianist_piano_teacherstudentlatent_1"
  # "robopianist_piano_teacherstudentlatent_2"
  "robopianist_piano_teacherstudentlatent_4"
  # "robopianist_piano_teacherstudentlatent_10"

  # "gymnasium_blindcuberotate"
  # "gymnasium_blindcuberotate_unprivileged"
  # "gymnasium_blindcuberotate_imitationlatent_0.1"
  # "gymnasium_blindcuberotate_imitationlatent_1"
  # "gymnasium_blindcuberotate_imitationlatent_2"
  # "gymnasium_blindcuberotate_imitationlatent_4"
  # "gymnasium_blindcuberotate_imitationlatent_10"
  # "gymnasium_blindcuberotate_studentteacherlatent_0.1"
  # "gymnasium_blindcuberotate_studentteacherlatent_1"
  # "gymnasium_blindcuberotate_studentteacherlatent_2"
  # "gymnasium_blindcuberotate_studentteacherlatent_4"
  # "gymnasium_blindcuberotate_studentteacherlatent_10"
  # "gymnasium_blindcuberotate_teacherstudentlatent_0.1"
  # "gymnasium_blindcuberotate_teacherstudentlatent_1"
  # "gymnasium_blindcuberotate_teacherstudentlatent_2"
  # "gymnasium_blindcuberotate_teacherstudentlatent_4"
  # "gymnasium_blindcuberotate_teacherstudentlatent_10"

  # "gymnasium_blindcheetah"
  # "gymnasium_blindcheetah_unprivileged"
  # "gymnasium_blindcheetah_imitationlatent_0.1"
  # "gymnasium_blindcheetah_imitationlatent_1"
  # "gymnasium_blindcheetah_imitationlatent_2"
  # "gymnasium_blindcheetah_imitationlatent_4"
  # "gymnasium_blindcheetah_imitationlatent_10"
  # "gymnasium_blindcheetah_studentteacherlatent_0.1"
  # "gymnasium_blindcheetah_studentteacherlatent_1"
  # "gymnasium_blindcheetah_studentteacherlatent_2"
  # "gymnasium_blindcheetah_studentteacherlatent_4"
  # "gymnasium_blindcheetah_studentteacherlatent_10"
  # "gymnasium_blindcheetah_teacherstudentlatent_0.1"
  # "gymnasium_blindcheetah_teacherstudentlatent_1"
  # "gymnasium_blindcheetah_teacherstudentlatent_2"
  # "gymnasium_blindcheetah_teacherstudentlatent_4"
  # "gymnasium_blindcheetah_teacherstudentlatent_10"
)

NUM_SEEDS=1
INITIAL_SEED=$(generate_unique_seed)

SEEDS=()
for ((i=0; i<$NUM_SEEDS; i++)); do
  SEEDS+=($((INITIAL_SEED + i)))
done

export MUJOCO_GL=egl;

# Iterate
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LOGDIR="${BASE_LOGDIR}/${CONFIG}_${SEED}"

    echo "Running Teacher-Student with config ${CONFIG} and seed ${SEED}, logging to ${LOGDIR}"

    timeout 4h python3 -u thesis/teacher_student/train.py \
      --configs ${CONFIG} \
      --seed "$SEED"

    if [ $? -eq 124 ]; then
      echo "Command timed out for config ${CONFIG} and seed ${SEED}."
    else
      echo "Command completed for config ${CONFIG} and seed ${SEED}."
    fi

    echo "-----------------------"
  done
done

echo "All tasks complete." 