#!/bin/bash

# Always run the baselines_isaac/ppo/train.py script via isaaclab.sh, passing through any extra arguments
/workspace/isaaclab/isaaclab.sh -p /workspace/sensor-dropout/baselines_isaac/ppo/train.py "$@" 