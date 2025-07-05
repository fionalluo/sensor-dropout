import os
# configs = [
#     "full", 
#     "side_cam_only",
#     "front_cam_only",
#     "wrist_cam_only",
#     "vision_only",
#     "touch_vision",
#     "robot_state_vision"
#     "oracle_state",
# ]

configs = [
    "oracle_state"
]
# difficulties = [0.03, 0.07]
difficulties = [0.07]
num_seeds = 4
num_envs = 32

# difficulties = [0.03]
# num_seeds = 1

wandb_project = "blindpick-ppo-oracle-state"
for config in configs:
    for difficulty in difficulties: 
        cmd = f"python train_blindpick.py --sensor_config {config} --difficulty {difficulty} --slurm --num_seeds {num_seeds} --wandb_project {wandb_project} --wandb_public --num_envs {num_envs} --eval_freq_steps 50000 --log_interval_steps 10000"
        os.system(cmd)