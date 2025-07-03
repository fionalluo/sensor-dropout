import os
configs = [
    "full", 
    "side_cam_only",
    "front_cam_only",
    "wrist_cam_only",
    "vision_only",
    "touch_vision",
    "robot_state_vision"
]

num_seeds = 4
for config in configs:
    cmd = f"python debug_blindpick.py --sensor_config {config} --slurm --num_seeds {num_seeds}"
    os.system(cmd)