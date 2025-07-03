import argparse
import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import sys
import re

import gymnasium as gym
import gymnasium_robotics as _gym_robo  # type: ignore
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
import wandb

# Environment registration & constants
# -----------------------------------------------------------------------------

os.environ.setdefault("MUJOCO_GL", "egl")  # Off-screen rendering on headless nodes

_gym_robo.register_robotics_envs()

ENV_NAME = "FOFixedGripper2DBlind7cmPick"
TOTAL_TIMESTEPS = 4_000_000

# Sensor configurations (similar to config.yaml)
SENSOR_CONFIGS = {
    "full": {  # full privileged observations
        "mlp_keys": ".*",
        "cnn_keys": ".*"
    },
    "vision_only": {  # NO robot state, NO touch, ONLY all cameras
        "mlp_keys": "^$",
        "cnn_keys": ".*"
    },
    # "state_only": {  # robot state, touch, NO cameras
    #     "mlp_keys": ".*",
    #     "cnn_keys": "^$"
    # },
    "robot_state_vision": {  # robot state, NO touch, but using all cameras
        "mlp_keys": "robot_state",
        "cnn_keys": ".*"
    },
    "touch_vision": {  # NO robot state, touch, but using all cameras
        "mlp_keys": "touch",
        "cnn_keys": ".*"
    },
    "wrist_cam_only": {  # robot state, touch, but only wrist camera
        "mlp_keys": ".*",
        "cnn_keys": "gripper_camera_rgb"
    },
    "front_cam_only": {  # robot state, touch, but only front camera
        "mlp_keys": ".*",
        "cnn_keys": "camera_front"
    },
    "side_cam_only": {  # robot state, touch, but only side camera
        "mlp_keys": ".*",
        "cnn_keys": "camera_side"
    }
}

# -----------------------------------------------------------------------------
# Observation Filtering Wrapper
# -----------------------------------------------------------------------------

class ObservationFilterWrapper(gym.ObservationWrapper):
    """Wrapper to filter observations based on mlp_keys and cnn_keys patterns."""
    
    def __init__(self, env, mlp_keys: str = ".*", cnn_keys: str = ".*"):
        super().__init__(env)
        self.mlp_pattern = re.compile(mlp_keys)
        self.cnn_pattern = re.compile(cnn_keys)
        
        # Filter the observation space
        self._filter_observation_space()
    
    def _filter_observation_space(self):
        """Filter the observation space based on key patterns."""
        original_spaces = self.env.observation_space.spaces
        filtered_spaces = {}
        
        for key, space in original_spaces.items():
            # Determine if this is an image observation (3D with channel dimension)
            is_image = len(space.shape) == 3 and space.shape[-1] == 3
            
            if is_image:
                # Apply CNN key filter for image observations
                if self.cnn_pattern.search(key):
                    filtered_spaces[key] = space
                    print(f"Including CNN key: {key}")
                else:
                    print(f"Excluding CNN key: {key}")
            else:
                # Apply MLP key filter for non-image observations
                if self.mlp_pattern.search(key):
                    filtered_spaces[key] = space
                    print(f"Including MLP key: {key}")
                else:
                    print(f"Excluding MLP key: {key}")
        
        self.observation_space = gym.spaces.Dict(filtered_spaces)
        print(f"Filtered observation space keys: {list(filtered_spaces.keys())}")
    
    def observation(self, obs):
        """Filter the observation based on the patterns."""
        filtered_obs = {}
        
        for key, value in obs.items():
            if key in self.observation_space.spaces:
                filtered_obs[key] = value
        
        return filtered_obs

# -----------------------------------------------------------------------------
# Helpers (seed + Slurm)
# -----------------------------------------------------------------------------

def generate_unique_seed() -> int:
    """Generate a (reasonably) unique 32-bit seed, similar to run_ppo.py"""
    ts_ns = time.time_ns() & 0xFFFF_FFFF
    return (ts_ns ^ random.getrandbits(32)) & 0xFFFF_FFFF


def submit_to_slurm(cmd: str, *, idx: int, args: argparse.Namespace) -> None:
    """Submit *cmd* to Slurm with provided CLI options (mirrors run_ppo.py)."""
    out_dir = Path("slurm_outs/blindpick_ppo")
    out_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{args.job_name}_{idx}"
    sbatch_cmd: List[str] = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={out_dir}/{job_name}-%j.out",
        f"--time={args.time}",
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        f"--gpus={args.gpus}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus}",
        "--wrap",
        cmd,
    ]

    print("[sbatch]", " ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd, check=True)


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def train(seed: int, sensor_config: str = "full") -> None:
    """Train PPO on ENV_NAME with the given *seed* and sensor configuration, log to W&B."""

    # Global seeding for reproducibility
    set_random_seed(seed)

    # Get sensor configuration
    if sensor_config not in SENSOR_CONFIGS:
        raise ValueError(f"Unknown sensor config: {sensor_config}. Available: {list(SENSOR_CONFIGS.keys())}")
    
    config = SENSOR_CONFIGS[sensor_config]
    print(f"Using sensor config '{sensor_config}': {config}")

    # ---- W&B ----
    run = wandb.init(
        project="blindpick-ppo",
        name=f"ppo-{ENV_NAME}-{sensor_config}-seed{seed}",
        config=dict(
            env=ENV_NAME,
            algo="PPO",
            total_timesteps=TOTAL_TIMESTEPS,
            policy="MultiInputPolicy",
            seed=seed,
            sensor_config=sensor_config,
            mlp_keys=config["mlp_keys"],
            cnn_keys=config["cnn_keys"],
        ),
        sync_tensorboard=True,  # Enable tensorboard syncing
    )

    # ---- Environment ----
    def _make_env():
        env = gym.make(ENV_NAME)
        # Apply observation filtering
        env = ObservationFilterWrapper(
            env, 
            mlp_keys=config["mlp_keys"],
            cnn_keys=config["cnn_keys"]
        )
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecMonitor(vec_env)

    # ---- Model ----
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo-{ENV_NAME}-{sensor_config}-seed{seed}",  # Enable tensorboard logging
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    vec_env.close()
    run.finish()


# -----------------------------------------------------------------------------
# CLI parsing + main logic
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO on FOFixedGripper2DBlind7cmPick with different sensor configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution mode
    p.add_argument("--slurm", action="store_true", help="Submit each seed as a Slurm job instead of running locally.")

    # Seed handling
    p.add_argument("--num_seeds", type=int, default=1, help="Number of consecutive seeds to run.")
    p.add_argument("--seed", type=int, help="Initial seed; if omitted one is generated.")

    # Sensor configuration
    p.add_argument("--sensor_config", choices=list(SENSOR_CONFIGS.keys()), default="full",
                   help="Sensor configuration to use.")

    # Slurm options
    p.add_argument("--job_name", default="blindpick_ppo", help="Base Slurm job name")
    p.add_argument("--time", default="09:00:00", help="Slurm time limit (HH:MM:SS)")
    p.add_argument("--partition", default="eaton-compute", help="Slurm partition")
    p.add_argument("--qos", default="ee-high", help="Quality of service")
    p.add_argument("--gpus", default="1", help="GPUs per job")
    p.add_argument("--mem", default="32G", help="Memory per job")
    p.add_argument("--cpus", default="40", help="CPUs per task")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Seed list
    initial_seed = args.seed if args.seed is not None else generate_unique_seed()
    seeds = [initial_seed + i for i in range(args.num_seeds)]

    if args.slurm:
        # Wrap each seed in its own Slurm job
        script_path = Path(__file__).resolve()
        for idx, seed in enumerate(seeds):
            cmd = f"python -u {script_path} --seed {seed} --sensor_config {args.sensor_config}"
            submit_to_slurm(cmd, idx=idx, args=args)
        print("✅ All jobs submitted to Slurm.")
        return

    # Local execution (sequential)
    for seed in seeds:
        print(f"\n==== Training seed {seed} with sensor config '{args.sensor_config}' ====")
        train(seed, args.sensor_config)


if __name__ == "__main__":
    main()