import argparse
import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import sys
import re
import multiprocessing as mp

import gymnasium as gym
import gymnasium_robotics as _gym_robo  # type: ignore
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
import wandb

# Environment registration & constants
# -----------------------------------------------------------------------------

os.environ.setdefault("MUJOCO_GL", "egl")  # Off-screen rendering on headless nodes

_gym_robo.register_robotics_envs()

TOTAL_TIMESTEPS = 50_000_000

# Sensor configurations (similar to config.yaml)
# .* means all keys and ^$ means no keys
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
    },
    ## oracle state use robot_state and obj_state and nothing else
    "oracle_state": {
        "mlp_keys": r"\b(robot_state|obj_state)\b",
        "cnn_keys": "^$"
    }
}

# ---- Environment ----
def _make_env(difficulty: float, config: dict, seed: int):
    env = gym.make(f"FOFixedGripper2DBlind{int(difficulty*100)}cmPick")
    # Apply observation filtering
    env = ObservationFilterWrapper(
        env, 
        mlp_keys=config["mlp_keys"],
        cnn_keys=config["cnn_keys"]
    )
    env.reset(seed=seed)
    return env


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

    def step(self, action):
        """Filter the terminal observation if it exists."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "terminal_observation" in info:
            info["terminal_observation"] = self.observation(info["terminal_observation"])
        return self.observation(obs), reward, terminated, truncated, info

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

def train(seed: int, sensor_config: str = "full", difficulty: float = 0.03,
          wandb_project: str = "blindpick-ppo", wandb_public: bool = False,
          num_envs: int = 1, n_steps: int = 4096, batch_size: int = 256, 
          eval_freq_steps: int = 50_000, log_interval_steps: int = 10_000) -> None:
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
        project=wandb_project,
        name=f"ppo-FOFixedGripper2DBlind{int(difficulty*100)}cmPick-{sensor_config}-seed{seed}",
        config=dict(
            env=f"FOFixedGripper2DBlind{int(difficulty*100)}cmPick",
            algo="PPO",
            total_timesteps=TOTAL_TIMESTEPS,
            policy="MultiInputPolicy",
            seed=seed,
            num_envs=num_envs,
            sensor_config=sensor_config,
            mlp_keys=config["mlp_keys"],
            cnn_keys=config["cnn_keys"],
            eval_freq_steps=eval_freq_steps,
            log_interval_steps=log_interval_steps,
        ),
        sync_tensorboard=True,  # Enable tensorboard syncing
        anonymous="never" if wandb_public else None,
    )

    if num_envs > 1:
        env_fns = [lambda i=i: _make_env(difficulty, config, seed + i) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([lambda: _make_env(difficulty, config, seed)])

    vec_env = VecMonitor(vec_env)

    # Create separate eval environment
    eval_env = _make_env(difficulty, config, seed + 1000)  # Different seed for eval
    
    # Calculate consistent logging frequencies (independent of num_envs)
    eval_freq = max(eval_freq_steps // num_envs, 1)  # Convert to env.step() calls
    log_interval = max(log_interval_steps // (n_steps * num_envs), 1)  # Convert to rollouts
    
    print(f"Eval frequency: every {eval_freq} env.step() calls (~{eval_freq * num_envs} total env steps)")
    print(f"Log interval: every {log_interval} rollouts (~{log_interval * n_steps * num_envs} total env steps)")

    # ---- Model ----
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log=f"./tb_logs/ppo-FOFixedGripper2DBlind{int(difficulty*100)}cmPick-{sensor_config}-seed{seed}",  # Enable tensorboard logging
    )

    # Create evaluation callback
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/ppo-FOFixedGripper2DBlind{int(difficulty*100)}cmPick-{sensor_config}-seed{seed}",
        log_path=f"./eval_logs/ppo-FOFixedGripper2DBlind{int(difficulty*100)}cmPick-{sensor_config}-seed{seed}",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            eval_callback,
            WandbCallback(
                gradient_save_freq=1_000,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
        ],
        log_interval=log_interval,  # Control rollout logging frequency
    )

    vec_env.close()
    eval_env.close()
    run.finish()


# -----------------------------------------------------------------------------
# CLI parsing + main logic
# -----------------------------------------------------------------------------

"""
eval_freq_steps: influence eval/mean_reward which is on a separate eval_env and the policy
is run deterministically

log_interval_steps: influence the logging of rollout stats which is on the same env as the policy
and the policy is stochastic.

`eval/` is more reliable for tracking progress.
"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO on FOFixedGripper2DBlind7cmPick with different sensor configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution mode
    p.add_argument("--wandb_project", default="blindpick-ppo", help="W&B project name")
    p.add_argument("--wandb_public", action="store_true", help="Make W&B project public.")

    p.add_argument("--slurm", action="store_true", help="Submit each seed as a Slurm job instead of running locally.")

    # Seed handling
    p.add_argument("--num_seeds", type=int, default=1, help="Number of consecutive seeds to run.")
    p.add_argument("--seed", type=int, help="Initial seed; if omitted one is generated.")
    p.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments for training.")

    # Sensor configuration
    p.add_argument("--sensor_config", choices=list(SENSOR_CONFIGS.keys()), default="full",
                   help="Sensor configuration to use.")
    p.add_argument("--difficulty", type=float, default=0.07, help="Difficulty of the task.")

    # Slurm options
    p.add_argument("--job_name", default="blindpick_ppo", help="Base Slurm job name")
    p.add_argument("--time", default="09:00:00", help="Slurm time limit (HH:MM:SS)")
    p.add_argument("--partition", default="eaton-compute", help="Slurm partition")
    p.add_argument("--qos", default="ee-high", help="Quality of service")
    p.add_argument("--gpus", default="1", help="GPUs per job")
    p.add_argument("--mem", default="32G", help="Memory per job")
    p.add_argument("--cpus", default="32", help="CPUs per task")

    # PPO sampling hyper-params
    p.add_argument("--target_rollout",
                   type=int,
                   default=4096,
                   help="Desired total number of environment steps per PPO iteration. ")
    p.add_argument("--n_steps", type=int, default=None,
                   help="[Optional] Rollout length per environment. Overrides --target_rollout.")
    p.add_argument("--batch_size", type=int,
                   default=256,
                   help="Mini-batch size for PPO gradients. Must divide n_steps*num_envs. "
                        "Defaults to 256 if omitted.")

    # Logging frequency control
    p.add_argument("--eval_freq_steps", type=int, default=50_000,
                   help="Evaluate model every N total environment steps (consistent across num_envs)")
    p.add_argument("--log_interval_steps", type=int, default=10_000,
                   help="Log rollout stats every N total environment steps (consistent across num_envs)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Use 'spawn' for multiprocessing to avoid issues with libraries like wandb
    # that might not be fork-safe.
    mp.set_start_method("spawn", force=True)

    # ----- Derive PPO sampling hyper-params -----
    if args.n_steps is None:
        # If n_steps not specified, derive from target_rollout
        import math
        n_steps_float = args.target_rollout / args.num_envs
        # Round up to nearest multiple of 8 for efficiency
        args.n_steps = math.ceil(n_steps_float / 8) * 8
        print(f"n_steps not provided. Derived from target_rollout: {args.n_steps}")

    rollout_len = args.n_steps * args.num_envs
    assert rollout_len % args.batch_size == 0, f"rollout_len must be divisible by batch_size, but got {rollout_len} and {args.batch_size}"

    # Seed list
    initial_seed = args.seed if args.seed is not None else generate_unique_seed()
    seeds = [initial_seed + i for i in range(args.num_seeds)]

    if args.slurm:
        # Wrap each seed in its own Slurm job
        script_path = Path(__file__).resolve()
        for idx, seed in enumerate(seeds):
            cmd = (
                f"python -u {script_path} --seed {seed} --sensor_config {args.sensor_config} "
                f"--difficulty {args.difficulty} --wandb_project {args.wandb_project} "
                f"--num_envs {args.num_envs} --n_steps {args.n_steps} --batch_size {args.batch_size} "
                f"--eval_freq_steps {args.eval_freq_steps} --log_interval_steps {args.log_interval_steps}"
            )
            if args.wandb_public:
                cmd += " --wandb_public"
            submit_to_slurm(cmd, idx=idx, args=args)
        print("✅ All jobs submitted to Slurm.")
        return

    # Local execution (sequential)
    for seed in seeds:
        print(f"\n==== Training seed {seed} with sensor config '{args.sensor_config}' ====")
        train(seed=seed, sensor_config=args.sensor_config, difficulty=args.difficulty,
              wandb_project=args.wandb_project, wandb_public=args.wandb_public,
              num_envs=args.num_envs, n_steps=args.n_steps, batch_size=args.batch_size,
              eval_freq_steps=args.eval_freq_steps, log_interval_steps=args.log_interval_steps)


if __name__ == "__main__":
    main()