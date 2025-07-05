#!/usr/bin/env python3
"""
Entry point script for SB3 PPO training that can be run directly.
This script handles the import issues and provides a clean interface.
"""

import os
import sys
import time
import random
import argparse
import warnings
import pathlib
import importlib
import multiprocessing as mp
from functools import partial as bind
from types import SimpleNamespace

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

# --- Standard imports
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics as _gym_robo  # type: ignore
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import re

_gym_robo.register_robotics_envs()
import trailenv 

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

# Import shared utilities
from baselines.shared.config_utils import load_config

# Environment registration & constants
# -----------------------------------------------------------------------------

os.environ.setdefault("MUJOCO_GL", "egl")  # Off-screen rendering on headless nodes

# -----------------------------------------------------------------------------
# Observation Filtering Wrapper (identical to train_blindpick.py)
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
# Training Function
# -----------------------------------------------------------------------------

def train_ppo_sb3(envs, config, seed, num_iterations=None):
    """Train PPO using Stable Baselines 3."""
    
    # Global seeding for reproducibility
    set_random_seed(seed)
    
    # Create environment function (identical to train_blindpick.py)
    def _make_env():
        # Create the base environment using gymnasium
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        
        # Apply observation filtering if keys are specified
        if hasattr(config, 'keys') and config.keys:
            mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
            cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
            env = ObservationFilterWrapper(
                env, 
                mlp_keys=mlp_keys,
                cnn_keys=cnn_keys
            )
        
        env.reset(seed=seed)
        return env
    
    # Create vectorized environment for SB3 (identical to train_blindpick.py)
    if config.num_envs > 1:
        env_fns = [lambda i=i: _make_env() for i in range(config.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([_make_env])

    vec_env = VecMonitor(vec_env)

    # Create separate eval environment
    eval_env = _make_env()
    
    # Calculate consistent logging frequencies (using SB3 defaults)
    eval_freq = max(1000 // config.num_envs, 1)  # Default eval every 1000 steps
    log_interval = 1  # Default log every rollout
    
    print(f"Eval frequency: every {eval_freq} env.step() calls (~{eval_freq * config.num_envs} total env steps)")
    print(f"Log interval: every {log_interval} rollouts")

    # Initialize W&B if enabled
    run = None
    if config.use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            name=f"ppo_sb3-{config.task}-{config.exp_name}-seed{seed}",
            config=dict(
                env=config.task,
                algo="PPO_SB3",
                total_timesteps=config.total_timesteps,
                policy="MultiInputPolicy",
                seed=seed,
                num_envs=config.num_envs,
            ),
            sync_tensorboard=True,
        )

    # Create PPO model with SB3 defaults
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo_sb3-{config.task}-{config.exp_name}-seed{seed}",
    )

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/ppo_sb3-{config.task}-{config.exp_name}-seed{seed}",
        log_path=f"./eval_logs/ppo_sb3-{config.task}-{config.exp_name}-seed{seed}",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Prepare callbacks
    callbacks = [eval_callback]
    if run is not None:
        callbacks.append(WandbCallback(
            gradient_save_freq=1_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))

    # Train the model
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
    )

    vec_env.close()
    eval_env.close()
    
    if run is not None:
        run.finish()
    
    return model

# -----------------------------------------------------------------------------
# CLI parsing + main logic
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=[], help="Which named configs to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Override seed manually.")
    parser.add_argument('--task', type=str, default="gymnasium_TigerDoorKey-v0", help="Environment task.")
    parser.add_argument('--num_envs', type=int, default=8, help="Number of parallel environments.")
    parser.add_argument('--total_timesteps', type=int, default=500000, help="Total timesteps for training.")
    parser.add_argument('--use_wandb', action='store_true', help="Enable wandb logging.")
    parser.add_argument('--no_wandb', action='store_true', help="Disable wandb logging.")
    parser.add_argument('--wandb_project', type=str, default="sensor-dropout", help="Wandb project name.")
    return parser.parse_args()

def main(argv=None):
    """Main training function."""
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args()
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(argv, config_path)

    # Use 'spawn' for multiprocessing to avoid issues with libraries like wandb
    mp.set_start_method("spawn", force=True)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Calculate number of iterations like in thesis
    num_iterations = config.total_timesteps // (config.num_envs * 2048)  # SB3 default n_steps

    # Run training
    print(f"Starting SB3 PPO training on {config.task} with {config.num_envs} environments")
    print(f"Training for {num_iterations} iterations")
    if config.use_wandb:
        print(f"Wandb logging enabled - project: {config.wandb_project}")
    else:
        print("Wandb logging disabled")
    
    trained_agent = train_ppo_sb3(None, config, seed, num_iterations=num_iterations)
    
    print("Training completed!")
    return trained_agent

if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds") 