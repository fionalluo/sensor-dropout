#!/usr/bin/env python3
"""
Entry point script for SB3 PPO training that can be run directly.
This script handles the import issues and provides a clean interface.
"""
try:
    import aerial_gym
except ImportError:
    print("[WARN] aerial_gym could not be imported. Continuing without it.")

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
warnings.filterwarnings('ignore', '.*The obs returned by the `reset\(\)` method was expecting numpy array dtype to be int32, actual type: int64.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `reset\(\)` method is not within the observation space.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `step\(\)` method was expecting numpy array dtype to be int32, actual type: int64.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `step\(\)` method is not within the observation space.*')
warnings.filterwarnings('ignore', '.*Overriding environment.*')

# --- Standard imports
import numpy as np
import gymnasium as gym
import gymnasium_robotics as _gym_robo  # type: ignore
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import re

_gym_robo.register_robotics_envs()
import trailenv 
import torch  # must be imported after aerial_gym / isaac gym
from aerial_gym.registry.task_registry import task_registry

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

# Import shared utilities
from baselines.shared.config_utils import load_config
from baselines.shared.masking_utils import mask_observations_for_student
from baselines.shared.isaac_vec_env_wrapper import IsaacVecEnvWrapper
from baselines.shared.eval_utils_sb3 import ObservationFilterWrapper, VecObservationFilterWrapper, CustomEvalCallback

# Environment registration & constants
# -----------------------------------------------------------------------------

os.environ.setdefault("MUJOCO_GL", "egl")  # Off-screen rendering on headless nodes

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------

def train_ppo(envs, config, seed, enable_custom_eval=True):
    """Train PPO using Stable Baselines 3.
    
    Args:
        envs: Vectorized environments (unused, kept for compatibility)
        config: Configuration object
        seed: Random seed
        enable_custom_eval: Whether to enable custom evaluation across subsets (default: True)
    """
    
    # Global seeding for reproducibility
    set_random_seed(seed)
    
    suite, task = config.task.split('_', 1)
    
    # Isaac Gym: Only create the base env ONCE per process
    if suite == "isaacgym":
        base_env = task_registry.make_task(task.replace('-v0', ''), num_envs=config.num_envs)
        base_env = IsaacVecEnvWrapper(base_env)
        # Training env (with filtering)
        mlp_keys = getattr(config.keys, 'mlp_keys', '.*') if hasattr(config, 'keys') and config.keys else '.*'
        cnn_keys = getattr(config.keys, 'cnn_keys', '.*') if hasattr(config, 'keys') and config.keys else '.*'
        vec_env = VecObservationFilterWrapper(base_env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
        vec_env = VecMonitor(vec_env)
        # For evaluation, just re-wrap the same base_env as needed, but do NOT create a new one
        def _make_filtered_eval_env():
            return VecObservationFilterWrapper(base_env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
        def _make_eval_env():
            return base_env  # unfiltered
        filtered_eval_env = _make_filtered_eval_env()
        unfiltered_eval_env = _make_eval_env()
        def _make_eval_env_for_callback():
            return base_env
        def _make_filtered_eval_env_for_callback():
            return VecObservationFilterWrapper(base_env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
    else:
        # Helper function to create environment with common logic
        def _create_base_env(task_name=None, apply_filtering=False):
            suite, task = config.task.split('_', 1)
            env = gym.make(task)
            if apply_filtering and hasattr(config, 'keys') and config.keys:
                mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
                cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
                env = ObservationFilterWrapper(env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
            return env
        def _make_env(env_idx=0):
            env = _create_base_env(apply_filtering=True)
            env_seed = seed + env_idx
            env.reset(seed=env_seed)
            return env
        def _make_eval_env():
            env = _create_base_env(apply_filtering=False)
            env.reset(seed=seed)
            return env
        def _make_filtered_eval_env():
            env = _create_base_env(apply_filtering=True)
            env.reset(seed=seed)
            return env
        def _make_eval_env_for_callback():
            env = _create_base_env(apply_filtering=False)
            env.reset(seed=seed)
            return env
        def _make_filtered_eval_env_for_callback():
            env = _create_base_env(apply_filtering=True)
            env.reset(seed=seed)
            return env
        if config.num_envs > 1:
            env_fns = [lambda i=i: _make_env(env_idx=i) for i in range(config.num_envs)]
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(lambda: _make_env(env_idx=0))
        vec_env = VecMonitor(vec_env)
        filtered_eval_env = _make_filtered_eval_env()
        unfiltered_eval_env = _make_eval_env()
    
    # Get evaluation settings from config
    eval_freq = getattr(config.eval, 'eval_freq', max(10000 // config.num_envs, 1))
    n_eval_episodes = getattr(config.eval, 'n_eval_episodes', 5)
    log_interval = getattr(config, 'log_interval', 1)
    
    print(f"Eval frequency: every {eval_freq} env.step() calls (~{eval_freq * config.num_envs} total env steps)")
    print(f"Log interval: every {log_interval} rollouts")
    print(f"Number of eval episodes: {n_eval_episodes}")

    # Initialize W&B if enabled
    run = None
    if config.use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            name=f"ppo-{config.task}-{config.exp_name}-seed{seed}",
            config=config,  # Pass the original config object
            sync_tensorboard=True,
        )

    # Get PPO hyperparameters from config
    ppo_config = config.ppo
    
    print(f"PPO Hyperparameters:")
    print(f"  learning_rate: {ppo_config.learning_rate}")
    print(f"  n_steps: {ppo_config.n_steps}")
    print(f"  batch_size: {ppo_config.batch_size}")
    print(f"  n_epochs: {ppo_config.n_epochs}")
    print(f"  clip_range_vf: {ppo_config.clip_range_vf}")
    print(f"  ent_coef: {ppo_config.ent_coef}")
    
    # Create PPO model with configurable hyperparameters
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        clip_range_vf=ppo_config.clip_range_vf,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo-{config.task}-{config.exp_name}-seed{seed}",
    )
    
    # Create evaluation callbacks
    callbacks = []
    # Only use EvalCallback for non-Isaac Gym environments
    if suite != "isaacgym":
        eval_callback = EvalCallback(
            filtered_eval_env,  # Use filtered environment (same as training)
            best_model_save_path=f"./best_models/ppo-{config.task}-{config.exp_name}-seed{seed}",
            log_path=f"./eval_logs/ppo-{config.task}-{config.exp_name}-seed{seed}",
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
    
    # Conditionally add custom evaluation callback
    if enable_custom_eval:
        print("Enabling custom evaluation across observation subsets")
        custom_eval_callback = CustomEvalCallback(
            unfiltered_eval_env,  # Use unfiltered environment for masking
            config,
            _make_eval_env_for_callback,  # Function to create unfiltered environments
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
            debug=False,  # Enable debug to see what's happening
            base_env_for_keys=base_env if suite == "isaacgym" else None
        )
        callbacks.append(custom_eval_callback)
    else:
        print("Custom evaluation across subsets disabled - using standard evaluation only")
    if run is not None:
        callbacks.append(WandbCallback(
            gradient_save_freq=1_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))

    # Perform initial evaluation at step 0 like simple_imitation.py
    print("\nInitial evaluation at step 0")
    for callback in callbacks:
        # Only run initial evaluation for CustomEvalCallback (comprehensive evaluation)
        # Skip standard EvalCallback since model logger isn't initialized yet
        if isinstance(callback, CustomEvalCallback):
            callback.init_callback(model)
            # Set num_timesteps to 0 for initial evaluation
            callback.num_timesteps = 0
            callback._run_comprehensive_evaluation()

    # Train the model
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
    )

    vec_env.close()
    filtered_eval_env.close()
    # Only close base_env once for Isaac Gym
    if suite != "isaacgym":
        unfiltered_eval_env.close()
    
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

    # Run training
    print(f"Starting SB3 PPO training on {config.task} with {config.num_envs} environments")
    print(f"Training for {config.total_timesteps} total timesteps")
    if config.use_wandb:
        print(f"Wandb logging enabled - project: {config.wandb_project}")
    else:
        print("Wandb logging disabled")
    
    trained_agent = train_ppo(None, config, seed, enable_custom_eval=True)
    
    print("Training completed!")
    return trained_agent

if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds") 