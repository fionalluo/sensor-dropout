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
warnings.filterwarnings('ignore', '.*The obs returned by the `reset\(\)` method was expecting numpy array dtype to be int32, actual type: int64.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `reset\(\)` method is not within the observation space.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `step\(\)` method was expecting numpy array dtype to be int32, actual type: int64.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `step\(\)` method is not within the observation space.*')
warnings.filterwarnings('ignore', '.*Overriding environment.*')

# --- Standard imports
import numpy as np
import torch
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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

# Import shared utilities
from baselines.shared.config_utils import load_config
from baselines.shared.masking_utils import mask_observations_for_student

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
# Custom Evaluation Callback for SB3
# -----------------------------------------------------------------------------

class CustomEvalCallback(BaseCallback):
    """Custom evaluation callback that evaluates across different observation subsets."""
    
    def __init__(
        self,
        eval_env,
        config,
        make_eval_env_func,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        debug=False
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        self.make_eval_env_func = make_eval_env_func
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.debug = debug
        self.last_eval = 0
        
        # Get the number of eval configs
        self.num_eval_configs = getattr(config, 'num_eval_configs', 4)
        
        # Parse student keys from the agent's training keys
        if hasattr(config, 'keys') and config.keys:
            # Create a filtered environment to get the actual training keys
            def _get_filtered_keys():
                suite, task = config.task.split('_', 1)
                env = gym.make(task)
                
                # Apply observation filtering if keys are specified (same as training)
                if hasattr(config, 'keys') and config.keys:
                    mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
                    cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
                    env = ObservationFilterWrapper(
                        env, 
                        mlp_keys=mlp_keys,
                        cnn_keys=cnn_keys
                    )
                
                obs, _ = env.reset()
                env.close()
                return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
            
            # Get the actual keys the agent was trained on
            training_keys = _get_filtered_keys()
            self.student_keys = training_keys
            
            if self.debug:
                print(f"[EVAL CALLBACK] Training keys (student keys): {self.student_keys}")
        else:
            self.student_keys = []
    
    def _get_available_keys(self):
        """Get available keys from the evaluation environment."""
        obs, _ = self.eval_env.reset()
        return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
    
    def _parse_keys_from_pattern(self, pattern, available_keys):
        """Parse keys from regex pattern."""
        import re
        regex = re.compile(pattern)
        matched_keys = [k for k in available_keys if regex.search(k)]
        return matched_keys
    
    def _on_step(self):
        """Called after each step."""
        # Check if we should run evaluation
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            print(f"Running custom evaluation at step {self.num_timesteps}...")
            
            # Run evaluation for each environment configuration
            for subset_idx in range(1, self.num_eval_configs + 1):
                env_name = f"env{subset_idx}"
                if not hasattr(self.config.eval_keys, env_name):
                    print(f"Warning: Missing eval_keys for {env_name}")
                    continue
                
                eval_keys = getattr(self.config.eval_keys, env_name)
                mlp_keys_pattern = getattr(eval_keys, 'mlp_keys', '.*')
                cnn_keys_pattern = getattr(eval_keys, 'cnn_keys', '.*')
                
                # Parse teacher keys for this environment
                available_keys = self._get_available_keys()
                teacher_mlp_keys = self._parse_keys_from_pattern(mlp_keys_pattern, available_keys)
                teacher_cnn_keys = self._parse_keys_from_pattern(cnn_keys_pattern, available_keys)
                teacher_keys = teacher_mlp_keys + teacher_cnn_keys
                
                if self.debug:
                    print(f"[EVAL CALLBACK] {env_name} - Teacher keys: {teacher_keys}")
                
                # Run evaluation for this environment
                self._evaluate_environment(env_name, teacher_keys)
            
            self.last_eval = self.num_timesteps
            
        return True
    
    def _evaluate_environment(self, env_name, teacher_keys):
        """Evaluate the agent on a specific environment configuration."""
        # Create a fresh evaluation environment
        eval_env = self.make_eval_env_func()
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_return = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Convert observations to tensors for masking
                obs_tensors = {}
                for key, value in obs.items():
                    if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                        if isinstance(value, np.ndarray):
                            obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                        else:
                            obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
                
                # Mask observations for the student
                masked_obs = mask_observations_for_student(
                    obs_tensors, 
                    self.student_keys, 
                    teacher_keys, 
                    device=None,  # Use CPU for evaluation
                    debug=self.debug and episode == 0
                )
                
                # Debug logging for first episode
                if self.debug and episode == 0:
                    print(f"[EVAL DEBUG] {env_name} - Original obs keys: {list(obs_tensors.keys())}")
                    print(f"[EVAL DEBUG] {env_name} - Student keys: {self.student_keys}")
                    print(f"[EVAL DEBUG] {env_name} - Teacher keys: {teacher_keys}")
                    print(f"[EVAL DEBUG] {env_name} - Masked obs keys: {list(masked_obs.keys())}")
                    # Show some values
                    for key in list(masked_obs.keys())[:3]:  # First 3 keys
                        val = masked_obs[key]
                        if isinstance(val, torch.Tensor):
                            print(f"[EVAL DEBUG] {env_name} - {key}: shape={tuple(val.shape)}, mean={val.float().mean().item():.4f}")
                        else:
                            print(f"[EVAL DEBUG] {env_name} - {key}: value={val}")
                
                # Convert masked observations to numpy for SB3
                masked_obs_numpy = {}
                for key, value in masked_obs.items():
                    if isinstance(value, torch.Tensor):
                        masked_obs_numpy[key] = value.cpu().numpy()
                    else:
                        masked_obs_numpy[key] = np.array(value)
                
                # Fix: Add batch dimension for SB3's MultiInputPolicy
                obs_batch = {k: np.expand_dims(v, axis=0) for k, v in masked_obs_numpy.items()}
                
                # Debug: Print shapes for first episode
                if self.debug and episode == 0:
                    print(f"[EVAL DEBUG] {env_name} - Batched obs shapes: {dict((k, v.shape) for k, v in obs_batch.items())}")
                
                # Get action from policy
                with torch.no_grad():
                    action, _ = self.model.predict(obs_batch, deterministic=self.deterministic)
                
                # Extract scalar action if it's a numpy array
                if isinstance(action, np.ndarray):
                    action = action.item() if action.size == 1 else action[0]
                
                # Step environment
                obs, reward, done, truncated, info = eval_env.step(action)
                
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        # Compute metrics
        episode_returns = np.array(episode_returns)
        episode_lengths = np.array(episode_lengths)
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"  {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
        
        # Log to wandb if available
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record(f"full_eval_return/{env_name}/mean_return", mean_return)
            self.logger.record(f"full_eval/{env_name}/std_return", std_return)
            self.logger.record(f"full_eval/{env_name}/mean_length", mean_length)
            self.logger.record(f"full_eval/{env_name}/std_length", std_length)
        
        eval_env.close()

# -----------------------------------------------------------------------------
# Training-time Episode Masking Wrapper
# -----------------------------------------------------------------------------

class EpisodeMaskingWrapper(gym.ObservationWrapper):
    """Wrapper that randomly selects an environment subset for each episode and masks observations accordingly.
    The observation space remains unchanged - only the observations are masked."""
    
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.num_eval_configs = getattr(config, 'num_eval_configs', 4)
        
        # Get all available keys from the environment once
        obs, _ = self.env.reset()
        self.available_keys = [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
        
        # Get student keys directly from the environment using config.keys patterns
        if hasattr(config, 'keys') and config.keys:
            # Parse student keys using config.keys patterns
            mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
            cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
            
            # Get keys that match the patterns
            student_mlp_keys = self._parse_keys_from_pattern(mlp_keys, self.available_keys)
            student_cnn_keys = self._parse_keys_from_pattern(cnn_keys, self.available_keys)
            self.student_keys = student_mlp_keys + student_cnn_keys
            
            # Set the observation space to match the student keys
            self._set_observation_space()
        else:
            print("No keys specified in config.")
            self.student_keys = []
        
        # Current episode's teacher keys (will be set at episode start)
        self.current_teacher_keys = None
        self.current_env_name = None
        
        # Track episode boundaries
        self.episode_start = True
    
    def _set_observation_space(self):
        """Set the observation space to match the student keys."""
        # Create observation space with only the student keys
        student_spaces = {}
        for key in self.student_keys:
            if key in self.env.observation_space.spaces:
                student_spaces[key] = self.env.observation_space.spaces[key]
        
        self.observation_space = gym.spaces.Dict(student_spaces)
    
    def _parse_keys_from_pattern(self, pattern, available_keys):
        """Parse keys from regex pattern."""
        import re
        regex = re.compile(pattern)
        matched_keys = [k for k in available_keys if regex.search(k)]
        return matched_keys
    
    def _select_env_subset(self):
        """Cycle through environment subsets for each episode."""
        # Use episode counter to cycle through environments
        if not hasattr(self, 'episode_counter'):
            self.episode_counter = 0
        
        subset_idx = (self.episode_counter % self.num_eval_configs) + 1
        env_name = f"env{subset_idx}"
        
        if not hasattr(self.config.eval_keys, env_name):
            print(f"Warning: Missing eval_keys for {env_name}, using env1")
            env_name = "env1"
        
        eval_keys = getattr(self.config.eval_keys, env_name)
        mlp_keys_pattern = getattr(eval_keys, 'mlp_keys', '.*')
        cnn_keys_pattern = getattr(eval_keys, 'cnn_keys', '.*')
        
        # Parse teacher keys for this environment
        available_keys = self.available_keys
        teacher_mlp_keys = self._parse_keys_from_pattern(mlp_keys_pattern, available_keys)
        teacher_cnn_keys = self._parse_keys_from_pattern(cnn_keys_pattern, available_keys)
        teacher_keys = teacher_mlp_keys + teacher_cnn_keys
        
        return env_name, teacher_keys
    
    def observation(self, obs):
        """Mask observations based on the current episode's environment subset."""
        if self.episode_start:
            # Select environment subset for this episode (cycle through them)
            self.current_env_name, self.current_teacher_keys = self._select_env_subset()
            self.episode_start = False
            # print(f"[TRAINING] Episode using {self.current_env_name} with {len(self.current_teacher_keys)} teacher keys")
            # print(f"[TRAINING] Student keys: {self.student_keys}")
            # print(f"[TRAINING] Observation space keys: {list(self.observation_space.spaces.keys())}")
        
        # Convert observations to tensors for masking
        obs_tensors = {}
        for key, value in obs.items():
            if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
        
        # Mask observations for the student using the guaranteed correct method
        masked_obs = mask_observations_for_student(
            obs_tensors, 
            self.student_keys, 
            self.current_teacher_keys, 
            device=None,  # Use CPU for training
            debug=False
        )
        
        # Convert back to numpy for the environment
        masked_obs_numpy = {}
        for key, value in masked_obs.items():
            if isinstance(value, torch.Tensor):
                masked_obs_numpy[key] = value.cpu().numpy()
            else:
                masked_obs_numpy[key] = np.array(value)
        
        return masked_obs_numpy
    
    def reset(self, **kwargs):
        """Reset the environment and select a new environment subset for the new episode."""
        obs, info = self.env.reset(**kwargs)
        self.episode_start = True  # Mark that we're starting a new episode
        self.episode_counter = getattr(self, 'episode_counter', 0) + 1  # Increment episode counter
        return self.observation(obs), info
    
    def step(self, action):
        """Step the environment and mask the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If episode is done, mark for new episode
        if terminated or truncated:
            self.episode_start = True
        
        return self.observation(obs), reward, terminated, truncated, info

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------

def train_ppo_dropout(envs, config, seed, num_iterations=None):
    """Train PPO with episode-level observation masking."""
    
    # Global seeding for reproducibility
    set_random_seed(seed)
    
    # Create base environment function (unfiltered - contains all keys)
    def _make_base_env():
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        env.reset(seed=seed)
        return env
    
    # Create environment function with episode-level masking
    def _make_env():
        # Create the base environment using gymnasium (unfiltered - contains all keys)
        base_env = _make_base_env()
        
        # Apply episode-level masking wrapper that randomly selects environment subsets
        # This wrapper will mask observations but keep the observation space unchanged
        env = EpisodeMaskingWrapper(base_env, config)
        
        return env
    
    # Create evaluation environment function (without filtering)
    def _make_eval_env():
        # Create the base environment using gymnasium
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        # No observation filtering for evaluation - keep all keys
        env.reset(seed=seed)
        return env
    
    # Create filtered evaluation environment function (same as training)
    def _make_filtered_eval_env():
        # Create the base environment using gymnasium
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        
        # Apply observation filtering if keys are specified (same as training)
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

    # Create separate eval environments
    # Standard eval environment (filtered, same as training)
    filtered_eval_env = _make_filtered_eval_env()
    # Custom eval environment (unfiltered, for masking)
    unfiltered_eval_env = _make_eval_env()
    
    # Calculate consistent logging frequencies (using SB3 defaults)
    eval_freq = max(10000 // config.num_envs, 1)  # Evaluate every 10000 steps (less frequent)
    log_interval = 1  # Default log every rollout
    
    print(f"Eval frequency: every {eval_freq} env.step() calls (~{eval_freq * config.num_envs} total env steps)")
    print(f"Log interval: every {log_interval} rollouts")

    # Initialize W&B if enabled
    run = None
    if config.use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            name=f"ppo_dropout-{config.task}-{config.exp_name}-seed{seed}",
            config=config,  # Pass the original config object
            sync_tensorboard=True,
        )

    # Create PPO model with SB3 defaults
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo_dropout-{config.task}-{config.exp_name}-seed{seed}",
    )

    # Create evaluation callbacks
    eval_callback = EvalCallback(
        filtered_eval_env,  # Use filtered environment (same as training)
        best_model_save_path=f"./best_models/ppo_dropout-{config.task}-{config.exp_name}-seed{seed}",
        log_path=f"./eval_logs/ppo_dropout-{config.task}-{config.exp_name}-seed{seed}",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    custom_eval_callback = CustomEvalCallback(
        unfiltered_eval_env,  # Use unfiltered environment for masking
        config,
        _make_eval_env,  # Function to create unfiltered environments
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        debug=False  # Enable debug to see what's happening
    )

    # Prepare callbacks
    callbacks = [eval_callback, custom_eval_callback]
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
    filtered_eval_env.close()
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

    # Calculate number of iterations like in thesis
    num_iterations = config.total_timesteps // (config.num_envs * 2048)  # SB3 default n_steps

    # Run training
    print(f"Starting PPO Dropout training on {config.task} with {config.num_envs} environments")
    print(f"Training for {num_iterations} iterations")
    if config.use_wandb:
        print(f"Wandb logging enabled - project: {config.wandb_project}")
    else:
        print("Wandb logging disabled")
    
    trained_agent = train_ppo_dropout(None, config, seed, num_iterations=num_iterations)
    
    print("Training completed!")
    return trained_agent

if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds") 