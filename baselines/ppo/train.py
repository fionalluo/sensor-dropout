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
import highway_env 

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
            is_image = (len(space.shape) == 3 or len(space.shape) == 4) and space.shape[-1] == 3
            
            if is_image:
                # Apply CNN key filter for image observations
                if self.cnn_pattern.search(key):
                    filtered_spaces[key] = space
                    print(f"Including CNN key: {key} with shape {space.shape}")
                else:
                    print(f"Excluding CNN key: {key} with shape {space.shape}")
            else:
                # Apply MLP key filter for non-image observations
                if self.mlp_pattern.search(key):
                    filtered_spaces[key] = space
                    print(f"Including MLP key: {key}")
                else:
                    print(f"Excluding MLP key: {key} with shape {space.shape}")
        
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

    def get_wrapper_attr(self, name):
        """Support SB3's get_attr by forwarding to the underlying env."""
        return getattr(self.env, name)

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
        self.last_eval_step = -1  # Track last evaluation step to handle irregular increments
        
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
        if pattern == '.*':
            return available_keys
        elif pattern == '^$':
            return []
        else:
            import re
            try:
                regex = re.compile(pattern)
                matched_keys = [k for k in available_keys if regex.search(k)]
                return matched_keys
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")
                return []
        
    def _on_step(self):
        """Called after each step."""
        # Use interval-based evaluation logic like simple_imitation.py
        if self.eval_freq > 0:
            current_step = self.num_timesteps
            
            # Handle initial evaluation at step 0
            if current_step == 0 and self.last_eval_step == -1:
                print(f"\nInitial evaluation at step {current_step}")
                self._run_comprehensive_evaluation()
                self.last_eval_step = current_step
            else:
                # Check if we've crossed an evaluation boundary
                prev_eval_count = self.last_eval_step // self.eval_freq
                current_eval_count = current_step // self.eval_freq
                
                if current_eval_count > prev_eval_count:
                    # We've crossed at least one evaluation boundary
                    next_eval_step = (prev_eval_count + 1) * self.eval_freq
                    print(f"\nEvaluation triggered: step {current_step} crossed boundary at {next_eval_step} (eval_freq={self.eval_freq})")
                    print(f"  Previous eval at step {self.last_eval_step}, current step {current_step}")
                    
                    # Run comprehensive evaluation like simple_imitation.py
                    self._run_comprehensive_evaluation()
                    self.last_eval_step = current_step
            
        return True

    def _run_comprehensive_evaluation(self):
        """Run comprehensive evaluation like simple_imitation.py."""
        print(f"Starting comprehensive evaluation with {self.n_eval_episodes} episodes...")
        print(f"Number of eval configs: {self.num_eval_configs}")
        
        # 1. Evaluate student on default training environment (with filtered observations, same as training)
        self._evaluate_student_default()
        
        # 2. Evaluate student on each teacher configuration (with proper masking)
        env_metrics = {}  # Collect metrics from all environments
        for i in range(1, self.num_eval_configs + 1):
            env_name = f'env{i}'
            if hasattr(self.config.eval_keys, env_name):
                eval_keys = getattr(self.config.eval_keys, env_name)
                metrics = self._evaluate_environment(env_name, eval_keys)
                env_metrics[env_name] = metrics
            else:
                print(f"Warning: Missing eval_keys for {env_name}")
        
        # 3. Compute and log mean metrics across all environments
        self._log_mean_metrics(env_metrics)
        
        print("Evaluation complete!")

    def _evaluate_student_default(self):
        """Evaluate student on training environment with student keys only (filtered, same as training)."""
        print(f"  Evaluating student default (training environment)...")
        
        # Create filtered evaluation environment (same as training)
        suite, task = self.config.task.split('_', 1)
        eval_env = gym.make(task)
        
        # Apply same observation filtering as training
        if hasattr(self.config, 'keys') and self.config.keys:
            mlp_keys = getattr(self.config.keys, 'mlp_keys', '.*')
            cnn_keys = getattr(self.config.keys, 'cnn_keys', '.*')
            eval_env = ObservationFilterWrapper(
                eval_env, 
                mlp_keys=mlp_keys,
                cnn_keys=cnn_keys
            )
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            # Use deterministic seed for evaluation reproducibility
            eval_seed = self.config.seed + 20000 + episode
            obs, _ = eval_env.reset(seed=eval_seed)
            episode_return = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Add batch dimension for SB3's MultiInputPolicy
                obs_batch = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
                
                # Get student action
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
        
        # Log directly to wandb with explicit step, bypassing SB3 logger
        import wandb
        if wandb.run is not None:
            # Use global step count for consistent x-axis scaling
            wandb.log({
                "eval/mean_return": mean_return,
                "eval/std_return": std_return,
                "eval/mean_length": mean_length,
                "global_step": self.num_timesteps
            }, step=self.num_timesteps)
        
        eval_env.close()

    def _evaluate_environment(self, env_name, eval_keys):
        """Evaluate student policy on a specific teacher configuration.
        
        Returns:
            dict: Dictionary containing mean_return, std_return, mean_length
        """
        # Parse teacher keys for this environment
        mlp_keys_pattern = getattr(eval_keys, 'mlp_keys', '.*')
        cnn_keys_pattern = getattr(eval_keys, 'cnn_keys', '.*')
        
        available_keys = self._get_available_keys()
        teacher_mlp_keys = self._parse_keys_from_pattern(mlp_keys_pattern, available_keys)
        teacher_cnn_keys = self._parse_keys_from_pattern(cnn_keys_pattern, available_keys)
        teacher_keys = teacher_mlp_keys + teacher_cnn_keys
        
        # Create a fresh evaluation environment (unfiltered)
        eval_env = self.make_eval_env_func()
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            # Use deterministic seed for environment evaluation reproducibility
            eval_seed = self.config.seed + 30000 + hash(env_name) % 1000 + episode
            obs, _ = eval_env.reset(seed=eval_seed)
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
                    debug=False
                )
                
                # Convert masked observations to numpy for SB3
                masked_obs_numpy = {}
                for key, value in masked_obs.items():
                    if isinstance(value, torch.Tensor):
                        masked_obs_numpy[key] = value.cpu().numpy()
                    else:
                        masked_obs_numpy[key] = np.array(value)
                
                # Add batch dimension for SB3's MultiInputPolicy
                obs_batch = {k: np.expand_dims(v, axis=0) for k, v in masked_obs_numpy.items()}
                
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
        
        print(f"    {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
        
        # Log individual environment metrics directly to wandb
        import wandb
        if wandb.run is not None:
            wandb.log({
                f"full_eval_return/{env_name}/mean_return": mean_return,
                f"full_eval/{env_name}/std_return": std_return,
                f"full_eval/{env_name}/mean_length": mean_length,
                "global_step": self.num_timesteps
            }, step=self.num_timesteps)
        
        eval_env.close()
        
        # Return metrics for aggregation
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length
        }

    def _log_mean_metrics(self, env_metrics: dict):
        """Compute and log mean metrics across all evaluation environments."""
        if not env_metrics:
            return
        
        # Compute means across all environments
        mean_mean_return = sum(metrics['mean_return'] for metrics in env_metrics.values()) / len(env_metrics)
        mean_std_return = sum(metrics['std_return'] for metrics in env_metrics.values()) / len(env_metrics)
        mean_mean_length = sum(metrics['mean_length'] for metrics in env_metrics.values()) / len(env_metrics)
        
        print(f"    env_mean: mean_return={mean_mean_return:.2f}, std_return={mean_std_return:.2f}, mean_length={mean_mean_length:.1f}")
        
        # Log mean metrics directly to wandb with explicit step
        import wandb
        if wandb.run is not None:
            wandb.log({
                f"full_eval_return/env_mean/mean_return": mean_mean_return,
                f"full_eval/env_mean/std_return": mean_std_return,
                f"full_eval/env_mean/mean_length": mean_mean_length,
                "global_step": self.num_timesteps
            }, step=self.num_timesteps)

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

    # Get PPO hyperparameters from config, with fallbacks to SB3 defaults
    ppo_config = getattr(config, 'ppo', {})
    
    # Extract hyperparameters with defaults
    learning_rate = getattr(ppo_config, 'learning_rate', 3e-4)
    n_steps = getattr(ppo_config, 'n_steps', 2048)
    batch_size = getattr(ppo_config, 'batch_size', 64)
    n_epochs = getattr(ppo_config, 'n_epochs', 10)
    gamma = getattr(ppo_config, 'gamma', 0.99)
    gae_lambda = getattr(ppo_config, 'gae_lambda', 0.95)
    clip_range = getattr(ppo_config, 'clip_range', 0.2)
    clip_range_vf = getattr(ppo_config, 'clip_range_vf', None)
    ent_coef = getattr(ppo_config, 'ent_coef', 0.0)
    vf_coef = getattr(ppo_config, 'vf_coef', 0.5)
    max_grad_norm = getattr(ppo_config, 'max_grad_norm', 0.5)
    
    print(f"PPO Hyperparameters:")
    print(f"  learning_rate: {learning_rate}")
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  clip_range_vf: {clip_range_vf}")
    print(f"  ent_coef: {ent_coef}")
    
    tb_log = f"./tb_logs/ppo-{config.task}-{config.exp_name}-seed{seed}"

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        seed=seed,
        tensorboard_log=tb_log,
    )

    # ------------------------------------------------------------------
    # SB3 v2.6+ no longer creates a default logger inside BaseAlgorithm.
    # We must explicitly configure one before any callbacks access it
    # (e.g., during the initial evaluation below).
    # ------------------------------------------------------------------
    from stable_baselines3.common.logger import configure  # local import to avoid circular issues

    # Use the same directory as tensorboard so W&B can sync it easily
    new_logger = configure(folder=tb_log, format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Create evaluation callbacks
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

    # Prepare callbacks - always include standard eval callback
    callbacks = [eval_callback]
    
    # Conditionally add custom evaluation callback
    if enable_custom_eval:
        print("Enabling custom evaluation across observation subsets")
        custom_eval_callback = CustomEvalCallback(
            unfiltered_eval_env,  # Use unfiltered environment for masking
            config,
            _make_eval_env,  # Function to create unfiltered environments
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
            debug=False  # Enable debug to see what's happening
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