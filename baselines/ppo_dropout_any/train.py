#!/usr/bin/env python3
"""
Entry point script for SB3 PPO training with probabilistic key dropout (PPODropoutAny).
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
# Observation Filtering Wrapper (identical to original)
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
# Probabilistic Dropout Scheduler (modular design)
# -----------------------------------------------------------------------------

class DropoutScheduler:
    """Modular dropout scheduler that can be easily extended with different probability schedules."""
    
    def __init__(self, dropout_config):
        self.dropout_config = dropout_config
        self.schedule_type = getattr(dropout_config, 'schedule_type', 'constant')
        
        if self.schedule_type == 'constant':
            self.base_prob = getattr(dropout_config, 'base_probability', 0.5)
        elif self.schedule_type == 'linear':
            self.start_prob = getattr(dropout_config, 'start_probability', 0.8)
            self.end_prob = getattr(dropout_config, 'end_probability', 0.2)
            self.total_episodes = getattr(dropout_config, 'total_episodes', 10000)
        elif self.schedule_type == 'exponential':
            self.start_prob = getattr(dropout_config, 'start_probability', 0.8)
            self.decay_rate = getattr(dropout_config, 'decay_rate', 0.995)
            self.min_prob = getattr(dropout_config, 'min_probability', 0.1)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_dropout_probability(self, episode_num):
        """Get the dropout probability for a given episode number."""
        if self.schedule_type == 'constant':
            return self.base_prob
        elif self.schedule_type == 'linear':
            progress = min(episode_num / self.total_episodes, 1.0)
            return self.start_prob + (self.end_prob - self.start_prob) * progress
        elif self.schedule_type == 'exponential':
            prob = self.start_prob * (self.decay_rate ** episode_num)
            return max(prob, self.min_prob)
        else:
            return 0.5  # fallback

# -----------------------------------------------------------------------------
# Custom Evaluation Callback (modified for probabilistic dropout)
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
        self.last_eval_step = -1
        
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
    
    def _on_step(self):
        # Check if we should evaluate
        if self.n_calls % self.eval_freq == 0:
            if self.num_timesteps != self.last_eval_step:
                self.last_eval_step = self.num_timesteps
                self._run_comprehensive_evaluation()
        
        return True
    
    def _run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across different key combinations."""
        if self.debug:
            print(f"\n[EVAL] Running comprehensive evaluation at step {self.num_timesteps}")
        
        # Get all available keys
        available_keys = self._get_available_keys()
        
        # Sample different dropout configurations for evaluation
        dropout_configs = self._generate_eval_dropout_configs(available_keys)
        
        all_metrics = {}
        
        for config_name, teacher_keys in dropout_configs.items():
            if self.debug:
                print(f"[EVAL] Evaluating with config {config_name}: {teacher_keys}")
            
            env_metrics = self._evaluate_environment(config_name, teacher_keys)
            all_metrics[config_name] = env_metrics
        
        # Log aggregated metrics
        self._log_mean_metrics(all_metrics)
        
        if self.debug:
            print(f"[EVAL] Comprehensive evaluation complete")
    
    def _generate_eval_dropout_configs(self, available_keys):
        """Generate a set of dropout configurations for evaluation."""
        configs = {}
        
        # Always include full observation
        configs['full_obs'] = available_keys.copy()
        
        # Include single key configurations
        for key in available_keys:
            configs[f'only_{key}'] = [key]
        
        # Include some random subsets with different sizes
        for size in [2, 3, len(available_keys)//2]:
            if size < len(available_keys) and size > 0:
                random_keys = random.sample(available_keys, size)
                configs[f'random_{size}keys'] = random_keys
        
        return configs
    
    def _evaluate_environment(self, env_name, eval_keys):
        """Evaluate the agent in a specific environment configuration."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Convert observations to tensors for masking
                obs_tensors = {}
                for key, value in obs.items():
                    if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                        if isinstance(value, np.ndarray):
                            obs_tensors[key] = torch.tensor(value.copy(), dtype=torch.float32)
                        else:
                            obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
                
                # Mask observations
                masked_obs = mask_observations_for_student(
                    obs_tensors, 
                    self.student_keys, 
                    eval_keys, 
                    device=None,
                    debug=False
                )
                
                # Convert back to numpy
                masked_obs_numpy = {}
                for key, value in masked_obs.items():
                    if isinstance(value, torch.Tensor):
                        masked_obs_numpy[key] = value.cpu().numpy()
                    else:
                        masked_obs_numpy[key] = np.array(value)
                
                # Get action from model
                action, _ = self.model.predict(masked_obs_numpy, deterministic=self.deterministic)
                
                # Convert action to scalar if it's a numpy array
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                # Step environment
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
    
    def _log_mean_metrics(self, env_metrics: dict):
        """Log mean metrics across all environment configurations."""
        # Calculate mean metrics
        mean_reward = np.mean([metrics['mean_reward'] for metrics in env_metrics.values()])
        mean_length = np.mean([metrics['mean_length'] for metrics in env_metrics.values()])
        
        # Log to wandb and tensorboard
        if hasattr(self.model, 'logger') and self.model.logger:
            self.model.logger.record("eval/mean_reward_across_configs", mean_reward)
            self.model.logger.record("eval/mean_length_across_configs", mean_length)
            
            # Log individual config metrics
            for config_name, metrics in env_metrics.items():
                self.model.logger.record(f"eval/{config_name}/mean_reward", metrics['mean_reward'])
                self.model.logger.record(f"eval/{config_name}/mean_length", metrics['mean_length'])
        
        if self.debug:
            print(f"[EVAL] Mean reward across configs: {mean_reward:.2f}")
            print(f"[EVAL] Mean length across configs: {mean_length:.2f}")

# -----------------------------------------------------------------------------
# Probabilistic Episode Masking Wrapper (main modification)
# -----------------------------------------------------------------------------

class ProbabilisticEpisodeMaskingWrapper(gym.ObservationWrapper):
    """
    Wrapper that probabilistically drops out keys for each episode.
    Instead of cycling through predefined subsets, each key has a probability of being included.
    """
    
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.episode_start = True
        self.current_teacher_keys = []
        
        # Initialize dropout scheduler
        dropout_config = getattr(config, 'dropout', SimpleNamespace())
        self.dropout_scheduler = DropoutScheduler(dropout_config)
        
        # Episode counter for schedule
        self.episode_counter = 0
        
        # Get all available keys
        obs, _ = env.reset()
        self.available_keys = [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
        
        # Parse student keys from config
        if hasattr(config, 'keys') and config.keys:
            mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
            cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
            
            # Get filtered keys
            mlp_matches = self._parse_keys_from_pattern(mlp_keys, self.available_keys)
            cnn_matches = self._parse_keys_from_pattern(cnn_keys, self.available_keys)
            self.student_keys = mlp_matches + cnn_matches
        else:
            self.student_keys = self.available_keys.copy()
        
        # Set observation space to match student keys
        self._set_observation_space()
        
        print(f"[PROB DROPOUT] Available keys: {self.available_keys}")
        print(f"[PROB DROPOUT] Student keys: {self.student_keys}")
    
    def _set_observation_space(self):
        """Set the observation space to match student keys."""
        original_spaces = self.env.observation_space.spaces
        filtered_spaces = {}
        
        for key in self.student_keys:
            if key in original_spaces:
                filtered_spaces[key] = original_spaces[key]
        
        self.observation_space = gym.spaces.Dict(filtered_spaces)
    
    def _parse_keys_from_pattern(self, pattern, available_keys):
        """Parse keys from regex pattern."""
        if pattern == '.*':
            return available_keys
        elif pattern == '^$':
            return []
        else:
            try:
                regex = re.compile(pattern)
                matched_keys = [k for k in available_keys if regex.search(k)]
                return matched_keys
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")
                return []
    
    def _select_probabilistic_keys(self):
        """Probabilistically select which keys to include for this episode."""
        # Get current dropout probability
        dropout_prob = self.dropout_scheduler.get_dropout_probability(self.episode_counter)
        
        # Probabilistically select keys
        selected_keys = []
        for key in self.available_keys:
            if random.random() > dropout_prob:  # Include key if random number > dropout_prob
                selected_keys.append(key)
        
        # Ensure at least one key is selected (constraint: can't have empty observations)
        if not selected_keys:
            # If no keys were selected, randomly pick one
            selected_keys = [random.choice(self.available_keys)]
        
        return selected_keys, dropout_prob
    
    def observation(self, obs):
        """Mask observations based on the current episode's probabilistic key selection."""
        if self.episode_start:
            # Select keys probabilistically for this episode
            self.current_teacher_keys, dropout_prob = self._select_probabilistic_keys()
            self.episode_start = False
            
            print(f"[PROB DROPOUT] Episode {self.episode_counter}: dropout_prob={dropout_prob:.3f}, selected_keys={self.current_teacher_keys}")
        
        # Convert observations to tensors for masking
        obs_tensors = {}
        for key, value in obs.items():
            if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.tensor(value.copy(), dtype=torch.float32)
                else:
                    obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
        
        # Mask observations for the student using the selected keys
        masked_obs = mask_observations_for_student(
            obs_tensors, 
            self.student_keys, 
            self.current_teacher_keys, 
            device=None,
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
        """Reset the environment and start a new episode with probabilistic key selection."""
        obs, info = self.env.reset(**kwargs)
        self.episode_start = True
        self.episode_counter += 1
        return self.observation(obs), info
    
    def step(self, action):
        """Step the environment and mask the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If episode is done, mark for new episode
        if terminated or truncated:
            self.episode_start = True
        
        return self.observation(obs), reward, terminated, truncated, info

# -----------------------------------------------------------------------------
# Training Function (modified for probabilistic dropout)
# -----------------------------------------------------------------------------

def train_ppo_dropout_any(envs, config, seed):
    """Train PPO with probabilistic episode-level observation masking."""
    
    # Global seeding for reproducibility
    set_random_seed(seed)
    
    # Create base environment function (unfiltered - contains all keys)
    def _make_base_env(env_idx=0):
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        # Each environment gets a different seed for proper randomization
        env_seed = seed + env_idx
        env.reset(seed=env_seed)
        return env
    
    # Create environment function with probabilistic episode-level masking
    def _make_env(env_idx=0):
        # Create the base environment using gymnasium (unfiltered - contains all keys)
        base_env = _make_base_env(env_idx=env_idx)
        
        # Apply probabilistic episode-level masking wrapper
        env = ProbabilisticEpisodeMaskingWrapper(base_env, config)
        
        return env
    
    # Create evaluation environment function (without filtering)
    def _make_eval_env():
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        env.reset(seed=seed)
        return env
    
    # Create filtered evaluation environment function (same as training)
    def _make_filtered_eval_env():
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

    # Create vectorized environment for SB3
    if config.num_envs > 1:
        env_fns = [lambda i=i: _make_env(env_idx=i) for i in range(config.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([lambda: _make_env(env_idx=0)])

    vec_env = VecMonitor(vec_env)

    # Create separate eval environments
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
            name=f"ppo_dropout_any-{config.task}-{config.exp_name}-seed{seed}",
            config=config,
            sync_tensorboard=True,
        )

    # Get PPO hyperparameters from config
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
    use_sde = getattr(ppo_config, 'use_sde', False)
    sde_sample_freq = getattr(ppo_config, 'sde_sample_freq', -1)
    target_kl = getattr(ppo_config, 'target_kl', None)
    policy_kwargs = getattr(ppo_config, 'policy_kwargs', {})
    
    print(f"PPO Configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per environment: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE Lambda: {gae_lambda}")

    # Create PPO model
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
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo_dropout_any-{config.task}-{config.exp_name}-seed{seed}",
    )

    # Create evaluation callbacks
    eval_callback = EvalCallback(
        filtered_eval_env,
        best_model_save_path=f"./best_models/ppo_dropout_any-{config.task}-{config.exp_name}-seed{seed}",
        log_path=f"./eval_logs/ppo_dropout_any-{config.task}-{config.exp_name}-seed{seed}",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    custom_eval_callback = CustomEvalCallback(
        unfiltered_eval_env,
        config,
        _make_eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
        debug=False
    )

    # Prepare callbacks
    callbacks = [eval_callback, custom_eval_callback]
    if run is not None:
        callbacks.append(WandbCallback(
            gradient_save_freq=1_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))

    # Perform initial evaluation
    print("\nInitial evaluation at step 0")
    for callback in callbacks:
        if isinstance(callback, CustomEvalCallback):
            callback.init_callback(model)
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
    parser.add_argument('--no_wandb', action='store_true', help="Disable wandb logging.")
    parser.add_argument('--wandb_project', type=str, default="sensor-dropout", help="Wandb project name.")
    return parser.parse_args()

def main(argv=None):
    """Main training function."""
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args()
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(argv, config_path)

    # Use 'spawn' for multiprocessing
    mp.set_start_method("spawn", force=True)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Run training
    print(f"Starting PPO Dropout Any training on {config.task} with {config.num_envs} environments")
    print(f"Training for {config.total_timesteps} total timesteps")
    if config.use_wandb:
        print(f"Wandb logging enabled - project: {config.wandb_project}")
    else:
        print("Wandb logging disabled")
    
    trained_agent = train_ppo_dropout_any(None, config, seed)
    
    print("Training completed!")
    return trained_agent

if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds") 