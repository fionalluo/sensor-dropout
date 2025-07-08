#!/usr/bin/env python3
"""
Multi-teacher to single student distillation using SB3.
Trains a student policy by cycling through multiple teacher policies,
using pure distillation loss without any RL components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import wandb
import os
import re
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from baselines.shared.masking_utils import mask_observations_for_student
from subset_policies_sb3.load_subset_policy_sb3 import SubsetPolicyLoader


class TeacherPolicyManager:
    """Manager for loading and interfacing with multiple teacher policies."""
    
    def __init__(self, expert_policy_dir: str, teacher_keys_by_config: Dict, device: str = 'cpu', debug: bool = False):
        self.expert_policy_dir = expert_policy_dir
        self.teacher_keys_by_config = teacher_keys_by_config
        self.device = device
        self.debug = debug
        self.policy_loader = SubsetPolicyLoader(expert_policy_dir, device)
        self.teacher_configs = list(teacher_keys_by_config.keys())
        
        print(f"Initialized TeacherPolicyManager with {len(self.teacher_configs)} teachers: {self.teacher_configs}")
    
    def get_teacher_action_logits(self, config_name: str, full_obs: Dict) -> torch.Tensor:
        # Load teacher policy
        agent, _, eval_keys = self.policy_loader.load_policy(config_name)
        
        # Get teacher keys from config (what this teacher was trained with)
        teacher_keys = self.teacher_keys_by_config[config_name]
        
        # Convert full_obs to numpy format first
        full_obs_numpy = {}
        for key, value in full_obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
                
            # Convert to numpy for SB3 teacher policy (SB3 expects numpy arrays)
            if isinstance(value, torch.Tensor):
                full_obs_numpy[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                full_obs_numpy[key] = value
            else:
                full_obs_numpy[key] = np.array(value)
        
        # Parse teacher keys from patterns to get the exact keys this teacher expects
        teacher_mlp_keys, teacher_cnn_keys = self._parse_teacher_keys(teacher_keys, full_obs_numpy.keys())
        all_teacher_keys = teacher_mlp_keys + teacher_cnn_keys
        
        # Filter observation to ONLY include the keys the teacher was trained with
        teacher_obs = {}
        for key in all_teacher_keys:
            if key in full_obs_numpy:
                teacher_obs[key] = full_obs_numpy[key]
        
        if self.debug:
            print(f"[DEBUG] Teacher {config_name}:")
            print(f"  Available keys: {list(full_obs.keys())}")
            print(f"  Teacher eval_keys: {teacher_keys}")
            print(f"  Parsed teacher keys: {all_teacher_keys}")
            print(f"  Filtered obs keys: {list(teacher_obs.keys())}")
            print(f"  Teacher policy obs space: {list(agent.policy.observation_space.spaces.keys())}")
        
        # Verify we have all required keys that the teacher policy expects
        missing_keys = []
        for expected_key in agent.policy.observation_space.spaces.keys():
            if expected_key not in teacher_obs:
                missing_keys.append(expected_key)
        
        if missing_keys:
            if self.debug:
                print(f"  Warning: Missing keys {missing_keys}, creating dummy values")
            # Add dummy values for missing keys
            for missing_key in missing_keys:
                expected_space = agent.policy.observation_space.spaces[missing_key]
                teacher_obs[missing_key] = np.zeros(expected_space.shape, dtype=expected_space.dtype)
        
        # Add batch dimension for SB3
        obs_batch = {k: np.expand_dims(v, axis=0) for k, v in teacher_obs.items()}
        
        if self.debug:
            print(f"  Final obs batch keys: {list(obs_batch.keys())}")
            print(f"  Obs batch shapes: {dict((k, v.shape) for k, v in obs_batch.items())}")
        
        # Get action logits from teacher policy
        with torch.no_grad():
            # For SB3, get the action distribution
            obs_tensor_flat = agent.policy.obs_to_tensor(obs_batch)[0]
            distribution = agent.policy.get_distribution(obs_tensor_flat)
            
            if hasattr(distribution, 'distribution'):
                # For discrete actions, get logits
                if hasattr(distribution.distribution, 'logits'):
                    logits = distribution.distribution.logits
                    # Return raw logits on correct device
                    if self.debug:
                        print(f"  Action logits shape: {logits.shape}")
                    return logits.to(self.device).float()
                # For continuous actions, get mean
                elif hasattr(distribution.distribution, 'mean'):
                    mean = distribution.distribution.mean
                    return mean.to(self.device).float()
            
            # Fallback: if we can't get logits, create uniform distribution
            action_space_size = getattr(agent.policy.action_space, 'n', 2)
            if self.debug:
                print(f"  Fallback: creating uniform logits with {action_space_size} actions")
            return torch.zeros(action_space_size, device=self.device, dtype=torch.float32)
    
    def _parse_teacher_keys(self, teacher_keys, available_keys: List[str]) -> Tuple[List[str], List[str]]:
        """Parse teacher keys from patterns."""
        mlp_keys = []
        cnn_keys = []
        
        # Handle both dict and SimpleNamespace objects
        if hasattr(teacher_keys, 'mlp_keys'):
            mlp_pattern = re.compile(teacher_keys.mlp_keys)
            mlp_keys = [k for k in available_keys if mlp_pattern.search(k)]
        elif isinstance(teacher_keys, dict) and 'mlp_keys' in teacher_keys:
            mlp_pattern = re.compile(teacher_keys['mlp_keys'])
            mlp_keys = [k for k in available_keys if mlp_pattern.search(k)]
        
        if hasattr(teacher_keys, 'cnn_keys'):
            cnn_pattern = re.compile(teacher_keys.cnn_keys)
            cnn_keys = [k for k in available_keys if cnn_pattern.search(k)]
        elif isinstance(teacher_keys, dict) and 'cnn_keys' in teacher_keys:
            cnn_pattern = re.compile(teacher_keys['cnn_keys'])
            cnn_keys = [k for k in available_keys if cnn_pattern.search(k)]
        
        return mlp_keys, cnn_keys


class EpisodeTeacherCycler:
    """Manages cycling through teachers for each episode."""
    
    def __init__(self, teacher_configs: List[str], num_envs: int):
        self.teacher_configs = teacher_configs
        self.num_envs = num_envs
        self.num_teachers = len(teacher_configs)
        
        # Initialize each environment with a different teacher offset
        self.current_teacher_indices = [(i % self.num_teachers) for i in range(num_envs)]
        self.episode_counts = [0] * num_envs
        
        print(f"Initialized EpisodeTeacherCycler with {self.num_teachers} teachers for {num_envs} envs")
        print(f"Initial teacher assignments: {[self.teacher_configs[idx] for idx in self.current_teacher_indices]}")
    
    def get_current_teachers(self) -> List[str]:
        """Get current teacher config for each environment."""
        return [self.teacher_configs[idx] for idx in self.current_teacher_indices]
    
    def cycle_teacher(self, env_idx: int):
        """Cycle to next teacher for a specific environment."""
        self.current_teacher_indices[env_idx] = (self.current_teacher_indices[env_idx] + 1) % self.num_teachers
        self.episode_counts[env_idx] += 1
        
        new_teacher = self.teacher_configs[self.current_teacher_indices[env_idx]]
        print(f"Environment {env_idx} episode {self.episode_counts[env_idx]}: switched to teacher {new_teacher}")


class MultiTeacherEnvironmentWrapper(gym.Wrapper):
    """Environment wrapper that handles observation masking for multi-teacher distillation."""
    
    def __init__(self, env, student_keys: List[str], teacher_keys_by_config: Dict, teacher_cycler: EpisodeTeacherCycler, env_idx: int):
        super().__init__(env)
        self.student_keys = student_keys
        self.teacher_keys_by_config = teacher_keys_by_config
        self.teacher_cycler = teacher_cycler
        self.env_idx = env_idx
        self.episode_start = True
        self.current_teacher_config = None
        self.last_full_obs = None
        
        # Set observation space to only include student keys
        self._setup_student_observation_space()
    
    def _setup_student_observation_space(self):
        """Set up observation space to only include student keys."""
        # Get a sample observation to determine spaces
        sample_obs, _ = self.env.reset()
        self.env.reset()  # Reset again to maintain proper state
        
        print(f"Available observation keys: {list(sample_obs.keys())}")
        print(f"Student keys: {self.student_keys}")
        
        # Create filtered observation space with only student keys
        filtered_spaces = {}
        for key in self.student_keys:
            if key in sample_obs:
                original_space = self.env.observation_space.spaces[key]
                
                # Check if this is a problematic CNN space (too small dimensions)
                if isinstance(original_space, gym.spaces.Box) and len(original_space.shape) >= 2:
                    height, width = original_space.shape[:2]
                    if height <= 4 or width <= 4:
                        print(f"Warning: CNN key '{key}' has small dimensions {original_space.shape}, may cause issues")
                        # Create a minimal valid CNN space
                        filtered_spaces[key] = gym.spaces.Box(
                            low=0.0, high=1.0, shape=(8, 8, 3), dtype=np.float32
                        )
                    else:
                        filtered_spaces[key] = original_space
                else:
                    filtered_spaces[key] = original_space
                    
                print(f"Including key '{key}' with space: {filtered_spaces[key]}")
            else:
                # If student key doesn't exist, create a dummy space
                # This will be handled by masking later
                filtered_spaces[key] = gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                )
                print(f"Creating dummy space for missing key '{key}'")
        
        # Ensure we have at least one observation key
        if not filtered_spaces:
            print("Warning: No student keys found, creating dummy observation space")
            filtered_spaces['dummy'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        
        # Set the observation space
        self.observation_space = gym.spaces.Dict(filtered_spaces)
        
        print(f"Final student observation space for env {self.env_idx}: {list(filtered_spaces.keys())}")
    
    def reset(self, **kwargs):
        """Reset environment and cycle to next teacher."""
        obs, info = self.env.reset(**kwargs)
        
        # Cycle to next teacher for new episode
        if not self.episode_start:  # Don't cycle on very first reset
            self.teacher_cycler.cycle_teacher(self.env_idx)
        
        self.episode_start = False
        self.current_teacher_config = self.teacher_cycler.get_current_teachers()[self.env_idx]
        
        # Store full observation for teacher action computation
        self.last_full_obs = obs.copy()
        
        # Mask observation for student
        masked_obs = self._mask_observation_for_student(obs)
        
        return masked_obs, info
    
    def step(self, action):
        """Step environment and mask observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store full observation for teacher action computation
        self.last_full_obs = obs.copy()
        
        # Mask observation for student
        masked_obs = self._mask_observation_for_student(obs)
        
        # If episode is done, mark for teacher cycling on next reset
        if terminated or truncated:
            self.episode_start = True
        
        # Set reward to 0 for pure distillation (no RL)
        reward = 0.0
        
        return masked_obs, reward, terminated, truncated, info
    
    def _mask_observation_for_student(self, obs: Dict) -> Dict:
        """Mask observations for student training."""
        # Get current teacher keys
        teacher_keys = self.teacher_keys_by_config[self.current_teacher_config]
        
        # Parse teacher keys from patterns
        teacher_mlp_keys, teacher_cnn_keys = self._parse_keys_from_patterns(teacher_keys, obs.keys())
        all_teacher_keys = teacher_mlp_keys + teacher_cnn_keys
        
        # Convert observations to tensors for masking
        obs_tensors = {}
        for key, value in obs.items():
            if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
        
        # Apply masking
        masked_obs = mask_observations_for_student(
            obs_tensors,
            self.student_keys,
            all_teacher_keys,
            device=None,
            debug=False
        )
        
        # Convert back to numpy and ensure consistency with observation space
        masked_obs_numpy = {}
        for key in self.observation_space.spaces.keys():
            if key in masked_obs:
                value = masked_obs[key]
                if isinstance(value, torch.Tensor):
                    value_numpy = value.cpu().numpy()
                else:
                    value_numpy = np.array(value)
                
                # Ensure the shape matches the observation space
                expected_shape = self.observation_space.spaces[key].shape
                if value_numpy.shape != expected_shape:
                    if key == 'dummy' or expected_shape == (1,):
                        # For dummy keys, create appropriate dummy data
                        value_numpy = np.zeros(expected_shape, dtype=np.float32)
                    elif len(expected_shape) >= 2 and expected_shape[:2] == (8, 8):
                        # For modified CNN spaces, create dummy image data
                        value_numpy = np.zeros(expected_shape, dtype=np.float32)
                    else:
                        # Try to reshape if possible
                        try:
                            value_numpy = value_numpy.reshape(expected_shape)
                        except:
                            # Fall back to zeros
                            value_numpy = np.zeros(expected_shape, dtype=np.float32)
                
                masked_obs_numpy[key] = value_numpy
            else:
                # Create dummy data if key is missing
                expected_shape = self.observation_space.spaces[key].shape
                masked_obs_numpy[key] = np.zeros(expected_shape, dtype=np.float32)
        
        return masked_obs_numpy
    
    def _parse_keys_from_patterns(self, key_patterns, available_keys: List[str]) -> Tuple[List[str], List[str]]:
        """Parse keys from regex patterns."""
        mlp_keys = []
        cnn_keys = []
        
        # Handle both dict and SimpleNamespace objects
        if hasattr(key_patterns, 'mlp_keys'):
            mlp_pattern = re.compile(key_patterns.mlp_keys)
            mlp_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and mlp_pattern.search(k)]
        elif isinstance(key_patterns, dict) and 'mlp_keys' in key_patterns:
            mlp_pattern = re.compile(key_patterns['mlp_keys'])
            mlp_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and mlp_pattern.search(k)]
        
        if hasattr(key_patterns, 'cnn_keys'):
            cnn_pattern = re.compile(key_patterns.cnn_keys)
            cnn_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and cnn_pattern.search(k)]
        elif isinstance(key_patterns, dict) and 'cnn_keys' in key_patterns:
            cnn_pattern = re.compile(key_patterns['cnn_keys'])
            cnn_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and cnn_pattern.search(k)]
        
        return mlp_keys, cnn_keys
    
    def get_last_full_observation(self) -> Dict:
        """Get the last full observation for teacher action computation."""
        return self.last_full_obs.copy() if self.last_full_obs is not None else {}
    
    def get_current_teacher_config(self) -> str:
        """Get current teacher configuration name."""
        return self.current_teacher_config


class MultiTeacherDistillationTrainer:
    """Main trainer for multi-teacher to single student distillation."""
    
    def __init__(self, config, expert_policy_dir: str, device: str = 'cpu', debug: bool = False):
        """
        Initialize the multi-teacher distillation trainer.
        
        Args:
            config: Training configuration
            expert_policy_dir: Directory containing expert policies
            device: Device for training
            debug: Enable debug mode
        """
        self.config = config
        self.expert_policy_dir = expert_policy_dir
        self.device = device
        self.debug = debug
        
        # Parse student keys
        self.student_keys = self._parse_student_keys(config)
        
        # Parse teacher keys by config
        self.teacher_keys_by_config = {}
        for i in range(1, config.num_eval_configs + 1):
            config_name = f'env{i}'
            if hasattr(config.eval_keys, config_name):
                self.teacher_keys_by_config[config_name] = getattr(config.eval_keys, config_name)
        
        # Initialize teacher manager
        self.teacher_manager = TeacherPolicyManager(
            expert_policy_dir, 
            self.teacher_keys_by_config, 
            device,
            debug=debug
        )
        
        # Initialize teacher cycler
        self.teacher_cycler = EpisodeTeacherCycler(
            list(self.teacher_keys_by_config.keys()),
            config.num_envs
        )
        
        # Initialize environments
        self.envs = self._create_environments()
        
        # Initialize student policy
        self.student_policy = self._create_student_policy()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.student_policy.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Training metrics
        self.global_step = 0
        self.episode_count = 0
        self.total_distillation_loss = 0.0
        self.loss_history = deque(maxlen=100)
        
        print(f"Initialized MultiTeacherDistillationTrainer")
        print(f"Student keys: {self.student_keys}")
        print(f"Teacher configs: {list(self.teacher_keys_by_config.keys())}")
        print(f"Device: {self.device}")
        print(f"Debug mode: {self.debug}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Distillation loss weight: {self.config.distillation_loss_weight}")
        
        # Show environment observation space
        if hasattr(self.envs, 'observation_space'):
            print(f"Environment observation space keys: {list(self.envs.observation_space.spaces.keys())}")
        else:
            print("Warning: Could not access environment observation space")
    
    def _parse_student_keys(self, config) -> List[str]:
        """Parse student keys from config patterns."""
        # Create a temporary environment to get available keys
        temp_env = gym.make(config.task.split('_', 1)[1])
        sample_obs, _ = temp_env.reset()
        temp_env.close()
        
        available_keys = [k for k in sample_obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
        
        student_keys = []
        if hasattr(config.keys, 'mlp_keys'):
            mlp_pattern = re.compile(config.keys.mlp_keys)
            student_keys.extend([k for k in available_keys if mlp_pattern.search(k)])
        
        if hasattr(config.keys, 'cnn_keys'):
            cnn_pattern = re.compile(config.keys.cnn_keys)
            student_keys.extend([k for k in available_keys if cnn_pattern.search(k)])
        
        return student_keys
    
    def _create_environments(self):
        """Create vectorized environments with multi-teacher wrappers."""
        def make_env(env_idx):
            def _init():
                # Create base environment
                suite, task = self.config.task.split('_', 1)
                env = gym.make(task)
                
                # Wrap with multi-teacher wrapper
                env = MultiTeacherEnvironmentWrapper(
                    env,
                    self.student_keys,
                    self.teacher_keys_by_config,
                    self.teacher_cycler,
                    env_idx
                )
                
                return env
            return _init
        
        # Create vectorized environments
        if self.config.num_envs == 1:
            envs = DummyVecEnv([make_env(0)])
        else:
            envs = SubprocVecEnv([make_env(i) for i in range(self.config.num_envs)])
        
        return envs
    
    def _create_student_policy(self):
        """Create SB3 student policy."""
        # Create a dummy environment to get observation space
        base_env = gym.make(self.config.task.split('_', 1)[1])
        temp_env = MultiTeacherEnvironmentWrapper(
            base_env,
            self.student_keys,
            self.teacher_keys_by_config,
            self.teacher_cycler,
            0
        )
        
        print(f"Creating student policy with observation space: {temp_env.observation_space}")
        
        # Create SB3 PPO policy (we'll train it manually)
        student_policy = PPO(
            "MultiInputPolicy",
            temp_env,
            learning_rate=self.config.learning_rate,
            device=self.device,
            verbose=0
        )
        
        temp_env.close()
        return student_policy
    
    def train(self):
        """Main training loop."""
        print(f"Starting multi-teacher distillation training for {self.config.total_timesteps} timesteps")
        print(f"Steps per rollout: {self.config.steps_per_rollout}")
        print(f"Number of environments: {self.config.num_envs}")
        print(f"Steps per rollout iteration: {self.config.steps_per_rollout * self.config.num_envs}")
        
        obs = self.envs.reset()
        rollout_count = 0
        
        while self.global_step < self.config.total_timesteps:
            rollout_count += 1
            print(f"\n--- Rollout {rollout_count} (Steps: {self.global_step}/{self.config.total_timesteps}) ---")
            
            # Collect rollout
            print("Collecting rollout...")
            rollout_data = self._collect_rollout(obs)
            
            # Train on collected data
            print("Training student...")
            distillation_loss = self._train_step(rollout_data)
            
            # Update metrics
            self.total_distillation_loss += distillation_loss
            self.loss_history.append(distillation_loss)
            
            # Logging - log every rollout for now
            self._log_training_metrics(distillation_loss)
            
            # Evaluation
            if self.global_step % self.config.eval_freq == 0:
                self._evaluate()
            
            # Update observation for next rollout
            obs = rollout_data['next_obs']
        
        print("Training completed!")
        return self.student_policy
    
    def _collect_rollout(self, initial_obs):
        """Collect a rollout of experience."""
        observations = []
        full_observations = []
        teacher_configs = []
        student_actions = []
        teacher_logits = []
        next_obs = initial_obs
        
        # Collect steps
        for step in range(self.config.steps_per_rollout):
            if step % 100 == 0:  # Print every 100 steps
                print(f"  Rollout step {step}/{self.config.steps_per_rollout}")
            # Get current full observations and teacher configs
            current_full_obs = []
            current_teacher_configs = []
            
            for env_idx in range(self.config.num_envs):
                # Use env_method to call wrapper methods on individual environments
                full_obs = self.envs.env_method('get_last_full_observation', indices=[env_idx])[0]
                teacher_config = self.envs.env_method('get_current_teacher_config', indices=[env_idx])[0]
                current_full_obs.append(full_obs)
                current_teacher_configs.append(teacher_config)
            
            # Get student actions
            with torch.no_grad():
                student_actions_step, _ = self.student_policy.predict(next_obs, deterministic=False)
            
            # Get teacher action logits for each environment
            teacher_logits_step = []
            for env_idx in range(self.config.num_envs):
                if current_full_obs[env_idx] and current_teacher_configs[env_idx]:
                    # Convert full obs to tensors
                    full_obs_tensors = {}
                    for key, value in current_full_obs[env_idx].items():
                        if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                            if isinstance(value, np.ndarray):
                                full_obs_tensors[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
                            else:
                                full_obs_tensors[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
                    
                    # Get teacher logits
                    teacher_logits_env = self.teacher_manager.get_teacher_action_logits(
                        current_teacher_configs[env_idx], 
                        full_obs_tensors
                    )
                    teacher_logits_step.append(teacher_logits_env)
                    
                    # Debug: Print teacher action info for first environment in first step
                    if self.debug and step == 0 and env_idx == 0:
                        teacher_probs = F.softmax(teacher_logits_env, dim=-1)
                        teacher_action = torch.argmax(teacher_probs).item()
                        print(f"    Env {env_idx}, Teacher {current_teacher_configs[env_idx]}:")
                        print(f"      Full obs keys: {list(current_full_obs[env_idx].keys())}")
                        print(f"      Teacher action probs: {teacher_probs.cpu().numpy()}")
                        print(f"      Teacher preferred action: {teacher_action}")
                        print(f"      Student obs keys: {list(next_obs[env_idx].keys())}")
                else:
                    # Dummy logits if no valid teacher
                    action_space_size = self.envs.action_space.n
                    teacher_logits_step.append(torch.zeros(action_space_size, device=self.device))
                    if self.debug and step == 0 and env_idx == 0:
                        print(f"    Env {env_idx}: No valid teacher or observation")
            
            # Store data
            observations.append(next_obs)
            full_observations.append(current_full_obs)
            teacher_configs.append(current_teacher_configs)
            student_actions.append(student_actions_step)
            teacher_logits.append(torch.stack(teacher_logits_step))
            
            # Step environments
            next_obs, rewards, dones, infos = self.envs.step(student_actions_step)
            self.global_step += self.config.num_envs
        
        return {
            'observations': observations,
            'full_observations': full_observations,
            'teacher_configs': teacher_configs,
            'student_actions': student_actions,
            'teacher_logits': teacher_logits,
            'next_obs': next_obs
        }
    
    def _train_step(self, rollout_data):
        """Train student on collected rollout using minibatch distillation loss."""
        total_loss = 0.0
        num_updates = 0
        
        # Extract data
        observations = rollout_data['observations']
        teacher_logits = rollout_data['teacher_logits']
        
        print(f"  Training on {len(observations)} rollout steps")
        
        # Flatten observations and teacher logits for minibatch processing
        flat_obs = []
        flat_teacher_logits = []
        
        for step_idx in range(len(observations)):
            for env_idx in range(self.config.num_envs):
                # Get observation for this step and environment
                obs_dict = {}
                for key in observations[step_idx].keys():
                    obs_dict[key] = observations[step_idx][key][env_idx]
                
                flat_obs.append(obs_dict)
                flat_teacher_logits.append(teacher_logits[step_idx][env_idx])
        
        total_samples = len(flat_obs)
        print(f"  Total samples: {total_samples}")
        
        # Get training parameters from config
        num_minibatches = getattr(self.config, 'num_minibatches', 4)
        update_epochs = getattr(self.config, 'update_epochs', 4)
        temperature = getattr(self.config, 'temperature', 1.0)
        
        # Ensure we can create evenly sized minibatches
        assert total_samples % num_minibatches == 0, f"Total samples {total_samples} must be divisible by num_minibatches {num_minibatches}"
        minibatch_size = total_samples // num_minibatches
        
        print(f"  Using {num_minibatches} minibatches of size {minibatch_size}, {update_epochs} epochs")
        
        # Multiple epochs over the data (like original PPO distill)
        for epoch in range(update_epochs):
            # Shuffle indices for this epoch
            indices = np.random.permutation(total_samples)
            
            # Process minibatches
            for start in range(0, total_samples, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Create minibatch
                mb_obs = [flat_obs[idx] for idx in mb_indices]
                mb_teacher_logits = torch.stack([flat_teacher_logits[idx] for idx in mb_indices])
                
                # Convert observations to tensor format
                mb_obs_tensor = self._convert_obs_to_tensor(mb_obs)
                
                # Get student logits
                with torch.enable_grad():
                    obs_tensor = self.student_policy.policy.obs_to_tensor(mb_obs_tensor)[0]
                    distribution = self.student_policy.policy.get_distribution(obs_tensor)
                    student_logits = distribution.distribution.logits
                    
                    # Debug: Print logits statistics for first minibatch
                    if num_updates == 0 and self.debug:
                        print(f"    Student logits: min={student_logits.min().item():.3f}, max={student_logits.max().item():.3f}, mean={student_logits.mean().item():.3f}")
                        print(f"    Teacher logits: min={mb_teacher_logits.min().item():.3f}, max={mb_teacher_logits.max().item():.3f}, mean={mb_teacher_logits.mean().item():.3f}")
                        print(f"    Temperature: {temperature}")
                    
                    # Apply temperature scaling (now configurable)
                    if temperature != 1.0:
                        student_logits_scaled = student_logits / temperature
                        teacher_logits_scaled = mb_teacher_logits / temperature
                    else:
                        student_logits_scaled = student_logits
                        teacher_logits_scaled = mb_teacher_logits
                    
                    # Compute distillation loss (KL divergence)
                    student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
                    teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
                    
                    # Compute loss
                    distillation_loss = F.kl_div(
                        student_log_probs, 
                        teacher_probs, 
                        reduction='batchmean'
                    )
                    
                    # Apply temperature scaling to loss if used
                    if temperature != 1.0:
                        distillation_loss = distillation_loss * (temperature ** 2)
                    
                    # Apply distillation weight
                    distillation_loss = distillation_loss * self.config.distillation_loss_weight
                    
                    # Debug: Print loss components for first minibatch
                    if num_updates == 0 and self.debug:
                        raw_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                        print(f"    Raw KL divergence: {raw_kl.item():.6f}")
                        print(f"    Final loss: {distillation_loss.item():.6f}")
                        teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=-1).mean()
                        print(f"    Teacher prob entropy: {teacher_entropy.item():.3f}")
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    distillation_loss.backward()
                    self.optimizer.step()
                    
                    total_loss += distillation_loss.item()
                    num_updates += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        print(f"  Completed {num_updates} training updates over {update_epochs} epochs, avg loss: {avg_loss:.6f}")
        return avg_loss
    
    def _convert_obs_to_tensor(self, obs_list):
        """Convert list of observation dicts to tensor dict."""
        if len(obs_list) == 0:
            return {}
        
        tensor_dict = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            tensor_dict[key] = np.array(values)
        
        return tensor_dict
    
    def _log_training_metrics(self, current_loss):
        """Log training metrics."""
        avg_loss = np.mean(self.loss_history) if self.loss_history else 0.0
        
        print(f"Step {self.global_step}/{self.config.total_timesteps}: Distillation Loss = {current_loss:.6f}, Avg Loss = {avg_loss:.6f}")
        
        # Log to wandb if available
        if hasattr(self.config, 'track') and self.config.track and wandb.run is not None:
            wandb.log({
                'train/distillation_loss': current_loss,
                'train/avg_distillation_loss': avg_loss,
                'train/global_step': self.global_step
            })
        elif self.global_step == self.config.num_envs * self.config.steps_per_rollout:  # First rollout
            print("Note: Wandb logging not available")
    
    def _evaluate(self):
        """Evaluate student policy."""
        print(f"Evaluating at step {self.global_step}...")
        
        # Simple evaluation - just log current training loss
        avg_loss = np.mean(self.loss_history) if self.loss_history else 0.0
        
        if self.config.track and wandb.run is not None:
            wandb.log({
                'eval/avg_distillation_loss': avg_loss,
                'eval/global_step': self.global_step
            })
        
        print(f"Evaluation complete. Avg loss: {avg_loss:.6f}")


def train_multi_teacher_distillation(config, expert_policy_dir: str, device: str = 'cpu', debug: bool = False):
    """
    Main training function for multi-teacher distillation.
    
    Args:
        config: Training configuration
        expert_policy_dir: Directory containing expert policies
        device: Device for training
        debug: Enable debug mode
        
    Returns:
        Trained student policy
    """
    # Initialize wandb if tracking enabled
    if hasattr(config, 'track') and config.track:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"{config.exp_name}_{config.seed}",
            config=vars(config),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )
        print(f"Wandb initialized for project: {config.wandb_project}")
    else:
        print("Wandb tracking disabled")
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Create trainer
    trainer = MultiTeacherDistillationTrainer(config, expert_policy_dir, device, debug=debug)
    
    # Train
    trained_policy = trainer.train()
    
    # Close wandb
    if config.track and wandb.run is not None:
        wandb.finish()
    
    return trained_policy 