#!/usr/bin/env python3
"""
Pure Distillation Training using SB3
Trains a student policy purely through imitation from expert policies, no RL components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import time
from collections import deque
import torch.optim as optim
import wandb
import os
import re
import warnings
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor

from baselines.shared.masking_utils import mask_observations_for_student
from baselines.shared.eval_utils_sb3 import CustomEvalCallback
from subset_policies_sb3.load_subset_policy_sb3 import SubsetPolicyLoader

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `reset\(\)` method was expecting numpy array dtype to be int32, actual type: int64.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `reset\(\)` method is not within the observation space.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `step\(\)` method was expecting numpy array dtype to be int32, actual type: int64.*')
warnings.filterwarnings('ignore', '.*The obs returned by the `step\(\)` method is not within the observation space.*')
warnings.filterwarnings('ignore', '.*Overriding environment.*')

def parse_keys_from_patterns(full_obs_space, mlp_pattern: str, cnn_pattern: str):
    import re
    
    mlp_regex = re.compile(mlp_pattern)
    cnn_regex = re.compile(cnn_pattern)
    
    matched_keys = []
    for key in full_obs_space.keys():
        if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
            continue
        if mlp_regex.search(key) or cnn_regex.search(key):
            matched_keys.append(key)
    
    return matched_keys


def get_full_observation_space(config):
    import gymnasium as gym
    
    suite, task = config.task.split('_', 1)
    temp_env = gym.make(task)
    full_obs, _ = temp_env.reset()
    temp_env.close()
    
    return full_obs


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
            if self.cnn_pattern.search(key) or self.mlp_pattern.search(key):
                filtered_spaces[key] = space
        
        self.observation_space = gym.spaces.Dict(filtered_spaces)
    
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


class DistillationRolloutBuffer(DictRolloutBuffer):
    """
    Custom rollout buffer that stores both full and filtered observations for distillation.
    """
    
    def __init__(self, *args, student_mlp_keys='.*', student_cnn_keys='.*', **kwargs):
        # Initialize custom attributes BEFORE parent class (which calls reset())
        self.student_mlp_keys = student_mlp_keys
        self.student_cnn_keys = student_cnn_keys
        self.full_observations = None
        self.expert_configs = None  # Store which expert config was active for each step
        self.env_wrapper = None  # Will be set by the environment wrapper
        
        # Now call parent constructor (which will call reset())
        super().__init__(*args, **kwargs)
        
    def set_env_wrapper(self, env_wrapper):
        """Set reference to environment wrapper for accessing full observations."""
        self.env_wrapper = env_wrapper
        
    def reset(self):
        """Reset the buffer and initialize full observation storage."""
        super().reset()
        if self.full_observations is not None:
            # Reset full observations storage
            for key in self.full_observations:
                self.full_observations[key].fill(0)
        
        if self.expert_configs is not None:
            # Reset expert configs storage
            self.expert_configs.fill('')
        
    def add(self, *args, **kwargs):
        """
        Override add to store full observations alongside filtered ones.
        
        IMPORTANT DATA FLOW:
        - rollout_buffer.observations: Masked observations that student actually saw during training
        - rollout_buffer.full_observations: Unmasked observations from environment (for expert targets)
        - rollout_buffer.expert_configs: Which expert config was active for each step
        """
        # Call parent add method with filtered observations
        super().add(*args, **kwargs)
        
        # Store full observations if wrapper is available
        if self.env_wrapper is not None and hasattr(self.env_wrapper, 'get_last_full_observations'):
            full_obs_list = self.env_wrapper.get_last_full_observations()
            
            if self.full_observations is None:
                # Initialize full observation storage on first add
                self._init_full_observation_storage(full_obs_list[0])
                # Also initialize expert config storage
                self._init_expert_config_storage()
            
            # Store full observations for this step
            for env_idx, full_obs in enumerate(full_obs_list):
                for key, value in full_obs.items():
                    if key in self.full_observations:
                        # Convert to numpy if needed
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        self.full_observations[key][self.pos - 1, env_idx] = value
            
            # Store expert configs directly from environment wrapper
            if hasattr(self.env_wrapper, 'get_current_teacher_config_names'):
                current_configs = self.env_wrapper.get_current_teacher_config_names()
                
                # Store configs for each environment
                step_config_summary = {}
                for env_idx, config_name in enumerate(current_configs):
                    if config_name and self.expert_configs is not None:
                        self.expert_configs[self.pos - 1, env_idx] = config_name
                        step_config_summary[config_name] = step_config_summary.get(config_name, 0) + 1
    
    def add_expert_config(self, expert_config_name):
        """Store which expert configuration was active for the current step."""
        if self.expert_configs is not None:
            # Store the expert config for all environments at this step
            for env_idx in range(self.n_envs):
                self.expert_configs[self.pos - 1, env_idx] = expert_config_name
    
    def _init_full_observation_storage(self, sample_obs):
        """Initialize storage for full observations."""
        self.full_observations = {}
        for key, value in sample_obs.items():
            if isinstance(value, torch.Tensor):
                obs_shape = value.shape
                dtype = value.dtype
                if dtype == torch.float64:
                    dtype = torch.float32
                self.full_observations[key] = np.zeros(
                    (self.buffer_size, self.n_envs) + obs_shape, 
                    dtype=np.float32
                )
            elif isinstance(value, np.ndarray):
                obs_shape = value.shape
                self.full_observations[key] = np.zeros(
                    (self.buffer_size, self.n_envs) + obs_shape, 
                    dtype=np.float32
                )
            else:
                # Scalar value
                self.full_observations[key] = np.zeros(
                    (self.buffer_size, self.n_envs), 
                    dtype=np.float32
                )
    
    def _init_expert_config_storage(self):
        """Initialize storage for expert configurations."""
        # Use object array to store strings
        self.expert_configs = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.expert_configs.fill('')  # Initialize with empty strings
    
    def get_full_observations(self, indices=None):
        if self.full_observations is None:
            return {}
            
        if indices is None:
            return self.full_observations
        
        step_indices, env_indices = indices
        full_obs = {}
        for key, value in self.full_observations.items():
            full_obs[key] = value[step_indices, env_indices]
        
        return full_obs
    
    def get_expert_configs(self, indices=None):
        if self.expert_configs is None:
            return np.array([])
            
        if indices is None:
            return self.expert_configs
        
        step_indices, env_indices = indices
        return self.expert_configs[step_indices, env_indices]
    
    def get_filtered_observations(self, indices=None):
        if indices is None:
            return self.observations
            
        step_indices, env_indices = indices
        filtered_obs = {}
        for key, value in self.observations.items():
            filtered_obs[key] = value[step_indices, env_indices]
            
        return filtered_obs


class EpisodeMaskingWrapper(gym.ObservationWrapper):
    """Wrapper that cycles through teacher configurations for each episode and masks observations accordingly.
    This ensures the student learns on masked observations that simulate what each teacher sees."""
    
    def __init__(self, env, student_keys, teacher_keys_by_config):
        super().__init__(env)
        self.student_keys = student_keys
        self.teacher_keys_by_config = teacher_keys_by_config
        self.config_names = list(teacher_keys_by_config.keys())
        
        # Episode tracking
        self.episode_count = 0
        self.current_config_idx = 0
        self.episode_start = True
    
    def _get_current_teacher_config(self):
        """Get the current teacher configuration for this episode."""
        config_name = self.config_names[self.current_config_idx]
        teacher_keys = self.teacher_keys_by_config[config_name]
        return config_name, teacher_keys
    
    def _cycle_to_next_config(self):
        """Cycle to the next teacher configuration."""
        self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
        self.episode_count += 1
        
    def observation(self, obs):
        """Mask observations based on the current episode's teacher configuration."""
        if self.episode_start:
            # Get current teacher config for this episode
            current_config_name, current_teacher_keys = self._get_current_teacher_config()
            self.current_teacher_keys = current_teacher_keys
            self.current_config_name = current_config_name
            self.episode_start = False
                    
        # Convert observations to tensors for masking
        obs_tensors = {}
        for key, value in obs.items():
            if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
        
        # Apply masking to simulate what the teacher sees
        # Student gets masked observations (canonical keys) based on current teacher config
        masked_obs = mask_observations_for_student(
            obs_tensors, 
            self.student_keys,  # What the student expects (canonical names)
            self.current_teacher_keys,  # What's available in current teacher config
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
        """Reset the environment and prepare for a new episode."""
        obs, info = self.env.reset(**kwargs)
        
        # Mark that we're starting a new episode and cycle teacher config
        self.episode_start = True
        self._cycle_to_next_config()
        
        return self.observation(obs), info
    
    def step(self, action):
        """Step the environment and mask the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If episode is done, mark for new episode
        if terminated or truncated:
            self.episode_start = True
        
        # Apply terminal observation masking if present
        if "terminal_observation" in info:
            # Convert terminal observation to tensors
            terminal_obs_tensors = {}
            for key, value in info["terminal_observation"].items():
                if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    if isinstance(value, np.ndarray):
                        terminal_obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                    else:
                        terminal_obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
            
            # Apply masking to terminal observation
            masked_terminal_obs = mask_observations_for_student(
                terminal_obs_tensors,
                self.student_keys,
                self.current_teacher_keys,
                device=None,
                debug=False
            )
            
            # Convert back to numpy
            masked_terminal_obs_numpy = {}
            for key, value in masked_terminal_obs.items():
                if isinstance(value, torch.Tensor):
                    masked_terminal_obs_numpy[key] = value.cpu().numpy()
                else:
                    masked_terminal_obs_numpy[key] = np.array(value)
            
            info["terminal_observation"] = masked_terminal_obs_numpy
        
        return self.observation(obs), reward, terminated, truncated, info


class VectorizedDistillationWrapper(VecEnvWrapper):
    """
    Wrapper for vectorized environments to handle full observation storage and episode-level masking.
    """
    
    def __init__(self, vec_env, student_keys, all_required_keys, teacher_keys_by_config):
        self.student_keys = student_keys
        self.all_required_keys = all_required_keys
        self.teacher_keys_by_config = teacher_keys_by_config
        self.config_names = list(teacher_keys_by_config.keys())
        self.last_full_observations = []
        
        # Episode tracking for each environment - IMPORTANT: Each env cycles independently
        self.episode_counts = [0] * vec_env.num_envs
        # Initialize each environment with a different config offset for proper cycling
        num_configs = len(self.config_names)
        self.current_config_indices = [env_idx % num_configs for env_idx in range(vec_env.num_envs)]
        self.episode_starts = [True] * vec_env.num_envs
        self.current_teacher_configs = [None] * vec_env.num_envs
        
        # Get sample observation and filter to required keys only
        sample_obs = vec_env.reset()
        
        # Filter sample observation to only include required keys
        filtered_sample_obs = {}
        for key in sample_obs.keys():
            if key in self.all_required_keys:
                filtered_sample_obs[key] = sample_obs[key]
        
        # Create filtered observation space (only student keys for SB3)
        filtered_obs_space = self._create_filtered_observation_space(filtered_sample_obs, vec_env.observation_space)
        
        # Initialize parent class with filtered observation space
        super().__init__(vec_env, observation_space=filtered_obs_space)
    
    def _create_filtered_observation_space(self, sample_obs, original_obs_space):
        """Create observation space with only student keys."""
        # Create filtered observation space using pre-parsed student keys
        filtered_obs_space = {}
        for key in self.student_keys:
            if key in sample_obs and key in original_obs_space.spaces:
                filtered_obs_space[key] = original_obs_space.spaces[key]
        
        return gym.spaces.Dict(filtered_obs_space)
    
    def _get_current_teacher_config(self, env_idx):
        """Get the current teacher configuration for a specific environment."""
        config_name = self.config_names[self.current_config_indices[env_idx]]
        teacher_keys = self.teacher_keys_by_config[config_name]
        return config_name, teacher_keys
    
    def _cycle_to_next_config(self, env_idx):
        """Cycle to the next teacher configuration for a specific environment."""
        self.current_config_indices[env_idx] = (self.current_config_indices[env_idx] + 1) % len(self.config_names)
        self.episode_counts[env_idx] += 1
    
    def _filter_observations(self, full_obs_list, teacher_configs_for_step=None):
        """Apply episode-level masking to observations based on teacher configs that were active during the step."""
        masked_obs_list = []
        
        for env_idx, full_obs in enumerate(full_obs_list):
            # Handle episode start for this environment
            if self.episode_starts[env_idx]:
                # Get current teacher config for this episode
                current_config_name, current_teacher_keys = self._get_current_teacher_config(env_idx)
                self.current_teacher_configs[env_idx] = (current_config_name, current_teacher_keys)
                self.episode_starts[env_idx] = False

            # CRITICAL FIX: Use the teacher config that was active during this step
            if teacher_configs_for_step is not None:
                # Use the config that was passed in (active during the step)
                if env_idx < len(teacher_configs_for_step) and teacher_configs_for_step[env_idx]:
                    current_config_name = teacher_configs_for_step[env_idx]
                    current_teacher_keys = self.teacher_keys_by_config[current_config_name]
                else:
                    # Fallback to current config
                    current_config_name, current_teacher_keys = self.current_teacher_configs[env_idx]
            else:
                # Use current teacher config for this environment (original behavior)
                current_config_name, current_teacher_keys = self.current_teacher_configs[env_idx]
            
            # Convert observations to tensors for masking
            obs_tensors = {}
            for key, value in full_obs.items():
                if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    if isinstance(value, np.ndarray):
                        obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                    else:
                        obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
            
            # Apply masking to simulate what the teacher sees
            masked_obs = mask_observations_for_student(
                obs_tensors, 
                self.student_keys,  # What the student expects (canonical names)
                current_teacher_keys,  # What's available in current teacher config
                device=None,  # Use CPU for training
                debug=False
            )
            
            # Convert back to numpy for SB3
            masked_obs_numpy = {}
            for key, value in masked_obs.items():
                if isinstance(value, torch.Tensor):
                    masked_obs_numpy[key] = value.cpu().numpy()
                else:
                    masked_obs_numpy[key] = np.array(value)
            
            masked_obs_list.append(masked_obs_numpy)
        
        return masked_obs_list
    
    def reset(self):
        """Reset environment and store full observations."""
        full_obs = self.venv.reset()
        
        # Convert to list of dictionaries and filter to required keys only
        full_obs_list = []
        for env_idx in range(self.venv.num_envs):
            env_obs = {}
            # Only include keys that are actually required (student + all experts)
            for key, values in full_obs.items():
                if key in self.all_required_keys:
                    env_obs[key] = values[env_idx]
            full_obs_list.append(env_obs)
        
        self.last_full_observations = full_obs_list
        
        # Initialize each environment with its current config (they start with different offsets)
        config_summary = {}
        for env_idx in range(self.venv.num_envs):
            self.episode_starts[env_idx] = True
            # Don't cycle here - each env already has its offset config from __init__
            # Just set the current teacher config for this environment
            current_config_name, current_teacher_keys = self._get_current_teacher_config(env_idx)
            self.current_teacher_configs[env_idx] = (current_config_name, current_teacher_keys)
            
            # Track config distribution for logging
            config_summary[current_config_name] = config_summary.get(current_config_name, 0) + 1
                
        # Initialize last_teacher_configs for the first step
        self.last_teacher_configs = [config[0] if config else None for config in self.current_teacher_configs]
        
        # Return masked observations in vectorized format
        masked_obs_list = self._filter_observations(full_obs_list)
        
        # Convert back to vectorized format
        masked_obs = {}
        for key in masked_obs_list[0].keys():
            masked_obs[key] = np.array([obs[key] for obs in masked_obs_list])
            
        return masked_obs
    
    def step_async(self, actions):
        """Forward to wrapped environment."""
        return self.venv.step_async(actions)
    
    def step_wait(self):
        """Step environment and store full observations."""
        full_obs, rewards, dones, infos = self.venv.step_wait()
        
        # Convert to list of dictionaries and filter to required keys only
        full_obs_list = []
        for env_idx in range(self.venv.num_envs):
            env_obs = {}
            
            # Only include keys that are actually required (student + all experts)
            for key, values in full_obs.items():
                if key in self.all_required_keys:
                    env_obs[key] = values[env_idx]
            
            # Verify all required keys are present (they should be based on pattern matching)
            for required_key in self.all_required_keys:
                if required_key not in env_obs:
                    print(f"❌ ERROR: Required key '{required_key}' missing from environment observations!")
                    print(f"Available keys: {sorted(list(env_obs.keys()))}")
                    raise KeyError(f"Required observation key '{required_key}' not found in environment")
            
            full_obs_list.append(env_obs)
        
        self.last_full_observations = full_obs_list
        
        # CRITICAL FIX: Store teacher configs BEFORE they potentially change due to episode completion
        # These are the configs that were actually used to generate the observations
        self.last_teacher_configs = []
        for env_idx in range(self.venv.num_envs):
            if self.current_teacher_configs[env_idx]:
                self.last_teacher_configs.append(self.current_teacher_configs[env_idx][0])
            else:
                self.last_teacher_configs.append(None)
        
        # Handle episode completion and cycling configurations
        config_changes = {}
        for env_idx in range(self.venv.num_envs):
            if dones[env_idx]:
                old_config = self.current_teacher_configs[env_idx][0] if self.current_teacher_configs[env_idx] else "unknown"
                self.episode_starts[env_idx] = True
                self._cycle_to_next_config(env_idx)
                # Immediately set current teacher config for this environment
                current_config_name, current_teacher_keys = self._get_current_teacher_config(env_idx)
                self.current_teacher_configs[env_idx] = (current_config_name, current_teacher_keys)
                config_changes[env_idx] = f"{old_config}→{current_config_name}"
        # Return masked observations in vectorized format
        # CRITICAL FIX: Use the configs that were active during the step for masking
        masked_obs_list = self._filter_observations(full_obs_list, self.last_teacher_configs)
        
        # Convert back to vectorized format
        masked_obs = {}
        for key in masked_obs_list[0].keys():
            masked_obs[key] = np.array([obs[key] for obs in masked_obs_list])
        
        # Apply masking to terminal observations in infos
        masked_infos = []
        for env_idx, info in enumerate(infos):
            masked_info = info.copy()
            if "terminal_observation" in info:
                # Get current teacher config for this environment
                current_config_name, current_teacher_keys = self.current_teacher_configs[env_idx]
                
                # Convert terminal observation to tensors
                terminal_obs_tensors = {}
                for key, value in info["terminal_observation"].items():
                    if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                        if isinstance(value, np.ndarray):
                            terminal_obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                        else:
                            terminal_obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
                
                # Apply masking to terminal observation
                masked_terminal_obs = mask_observations_for_student(
                    terminal_obs_tensors,
                    self.student_keys,
                    current_teacher_keys,
                    device=None,
                    debug=False
                )
                
                # Convert back to numpy
                masked_terminal_obs_numpy = {}
                for key, value in masked_terminal_obs.items():
                    if isinstance(value, torch.Tensor):
                        masked_terminal_obs_numpy[key] = value.cpu().numpy()
                    else:
                        masked_terminal_obs_numpy[key] = np.array(value)
                
                masked_info["terminal_observation"] = masked_terminal_obs_numpy
            
            masked_infos.append(masked_info)
            
        return masked_obs, rewards, dones, masked_infos
    
    def get_last_full_observations(self):
        """Get the last full observations for buffer storage.""" 
        return self.last_full_observations
    
    def get_current_teacher_config_names(self):
        """Get the teacher configuration names that were active DURING the last step."""
        # Return the configs that were used to generate the last observations
        if hasattr(self, 'last_teacher_configs'):
            return self.last_teacher_configs
        else:
            # Fallback to current configs (shouldn't happen after first step)
            return [config[0] if config else None for config in self.current_teacher_configs]


class ExpertPolicyManager:
    """Manages loading and using expert policies for distillation."""
    
    def __init__(self, policy_dir: str, device: str = 'cpu'):
        """
        Initialize the expert policy manager.
        
        Args:
            policy_dir: Directory containing expert policies
            device: Device to load policies on
        """
        self.policy_dir = policy_dir
        self.device = device
        self.expert_policies = {}
        self.expert_eval_keys = {}
        
        # Load all expert policies using SB3 loader
        self._load_expert_policies()
    
    def _load_expert_policies(self):
        """Load all expert policies from the directory using SB3 loader."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Expert policy directory not found: {self.policy_dir}")
        
        # Use SB3 subset policy loader
        loader = SubsetPolicyLoader(self.policy_dir, device=self.device)
        
        # Store the loaded policies
        for subset_name in loader.policies.keys():
            agent, config, eval_keys = loader.load_policy(subset_name)
            self.expert_policies[subset_name] = agent
            self.expert_eval_keys[subset_name] = eval_keys
    
    def get_expert_action_logits(self, subset_name: str, obs: Dict) -> torch.Tensor:
        """
        Get action logits from a specific expert policy for distillation.
        Returns RAW LOGITS (not softmaxed) for KL divergence computation.
        """
        if subset_name not in self.expert_policies:
            raise ValueError(f"Expert policy {subset_name} not found")
        
        expert_agent = self.expert_policies[subset_name]

        # CRITICAL FIX: Don't filter observations - pass all keys to expert policy
        # The expert policy will handle its own internal filtering based on what it was trained with
        # We just need to exclude the special environment keys and convert to numpy
        obs_numpy = {}
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
                
            # Convert to numpy for SB3 expert policy (SB3 expects numpy arrays)
            if isinstance(value, torch.Tensor):
                obs_numpy[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                obs_numpy[key] = value
            else:
                obs_numpy[key] = np.array(value)

        # Get action logits from expert agent
        with torch.no_grad():
            # For SB3, we need to get the action logits from the policy
            if hasattr(expert_agent.policy, 'get_distribution'):
                # Get the policy's action distribution
                obs_tensor_flat = expert_agent.policy.obs_to_tensor(obs_numpy)[0]
                distribution = expert_agent.policy.get_distribution(obs_tensor_flat)
                if hasattr(distribution, 'distribution'):
                    # For discrete actions, get logits
                    if hasattr(distribution.distribution, 'logits'):
                        logits = distribution.distribution.logits
                        # CRITICAL FIX: Ensure logits are on the correct device and return RAW logits
                        return logits.to(self.device).float()
                    # For continuous actions, get mean
                    elif hasattr(distribution.distribution, 'mean'):
                        mean = distribution.distribution.mean
                        return mean.to(self.device).float()
                # If we can't get logits, create a fallback
                action_space_size = getattr(expert_agent.policy.action_space, 'n', 2)
                return torch.zeros(action_space_size, device=self.device, dtype=torch.float32)
            else:
                # Fallback: get action and convert to high-confidence logits if discrete
                action, _ = expert_agent.predict(obs_numpy, deterministic=True)
                
                # Check if action space is discrete
                if hasattr(expert_agent.policy.action_space, 'n'):
                    action_space_size = expert_agent.policy.action_space.n
                    # Create high-confidence logits (not one-hot)
                    action_logits = torch.full((action_space_size,), -10.0, device=self.device, dtype=torch.float32)
                    if isinstance(action, (int, np.integer)):
                        action_logits[action] = 10.0  # High confidence for chosen action
                    elif isinstance(action, np.ndarray) and action.size == 1:
                        action_logits[int(action.item())] = 10.0
                    return action_logits
                else:
                    # Continuous action space
                    if isinstance(action, np.ndarray):
                        return torch.tensor(action, device=self.device, dtype=torch.float32)
                    else:
                        return torch.tensor([action], device=self.device, dtype=torch.float32)
    
    def get_all_expert_action_logits(self, obs: Dict) -> Dict[str, torch.Tensor]:
        expert_logits = {}
        
        for subset_name in self.expert_policies.keys():
            expert_logits[subset_name] = self.get_expert_action_logits(subset_name, obs)
        
        return expert_logits


class ConfigurationScheduler:
    """Manages cycling through different observation configurations on episode completion."""
    
    def __init__(self, teacher_keys_by_config: Dict):
        self.teacher_keys_by_config = teacher_keys_by_config
        self.config_names = list(teacher_keys_by_config.keys())
        self.current_config_idx = 0
        self.episode_count = 0
        
    def get_current_config(self) -> Tuple[str, List[str]]:
        config_name = self.config_names[self.current_config_idx]
        return config_name, self.teacher_keys_by_config[config_name]
    
    def cycle_config(self, episode_done: bool = False):
        if episode_done:
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            self.episode_count += 1
            # new_config_name = self.config_names[self.current_config_idx]
            # print(f"✅ Cycled to configuration: {new_config_name} (episode {self.episode_count})")


class DistillationTrainer:
    """Custom trainer that implements pure distillation without RL."""
    
    def __init__(self, student_model: PPO, expert_manager: ExpertPolicyManager, 
                student_keys, teacher_keys_by_config, device: str = 'cpu'):
        self.student_model = student_model
        self.expert_manager = expert_manager
        self.student_keys = student_keys
        self.teacher_keys_by_config = teacher_keys_by_config
        self.device = device
        
        if device != 'cpu':
            self.student_model.policy = self.student_model.policy.to(device)
        
        # Create optimizer AFTER moving model to device - much higher LR for KL divergence
        self.optimizer = optim.Adam(student_model.policy.parameters(), lr=3e-2, weight_decay=1e-5)  # Much higher LR + regularization
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, verbose=True)
        self.distillation_losses = []
    
    def _get_student_action_logits(self, student_obs: Dict) -> torch.Tensor:
        # Convert to numpy for SB3 student policy
        obs_numpy = {}
        for key, value in student_obs.items():
            if isinstance(value, torch.Tensor):
                obs_numpy[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                obs_numpy[key] = value
            else:
                obs_numpy[key] = np.array(value)
        
        # Get student policy distribution
        obs_flat = self.student_model.policy.obs_to_tensor(obs_numpy)[0]
        student_distribution = self.student_model.policy.get_distribution(obs_flat)
        
        if hasattr(student_distribution, 'distribution'):
            if hasattr(student_distribution.distribution, 'logits'):
                logits = student_distribution.distribution.logits.to(self.device)
                return logits
            elif hasattr(student_distribution.distribution, 'mean'):
                # For continuous actions, return mean as "logits"
                return student_distribution.distribution.mean.to(self.device)
        
        # Fallback: return zeros
        action_space = self.student_model.policy.action_space
        if hasattr(action_space, 'n'):  # Discrete
            return torch.zeros(action_space.n, device=self.device)
        else:  # Continuous
            return torch.zeros(action_space.shape[0], device=self.device)

    
    def train_step(self, rollout_buffer: DistillationRolloutBuffer) -> float:
        # Get buffer dimensions
        if rollout_buffer.full_observations is None:
            print("Warning: No full observations in buffer")
            return 0.0
            
        full_obs_keys = list(rollout_buffer.full_observations.keys())
        num_steps = rollout_buffer.full_observations[full_obs_keys[0]].shape[0]
        num_envs = rollout_buffer.full_observations[full_obs_keys[0]].shape[1]
        
        # Get expert configurations for each step
        expert_configs = rollout_buffer.get_expert_configs()
        if expert_configs is None or expert_configs.size == 0:
            print("Warning: No expert configurations stored in buffer")
            return 0.0

        # Debug: Check what expert configs are stored and their distribution
        config_counts = {}
        total_valid_configs = 0
        for step in range(num_steps):
            for env in range(num_envs):
                config = expert_configs[step, env]
                if config and config != '':
                    config_counts[config] = config_counts.get(config, 0) + 1
                    total_valid_configs += 1
        
        # Verify config distribution matches expected number of eval configs
        expected_num_configs = len(self.teacher_keys_by_config)
        actual_num_configs = len([c for c in config_counts.keys() if c])
        
        # CRITICAL FIX: Store training data (observations and configs) for recomputation
        training_data = []  # Store (student_obs, expert_obs_masked, expert_config) tuples
        
        # Track training step count
        if not hasattr(self, '_training_step_count'):
            self._training_step_count = 0
        
        # Process each step and environment to collect training data
        for step in range(num_steps):
            for env in range(num_envs):
                # Get the expert configuration that was active for this specific step
                step_expert_config = expert_configs[step, env]
                
                # Validate expert config
                if not step_expert_config or step_expert_config == '':
                    continue
                
                if step_expert_config not in self.teacher_keys_by_config:
                    print(f"❌ ERROR: Unknown expert config '{step_expert_config}' at step {step}, env {env}")
                    print(f"Available configs: {list(self.teacher_keys_by_config.keys())}")
                    continue
                
                step_teacher_keys = self.teacher_keys_by_config[step_expert_config]
                
                # Get the full observation for this step and environment  
                full_obs = {}
                for key in full_obs_keys:
                    value = rollout_buffer.full_observations[key][step, env]
                    if isinstance(value, np.ndarray):
                        full_obs[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
                    else:
                        full_obs[key] = torch.tensor([value], device=self.device, dtype=torch.float32)
                
                # For STUDENT: Use the stored observations directly (already correctly masked during training)
                student_obs = {}
                for key in rollout_buffer.observations.keys():
                    value = rollout_buffer.observations[key][step, env]
                    if isinstance(value, np.ndarray):
                        student_obs[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
                    else:
                        student_obs[key] = torch.tensor([value], device=self.device, dtype=torch.float32)
                
                # For TEACHER: Filter full observation to only teacher's keys (no masking)
                # Teacher should see only the keys it was trained with, in their original form
                teacher_obs_filtered = {}
                for key in step_teacher_keys:
                    if key in full_obs:
                        teacher_obs_filtered[key] = full_obs[key]
                
                # Store training data for recomputation during training loop
                # student_obs: masked observation that student saw during training
                # teacher_obs_filtered: unmasked observation with only teacher's keys  
                training_data.append((student_obs, teacher_obs_filtered, step_expert_config))
        
        if len(training_data) == 0:
            print("Warning: No valid samples for distillation")
            return 0.0
        
        # CRITICAL FIX: Ensure student model is in training mode before computing gradients
        self.student_model.policy.train()
        
        # CRITICAL FIX: Multiple gradient steps per rollout for stronger learning
        num_epochs = 4  # Multiple epochs like standard PPO
        batch_size = min(512, len(training_data) // 4)  # Use mini-batches for stability
        
        # Compute initial loss for logging
        initial_losses = []
        for student_obs, teacher_obs_filtered, expert_config in training_data[:min(10, len(training_data))]:
            with torch.no_grad():
                expert_logits = self.expert_manager.get_expert_action_logits(expert_config, teacher_obs_filtered)
                student_logits = self._get_student_action_logits(student_obs)
                
                if expert_logits.numel() > 0 and student_logits.numel() > 0:
                    if student_logits.dim() == 1:
                        student_logits = student_logits.unsqueeze(0)
                    if expert_logits.dim() == 1:
                        expert_logits = expert_logits.unsqueeze(0)
                    
                    student_log_probs = F.log_softmax(student_logits, dim=-1)
                    expert_probs = F.softmax(expert_logits, dim=-1)
                    kl_loss = F.kl_div(student_log_probs, expert_probs, reduction='batchmean')
                    initial_losses.append(kl_loss.item())
        
        initial_loss = sum(initial_losses) / len(initial_losses) if initial_losses else 0.0
        
        for epoch in range(num_epochs):
            # Shuffle the training data for each epoch
            indices = torch.randperm(len(training_data))
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Process in mini-batches
            for i in range(0, len(training_data), batch_size):
                self.optimizer.zero_grad()
                
                # Get mini-batch indices
                batch_indices = indices[i:i+batch_size]
                
                # Collect logits for this mini-batch by recomputing them fresh
                mini_student_logits = []
                mini_expert_logits = []
                
                for idx in batch_indices:
                    student_obs, teacher_obs_filtered, expert_config = training_data[idx]
                    
                    # Get expert logits (detached since we don't need gradients for expert)
                    with torch.no_grad():
                        expert_logits = self.expert_manager.get_expert_action_logits(expert_config, teacher_obs_filtered)
                    
                    # Get student logits (with gradients attached)
                    student_logits = self._get_student_action_logits(student_obs)
                    
                    if expert_logits.numel() > 0 and student_logits.numel() > 0:
                        # Ensure correct shapes for batching
                        if student_logits.dim() == 1:
                            student_logits = student_logits.unsqueeze(0)
                        if expert_logits.dim() == 1:
                            expert_logits = expert_logits.unsqueeze(0)
                        
                        mini_student_logits.append(student_logits)
                        mini_expert_logits.append(expert_logits)
                
                if len(mini_student_logits) == 0:
                    continue
                
                # Concatenate mini-batch logits
                batch_student_logits = torch.cat(mini_student_logits, dim=0)
                batch_expert_logits = torch.cat(mini_expert_logits, dim=0)
                
                # Compute loss for this mini-batch
                mini_student_log_probs = F.log_softmax(batch_student_logits, dim=-1)
                mini_expert_probs = F.softmax(batch_expert_logits, dim=-1)
                mini_loss = F.kl_div(mini_student_log_probs, mini_expert_probs, reduction='batchmean')
                
                mini_loss.backward()
                
                # Add gradient clipping for stability
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.policy.parameters(), max_norm=1.0)
                
                # APPLY OPTIMIZER STEP
                self.optimizer.step()
                
                epoch_loss += mini_loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Debug gradient info
            if epoch == 0 or self._training_step_count % 10 == 0:
                print(f"🔧 Epoch {epoch}: Avg_loss={avg_epoch_loss:.4f}, Grad_norm={total_grad_norm:.4f}, Batches={num_batches}")
            
            # Store final loss for return (use the average epoch loss)
            distillation_loss = avg_epoch_loss
        
        final_loss = distillation_loss if isinstance(distillation_loss, float) else distillation_loss.item()
        improvement = initial_loss - final_loss
        
        # Store loss for logging and update scheduler
        self.distillation_losses.append(final_loss)
        self.scheduler.step(final_loss)  # Reduce LR on plateau
        self._training_step_count += 1
        
        # Compute accuracy metric: how often student agrees with expert (sample a few for efficiency)
        accuracy = 0.0
        if len(training_data) > 0:
            sample_size = min(50, len(training_data))
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(sample_size):
                student_obs, teacher_obs_filtered, expert_config = training_data[i]
                
                with torch.no_grad():
                    expert_logits = self.expert_manager.get_expert_action_logits(expert_config, teacher_obs_filtered)
                    student_logits = self._get_student_action_logits(student_obs)
                    
                    if expert_logits.numel() > 0 and student_logits.numel() > 0:
                        student_action = torch.argmax(student_logits, dim=-1)
                        expert_action = torch.argmax(expert_logits, dim=-1)
                        
                        if student_action.shape == expert_action.shape:
                            correct_predictions += (student_action == expert_action).sum().item()
                            total_predictions += student_action.numel()
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Enhanced debugging every 10 steps
        current_lr = self.optimizer.param_groups[0]['lr']
        if self._training_step_count % 10 == 0:
            print(f"🎯 Training Step {self._training_step_count}: Initial_loss={initial_loss:.4f}, Final_loss={final_loss:.4f}, Improvement={improvement:.4f}, Accuracy={accuracy:.3f}, LR={current_lr:.5f}")
            
            # Check if student predictions are becoming less random
            if len(training_data) > 0:
                # Use first training sample for debugging
                test_student_obs, test_teacher_obs_filtered, test_expert_config = training_data[0]
                
                with torch.no_grad():
                    test_logits = self._get_student_action_logits(test_student_obs)
                    test_probs = F.softmax(test_logits, dim=-1)
                    test_entropy = -(test_probs * torch.log(test_probs + 1e-8)).sum()
                    
                    # Also check max probability (confidence)
                    max_prob = torch.max(test_probs).item()
                    print(f"🎯 Student entropy={test_entropy.item():.4f} (lower=better, max=1.386), max_prob={max_prob:.4f} (higher=more confident)")
                    
                    # Compare with expert for same observation
                    sample_expert_logits = self.expert_manager.get_expert_action_logits(test_expert_config, test_teacher_obs_filtered)
                    sample_expert_probs = F.softmax(sample_expert_logits, dim=-1)
                    expert_entropy = -(sample_expert_probs * torch.log(sample_expert_probs + 1e-8)).sum()
                    expert_max_prob = torch.max(sample_expert_probs).item()
                    print(f"🧑‍🏫 Expert entropy={expert_entropy.item():.4f}, max_prob={expert_max_prob:.4f} (target for student)")
        
        # Track recent loss trend for debugging
        if self._training_step_count % 50 == 0 and len(self.distillation_losses) >= 10:
            recent_losses = self.distillation_losses[-10:]
            loss_trend = recent_losses[-1] - recent_losses[0]  # Negative = decreasing
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            print(f"📈 Loss Trend (last 10 steps): {loss_trend:.4f} (negative=improving), Avg={avg_recent_loss:.4f}")
            
            if abs(loss_trend) < 0.01:  # Very small change
                print(f"⚠️  Warning: Loss plateau detected. Consider increasing LR or checking data diversity.")
        
        return final_loss


class PureDistillationCallback(BaseCallback):
    """Custom callback for pure distillation training - no RL, only imitation."""
    
    def __init__(self, expert_manager, student_keys, teacher_keys_by_config, device='cpu', verbose=1):
        super().__init__(verbose)
        self.expert_manager = expert_manager
        self.student_keys = student_keys
        self.teacher_keys_by_config = teacher_keys_by_config
        self.device = device
        self.distillation_trainer = None
        self.training_step_count = 0
        
    def _on_training_start(self):
        """Called when training starts."""
        # Initialize distillation trainer
        self.distillation_trainer = DistillationTrainer(
            self.model, 
            self.expert_manager, 
            self.student_keys,
            self.teacher_keys_by_config,
            self.device
        )
        print("🎯 Initialized pure distillation trainer")
        
    def _on_step(self):
        """Called after each step - no longer needed for expert config storage."""
        return True
    
    def _on_rollout_start(self):
        """Called at the start of a rollout."""
        return True
    
    def _on_rollout_step(self):
        """Called after each step in the rollout - no longer needed since we handle this in _on_step."""
        return True
    
    def _on_rollout_end(self):
        """Called at the end of a rollout - apply pure distillation loss."""
        if self.distillation_trainer is None:
            return True
            
        # Apply distillation loss using the stored expert configs for each step
        rollout_buffer = self.model.rollout_buffer
        distillation_loss = self.distillation_trainer.train_step(rollout_buffer)
        
        # Log distillation loss
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            self.model.logger.record("distillation/loss", distillation_loss)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "distillation/loss": distillation_loss,
                "distillation/training_step": self.training_step_count
            })
        
        self.training_step_count += 1
        
        return True


class RewardZeroingWrapper(gym.RewardWrapper):
    """Wrapper that zeros out all rewards to prevent RL learning."""
    
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        """Always return zero reward."""
        return 0.0


def create_distillation_envs(config, seed, student_keys, all_required_keys, teacher_keys_by_config):
    """Create environments for distillation training."""
    
    def _make_env():
        # Create the base environment using gymnasium
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        
        # Zero out rewards to prevent RL learning - PURE DISTILLATION ONLY
        env = RewardZeroingWrapper(env)
                
        env.reset(seed=seed)
        return env
    
    # Create vectorized environment
    if config.num_envs > 1:
        env_fns = [lambda i=i: _make_env() for i in range(config.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([_make_env])

    vec_env = VecMonitor(vec_env)
    
    # Wrap with vectorized distillation wrapper using pre-parsed keys
    vec_env = VectorizedDistillationWrapper(vec_env, student_keys, all_required_keys, teacher_keys_by_config)
    
    return vec_env


class PureDistillationPPO(PPO):
    """Custom PPO that disables standard training and only uses distillation."""
    
    def train(self):
        """
        Override PPO's train method to disable standard training.
        All learning happens through distillation callback only.
        """
        # Don't do any standard PPO training - distillation callback handles all learning
        # But keep the logging mechanisms that SB3 expects
        
        # DON'T update SB3's optimizer - it conflicts with our custom distillation optimizer
        # self._update_learning_rate([self.policy.optimizer])  # REMOVED - this was causing conflicts!
        
        # Record basic training metrics (keeping SB3's logging happy)
        self.logger.record("train/n_updates", self._n_updates)
        # Use a dummy learning rate for logging since we're not using SB3's optimizer
        self.logger.record("train/learning_rate", 1e-3)  # Our distillation LR
        self.logger.record("train/clip_range", self.clip_range)
        
        # Increment update counter for consistency
        self._n_updates += self.n_epochs
        
        # Don't call parent train() - that would do actual PPO training
        return


def train_ppo_distill_sb3(config, seed: int, expert_policy_dir: str, device: str = 'cpu'):
    """
    Train a student policy using pure distillation (no RL) from expert policies.
    
    Args:
        config: Configuration object
        seed: Random seed
        expert_policy_dir: Directory containing expert subset policies
        device: Device to use
        
    Returns:
        PPO: Trained SB3 PPO model (student policy)
    """
    # Set random seed
    set_random_seed(seed)
    
    # Create expert policy manager
    expert_manager = ExpertPolicyManager(expert_policy_dir, device)
    
    # Parse all observation keys once by creating a temporary environment
    full_obs_space = get_full_observation_space(config)
    
    # Get student keys directly from config (don't infer)
    if not hasattr(config, 'keys'):
        raise ValueError("config.keys is required for student key configuration")
    
    student_mlp_pattern = getattr(config.keys, 'mlp_keys', '.*')
    student_cnn_pattern = getattr(config.keys, 'cnn_keys', '.*')
    student_keys = parse_keys_from_patterns(full_obs_space, student_mlp_pattern, student_cnn_pattern)
    
    # Get teacher keys for all expert configurations directly from config
    teacher_keys_by_config = {}
    if not hasattr(config, 'eval_keys'):
        raise ValueError("config.eval_keys is required for teacher key configurations")
    
    num_configs = getattr(config, 'num_eval_configs', 4)
    
    for env_idx in range(1, num_configs + 1):
        env_name = f"env{env_idx}"
        if not hasattr(config.eval_keys, env_name):
            raise ValueError(f"config.eval_keys.{env_name} is required but not found")
        
        eval_config = getattr(config.eval_keys, env_name)
        teacher_mlp_pattern = getattr(eval_config, 'mlp_keys', '.*')
        teacher_cnn_pattern = getattr(eval_config, 'cnn_keys', '.*')
        teacher_keys = parse_keys_from_patterns(full_obs_space, teacher_mlp_pattern, teacher_cnn_pattern)
        teacher_keys_by_config[env_name] = teacher_keys
    
    # Create all required keys (union of student + all teacher keys)
    all_required_keys = set(student_keys)
    for teacher_keys in teacher_keys_by_config.values():
        all_required_keys.update(teacher_keys)
    all_required_keys = sorted(list(all_required_keys))
    print(f"🔗 All required keys (student + teachers): {all_required_keys}")
    
    # Note: Episode-level configuration cycling is now handled by the environment wrapper
    print(f"🚀 Environment wrapper will cycle through {len(teacher_keys_by_config)} configurations")
    
    # Create environments
    print("🚀 Creating environments...")
    vec_env = create_distillation_envs(config, seed, student_keys, all_required_keys, teacher_keys_by_config)
    
    # Create evaluation environment with SAME filtering as training (for consistency)
    def _make_eval_env():
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        
        # Apply the SAME observation filtering as training environment
        if hasattr(config, 'keys') and config.keys:
            student_mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
            student_cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
            env = ObservationFilterWrapper(env, student_mlp_keys, student_cnn_keys)
        
        env.reset(seed=seed)
        return env
    
    eval_env = _make_eval_env()
    
    # Create unfiltered evaluation environment for custom evaluation
    def _make_unfiltered_eval_env():
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        # No filtering - keep all observations for masking
        env.reset(seed=seed)
        return env
    
    unfiltered_eval_env = _make_unfiltered_eval_env()
    
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
            name=f"ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
            config=config,
            sync_tensorboard=True,
        )

    # Create custom rollout buffer for distillation
    rollout_buffer_kwargs = {}
    
    # Create custom PPO model that disables standard training
    model = PureDistillationPPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for dictionary observations
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
        rollout_buffer_class=DistillationRolloutBuffer,
        rollout_buffer_kwargs=rollout_buffer_kwargs,
    )
    
    # Connect the rollout buffer with the environment wrapper for full observation storage
    if hasattr(vec_env, 'get_last_full_observations'):  # VectorizedDistillationWrapper
        model.rollout_buffer.set_env_wrapper(vec_env)
    
    # Create evaluation callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
        log_path=f"./eval_logs/ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create pure distillation callback (no RL, only imitation)
    distill_callback = PureDistillationCallback(
        expert_manager, 
        student_keys,
        teacher_keys_by_config,
        device=device
    )

    # Create custom evaluation callback
    custom_eval_callback = CustomEvalCallback(
        unfiltered_eval_env,  # Use unfiltered environment for masking
        _make_unfiltered_eval_env,  # Function to create unfiltered environments
        expert_manager,  # Pass expert manager for teacher evaluation
        student_keys,
        teacher_keys_by_config,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
        debug=False
    )

    # Prepare callbacks
    callbacks = [eval_callback, distill_callback, custom_eval_callback]
    
    # Train the model using pure distillation (no RL loss)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
    )

    vec_env.close()
    eval_env.close()
    unfiltered_eval_env.close()
    
    if run is not None:
        run.finish()
    
    return model 