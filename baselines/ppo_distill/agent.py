#!/usr/bin/env python3
"""
PPO Distill Agent that learns from multiple expert subset policies.
This agent cycles through different observation configurations and learns to mimic expert behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from types import SimpleNamespace
import re
import os
import time


from baselines.shared.eval_utils import filter_observations_by_keys
from baselines.shared.policy_utils import (
    load_policy_like_subset_policies,
    find_policy_files, 
    load_metadata_from_dir
)
from subset_policies.load_subset_policy import SubsetPolicyLoader
from baselines.shared.masking_utils import mask_observations_for_student


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
        self.expert_configs = {}
        self.expert_eval_keys = {}
        
        # Load all expert policies
        self._load_expert_policies()
    
    def _load_expert_policies(self):
        """Load all expert policies from the directory using the working approach."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Expert policy directory not found: {self.policy_dir}")
        
        # Load metadata to determine policy type
        metadata = load_metadata_from_dir(self.policy_dir)
        if metadata:
            policy_type = metadata.get('policy_type', 'ppo')
        else:
            # Fallback: try to determine from directory structure
            policy_type = 'ppo'  # Default to ppo
        
        # Use the working approach to load policies
        loaded_policies = load_policy_like_subset_policies(self.policy_dir, policy_type, self.device)
        
        # Store the loaded policies
        for subset_name, (agent, config, eval_keys) in loaded_policies.items():
            self.expert_policies[subset_name] = agent
            self.expert_configs[subset_name] = config
            self.expert_eval_keys[subset_name] = eval_keys
            
            print(f"Loaded expert policy: {subset_name}")
        
        print(f"Loaded {len(self.expert_policies)} expert policies")
    
    def get_expert_action(self, subset_name: str, obs: Dict) -> torch.Tensor:
        """
        Get action logits from a specific expert policy for distillation.
        Only select the keys from the raw obs dict that match the expert's eval_keys patterns. Raise an error if any expected key is missing.
        """
        import re
        if subset_name not in self.expert_policies:
            raise ValueError(f"Expert policy {subset_name} not found")
        expert_agent = self.expert_policies[subset_name]
        eval_keys = self.expert_eval_keys[subset_name]

        # Only select the keys that match the expert's eval_keys patterns
        filtered_obs = {}
        
        # Get all keys that match the eval_keys pattern
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if re.match(eval_keys['mlp_keys'], key) or re.match(eval_keys['cnn_keys'], key):
                filtered_obs[key] = value

        # Convert to tensor
        obs_tensor = {}
        for key, value in filtered_obs.items():
            if isinstance(value, torch.Tensor):
                obs_tensor[key] = value
            else:
                obs_tensor[key] = torch.tensor(value, device=self.device)

        # Forward through expert agent
        hidden = expert_agent.encode_observations(obs_tensor)
        expert_logits = expert_agent.actor(hidden)
        return expert_logits
    
    def get_all_expert_actions(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        Get action logits from all expert policies for distillation.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            dict: Dictionary of expert action logits for each subset
        """
        expert_actions = {}
        
        for subset_name in self.expert_policies.keys():
            expert_logits = self.get_expert_action(subset_name, obs)
            expert_actions[subset_name] = expert_logits
        
        return expert_actions


class ConfigurationScheduler:
    """Manages cycling through different observation configurations."""
    
    def __init__(self, eval_keys: Dict, cycle_mode: str = 'episode'):
        """
        Initialize the configuration scheduler.
        
        Args:
            eval_keys: Dictionary of eval_keys for each configuration
            cycle_mode: How to cycle configurations ('episode' or 'batch')
        """
        self.eval_keys = eval_keys
        self.cycle_mode = cycle_mode
        self.config_names = list(eval_keys.keys())
        self.current_config_idx = 0
        self.episode_count = 0
        
    def get_current_config(self) -> Tuple[str, Dict]:
        """
        Get the current configuration.
        
        Returns:
            tuple: (config_name, eval_keys_dict)
        """
        config_name = self.config_names[self.current_config_idx]
        return config_name, self.eval_keys[config_name]
    
    def cycle_config(self, episode_done: bool = False):
        """
        Cycle to the next configuration.
        
        Args:
            episode_done: Whether the current episode is done (for episode-level cycling)
        """
        if self.cycle_mode == 'episode' and episode_done:
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            self.episode_count += 1
            new_config_name = self.config_names[self.current_config_idx]
            print(f"✅ Cycled to configuration: {new_config_name} (episode {self.episode_count})")
        elif self.cycle_mode == 'batch':
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            new_config_name = self.config_names[self.current_config_idx]
            print(f"✅ Cycled to configuration: {new_config_name} (batch mode)")


class PPODistillAgent:
    """PPO Distill Agent that learns from expert subset policies."""
    
    def __init__(self, envs, config, expert_policy_dir: str, device: str = 'cuda'):
        """
        Initialize the PPO Distill agent.
        
        Args:
            envs: Environment
            config: Configuration
            expert_policy_dir: Directory containing expert subset policies
            device: Device to use
        """
        
        # Create the base PPO agent
        from baselines.ppo.agent import PPOAgent
        self.base_agent = PPOAgent(envs, config)
        
        # Set device
        self.device = device
        self.base_agent.to(device)
        
        # Copy all attributes from the base agent
        for attr_name in dir(self.base_agent):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.base_agent, attr_name)
                if not callable(attr_value) or attr_name not in ['get_action_and_value', 'get_value', 'get_states', 'get_action_logits']:
                    setattr(self, attr_name, attr_value)
        
        # Initialize expert policy manager
        self.expert_manager = ExpertPolicyManager(expert_policy_dir, device)
        
        # Check if any expert policies were loaded
        if not self.expert_manager.expert_policies:
            raise RuntimeError("No expert policies loaded. Distillation cannot proceed without expert policies.")
        
        # Initialize configuration scheduler
        self.config_scheduler = ConfigurationScheduler(
            self.expert_manager.expert_eval_keys, 
            cycle_mode=getattr(config, 'cycle_mode', 'episode')
        )
        
        print(f"🔧 Expert configurations loaded: {list(self.expert_manager.expert_eval_keys.keys())}")
        print(f"🔧 Cycle mode: {getattr(config, 'cycle_mode', 'episode')}")
        print(f"🔧 Configuration scheduler initialized with {len(self.expert_manager.expert_eval_keys)} configs")
        
        # Current configuration tracking
        self.current_config_name = None
        self.current_eval_keys = None
        
        # Initialize current configuration
        self.current_config_name, self.current_eval_keys = self.get_current_config()
    
    def get_current_config(self) -> Tuple[str, Dict]:
        """Get the current configuration."""
        config_name, eval_keys = self.config_scheduler.get_current_config()
        
        # Debug logging to see if this is being called (less frequent for debugging)
        if not hasattr(self, '_last_config_log_time'):
            self._last_config_log_time = 0
        
        current_time = time.time()
        if current_time - self._last_config_log_time > 30.0:  # Log every 30 seconds
            print(f"🔍 get_current_config called: {config_name}")
            self._last_config_log_time = current_time
        
        return config_name, eval_keys
    
    def cycle_config(self, episode_done: bool = False):
        """Cycle to the next configuration."""
        # Remove excessive debug prints
        old_config_name = self.current_config_name
        self.config_scheduler.cycle_config(episode_done)
        self.current_config_name, self.current_eval_keys = self.get_current_config()
    
    def _mask_observations(self, obs: Dict, eval_keys: Dict) -> Dict:
        """
        Mask observations based on eval_keys while keeping the original structure.
        This matches the exact logic from eval_utils.py.
        The student receives full_keys, but we map them to available keys in the current subset.
        Args:
            obs: Full observations (full_keys)
            eval_keys: Eval keys for current configuration (subset keys)
        Returns:
            Dict: Masked observations with same structure as input
        """
        student_keys = self.mlp_keys + self.cnn_keys
        
        # Get teacher keys by matching eval_keys patterns against available observation keys
        teacher_keys = []
        for key in obs.keys():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if re.match(eval_keys['mlp_keys'], key) or re.match(eval_keys['cnn_keys'], key):
                teacher_keys.append(key)
        
        masked_obs = mask_observations_for_student(obs, student_keys, teacher_keys, device=self.device)
        return masked_obs
    
    def get_action_and_value(self, obs, action=None, evaluation_mode=False):
        """
        Get action and value from the student policy, along with expert actions for distillation.
        
        Args:
            obs: Observations from environment
            action: Action (for PPO)
            evaluation_mode: If True, skip expert computation and internal masking
        """
        if evaluation_mode:
            # In evaluation mode, observations are already masked externally
            # Skip expert computation and internal masking
            action, logprob, entropy, value = self.base_agent.get_action_and_value(
                obs, action
            )
            
            # Return dummy values for expert_actions and student_logits to maintain interface
            dummy_expert_actions = {}
            dummy_student_logits = torch.zeros_like(action) if hasattr(action, 'shape') else torch.tensor(0.0)
            
            return action, logprob, entropy, value, None, dummy_expert_actions, dummy_student_logits
        
        # Normal training mode - get current configuration and mask observations
        config_name, eval_keys = self.get_current_config()
        
        # Mask observations based on current configuration
        masked_obs = self._mask_observations(obs, eval_keys)
        
        # Get student action and value using masked observations
        action, logprob, entropy, value = self.base_agent.get_action_and_value(
            masked_obs, action
        )
        
        # Get student logits for distillation loss computation
        hidden = self.base_agent.encode_observations(masked_obs)
        student_logits = self.base_agent.actor(hidden)
        
        # Get expert action for current configuration only
        try:
            expert_logits = self.expert_manager.get_expert_action(config_name, obs)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to get expert action for {config_name}: {e}")
            print(f"❌ Available expert policies: {list(self.expert_manager.expert_policies.keys())}")
            print(f"❌ Current config: {config_name}")
            print(f"❌ Original observation keys: {list(obs.keys())}")
            raise RuntimeError(f"Failed to get expert action for {config_name}: {e}")
        
        # Verify we have expert action for the current config
        if expert_logits is None:
            print(f"❌ CRITICAL ERROR: Expert logits for {config_name} is None!")
            raise RuntimeError(f"Expert logits for {config_name} is None")

        return action, logprob, entropy, value, None, {config_name: expert_logits}, student_logits

    def compute_distillation_loss(self, student_logits: torch.Tensor, expert_actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute distillation loss between student and expert action distributions.
        Adds debug prints for diagnostics.
        """
        config_name, _ = self.get_current_config()
        if not expert_actions:
            print("❌ CRITICAL ERROR: No expert actions provided for distillation loss!")
            raise RuntimeError("No expert actions provided for distillation loss")
        if config_name not in expert_actions:
            print(f"❌ CRITICAL ERROR: Current config '{config_name}' not found in expert actions!")
            print(f"❌ Available expert actions: {list(expert_actions.keys())}")
            raise RuntimeError(f"Current config '{config_name}' not found in expert actions")
        expert_logits = expert_actions[config_name]
        if expert_logits is None:
            print(f"❌ CRITICAL ERROR: Expert logits for {config_name} is None!")
            raise RuntimeError(f"Expert logits for {config_name} is None")

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        expert_probs = F.softmax(expert_logits, dim=-1)
        distill_loss = F.kl_div(
            student_log_probs, expert_probs,
            reduction='batchmean'
        )
        return distill_loss
    
    def get_value(self, obs):
        """Get value from the student policy."""
        config_name, eval_keys = self.get_current_config()
        masked_obs = self._mask_observations(obs, eval_keys)
        
        return self.base_agent.get_value(masked_obs)
    
    def get_states(self, obs):
        """Get states from the student policy."""
        config_name, eval_keys = self.get_current_config()
        masked_obs = self._mask_observations(obs, eval_keys)
        
        return self.base_agent.get_states(masked_obs)
    
    def get_action_logits(self, obs):
        """Get action logits from the student policy."""
        config_name, eval_keys = self.get_current_config()
        masked_obs = self._mask_observations(obs, eval_keys)
        
        return self.base_agent.get_action_logits(masked_obs)
    
    def get_initial_lstm_state(self):
        """Get initial LSTM state (not used for PPO)."""
        return None 