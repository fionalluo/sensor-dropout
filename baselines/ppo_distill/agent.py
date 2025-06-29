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

from baselines.ppo_rnn.agent import PPORnnAgent
from baselines.shared.eval_utils import filter_observations_by_keys, substitute_unprivileged_for_agent
from baselines.shared.policy_utils import (
    load_policy_like_subset_policies,
    find_policy_files, 
    load_metadata_from_dir
)
from subset_policies.load_subset_policy import SubsetPolicyLoader


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
            policy_type = metadata.get('policy_type', 'ppo_rnn')
        else:
            # Fallback: try to determine from directory structure
            policy_type = 'ppo_rnn'  # Default to ppo_rnn
        
        # Use the working approach to load policies
        loaded_policies = load_policy_like_subset_policies(self.policy_dir, policy_type, self.device)
        
        # Store the loaded policies
        for subset_name, (agent, config, eval_keys) in loaded_policies.items():
            self.expert_policies[subset_name] = agent
            self.expert_configs[subset_name] = config
            self.expert_eval_keys[subset_name] = eval_keys
            
            print(f"Loaded expert policy: {subset_name}")
        
        print(f"Loaded {len(self.expert_policies)} expert policies")
    
    def get_expert_action(self, subset_name: str, obs: Dict, lstm_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Get action logits from a specific expert policy for distillation.
        
        Args:
            subset_name: Name of the subset (env1, env2, env3, env4)
            obs: Observation dictionary
            lstm_state: LSTM state (optional)
            
        Returns:
            tuple: (expert_logits, new_lstm_state)
        """
        if subset_name not in self.expert_policies:
            raise ValueError(f"Expert policy {subset_name} not found")
        
        expert_agent = self.expert_policies[subset_name]
        eval_keys = self.expert_eval_keys[subset_name]
        
        # Filter observations based on the expert's eval_keys
        filtered_obs = filter_observations_by_keys(
            obs, eval_keys['mlp_keys'], eval_keys['cnn_keys']
        )
        
        # Convert to tensor and add batch dimension
        obs_tensor = {}
        for key, value in filtered_obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
            else:
                obs_tensor[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
        
        # Get expert action logits for distillation
        with torch.no_grad():
            # Get hidden states from LSTM
            hidden, new_lstm_state = expert_agent.get_states(obs_tensor, lstm_state, torch.zeros(1, device=self.device))
            
            # Get action logits from the actor network
            expert_logits = expert_agent.actor(hidden)
            
            return expert_logits, new_lstm_state
    
    def get_all_expert_actions(self, obs: Dict, lstm_states: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Get action logits from all expert policies for the same observation.
        
        Args:
            obs: Observation dictionary
            lstm_states: Dictionary of LSTM states for each expert (optional)
            
        Returns:
            dict: Dictionary of expert action logits for each subset
        """
        expert_actions = {}
        new_lstm_states = {}
        
        for subset_name in self.expert_policies.keys():
            lstm_state = lstm_states.get(subset_name) if lstm_states else None
            expert_logits, new_lstm_state = self.get_expert_action(subset_name, obs, lstm_state)
            expert_actions[subset_name] = expert_logits
            new_lstm_states[subset_name] = new_lstm_state
        
        return expert_actions, new_lstm_states


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
        print(f"🔄 ConfigurationScheduler.cycle_config called with episode_done={episode_done}")
        print(f"🔄 cycle_mode={self.cycle_mode}, current_idx={self.current_config_idx}")
        
        if self.cycle_mode == 'episode' and episode_done:
            print(f"🔄 Episode done, cycling from {self.config_names[self.current_config_idx]}")
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            self.episode_count += 1
            print(f"🔄 Now using config: {self.config_names[self.current_config_idx]}")
        elif self.cycle_mode == 'batch':
            print(f"🔄 Batch mode cycling from {self.config_names[self.current_config_idx]}")
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            print(f"🔄 Now using config: {self.config_names[self.current_config_idx]}")
        else:
            print(f"🔄 No cycling: episode_done={episode_done}, cycle_mode={self.cycle_mode}")


class PPODistillAgent:
    """PPO Distill Agent that learns from expert subset policies."""
    
    def __init__(self, envs, config, expert_policy_dir: str, device: str = 'cpu', student_policy_type: str = "ppo_rnn"):
        """
        Initialize the PPO Distill agent.
        
        Args:
            envs: Environment
            config: Configuration
            expert_policy_dir: Directory containing expert subset policies
            device: Device to use
            student_policy_type: Type of student agent ("ppo" or "ppo_rnn")
        """
        self.student_policy_type = student_policy_type
        
        # Create the appropriate base agent based on student_policy_type
        if student_policy_type == "ppo":
            from baselines.ppo.agent import PPOAgent
            self.base_agent = PPOAgent(envs, config)
        elif student_policy_type == "ppo_rnn":
            from baselines.ppo_rnn.agent import PPORnnAgent
            self.base_agent = PPORnnAgent(envs, config)
        else:
            raise ValueError(f"Unknown student policy type: {student_policy_type}")
        
        # Set device
        self.device = device
        self.base_agent.to(device)
        
        # Copy all attributes from the base agent
        for attr_name in dir(self.base_agent):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.base_agent, attr_name)
                if not callable(attr_value) or attr_name in ['get_action_and_value', 'get_value', 'get_states', 'get_action_logits']:
                    setattr(self, attr_name, attr_value)
        
        # Initialize expert policy manager
        self.expert_manager = ExpertPolicyManager(expert_policy_dir, device)
        
        # Check if any expert policies were loaded
        if not self.expert_manager.expert_policies:
            print("Warning: No expert policies loaded. Using default configuration.")
            # Create a default configuration with full keys
            default_eval_keys = {
                'env1': {
                    'mlp_keys': '.*',
                    'cnn_keys': '.*'
                }
            }
            self.expert_manager.expert_eval_keys = default_eval_keys
        
        # Initialize configuration scheduler
        self.config_scheduler = ConfigurationScheduler(
            self.expert_manager.expert_eval_keys, 
            cycle_mode=getattr(config, 'cycle_mode', 'episode')
        )
        
        print(f"🔧 Expert configurations loaded: {list(self.expert_manager.expert_eval_keys.keys())}")
        print(f"🔧 Cycle mode: {getattr(config, 'cycle_mode', 'episode')}")
        print(f"🔧 Configuration scheduler initialized with {len(self.expert_manager.expert_eval_keys)} configs")
        
        # Distillation parameters
        self.distill_coef = getattr(config, 'distill_coef', 0.1)
        self.expert_coef = getattr(config, 'expert_coef', 0.5)
        
        # Current configuration tracking
        self.current_config_name = None
        self.current_eval_keys = None
        
        # Initialize current configuration
        self.current_config_name, self.current_eval_keys = self.get_current_config()
    
    def get_current_config(self) -> Tuple[str, Dict]:
        """Get the current configuration."""
        config_name, eval_keys = self.config_scheduler.get_current_config()
        
        # Debug logging to see if this is being called (immediate for debugging)
        print(f"🔍 get_current_config called: {config_name}")
        
        return config_name, eval_keys
    
    def cycle_config(self, episode_done: bool = False):
        """Cycle to the next configuration."""
        print(f"🔄 cycle_config called with episode_done={episode_done}")
        print(f"🔄 Current cycle_mode: {self.config_scheduler.cycle_mode}")
        print(f"🔄 Current config_idx: {self.config_scheduler.current_config_idx}")
        print(f"🔄 Total configs: {len(self.config_scheduler.config_names)}")
        
        old_config_name = self.current_config_name
        self.config_scheduler.cycle_config(episode_done)
        self.current_config_name, self.current_eval_keys = self.get_current_config()
        
        # Print detailed logging when configuration changes
        if old_config_name != self.current_config_name:
            print(f"\n🔄 Cycling to configuration: {self.current_config_name}")
            print(f"📋 Expert policy to mimic: {self.current_config_name}")
            print(f"🔍 Observation filtering patterns:")
            print(f"   MLP keys: {self.current_eval_keys['mlp_keys']}")
            print(f"   CNN keys: {self.current_eval_keys['cnn_keys']}")
            print("-" * 50)
        else:
            print(f"🔄 No config change - still using: {self.current_config_name}")
    
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
        # Debug logging to see if this method is being called (less frequent)
        if not hasattr(self, '_last_mask_log_time'):
            self._last_mask_log_time = 0
        
        current_time = time.time()
        if current_time - self._last_mask_log_time > 10.0:  # Log every 10 seconds
            print(f"🎭 _mask_observations called with config: {self.current_config_name}")
            self._last_mask_log_time = current_time
        
        # Step 1: Filter observations by patterns (like filter_observations_by_keys)
        mlp_pattern = eval_keys['mlp_keys']
        cnn_pattern = eval_keys['cnn_keys']
        
        def matches_pattern(key, pattern):
            if pattern == '.*':
                return True
            elif pattern == '^$':
                return False
            else:
                return re.search(pattern, key) is not None
        
        # Collect all keys that match the patterns (these are the "available" keys)
        available_keys = set()
        
        # Process MLP keys
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if matches_pattern(key, mlp_pattern):
                available_keys.add(key)
        
        # Process CNN keys
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if matches_pattern(key, cnn_pattern):
                available_keys.add(key)
        
        # Create filtered observations dict
        filtered_obs = {}
        for key, value in obs.items():
            if key in available_keys:
                filtered_obs[key] = value
        
        # Step 2: Substitute unprivileged keys for agent keys (like substitute_unprivileged_for_agent)
        # Get all keys the agent expects (full_keys)
        all_keys = self.mlp_keys + self.cnn_keys
        final_obs = {}
        
        # Detailed logging for key processing (less frequent)
        key_logs = []
        
        for key in all_keys:
            if key in filtered_obs:
                # Key is directly available
                final_obs[key] = filtered_obs[key]
                key_logs.append(f"  {key}: {key} (direct)")
            else:
                # Key is not available, look for unprivileged version with prefix matching
                unprivileged_key = self._find_unprivileged_key(key, filtered_obs)
                if unprivileged_key:
                    # Use unprivileged version as substitute
                    final_obs[key] = filtered_obs[unprivileged_key]
                    key_logs.append(f"  {key}: {unprivileged_key} (unprivileged substitute)")
                else:
                    # Neither privileged nor unprivileged available, will be zeroed later
                    final_obs[key] = None
                    key_logs.append(f"  {key}: zeroed (not available)")
        
        # Print detailed key processing log (less frequent to avoid spam)
        if not hasattr(self, '_last_key_log_time'):
            self._last_key_log_time = 0
        
        current_time = time.time()
        if current_time - self._last_key_log_time > 15.0:  # Log every 15 seconds
            print(f"🔑 Key processing for {self.current_config_name}:")
            for log in key_logs:
                print(log)
            print("-" * 30)
            self._last_key_log_time = current_time
        
        # Step 3: Convert to tensors and handle None values (zero out)
        masked_obs = {}
        for key in all_keys:
            if final_obs[key] is not None:
                if isinstance(final_obs[key], torch.Tensor):
                    masked_obs[key] = final_obs[key].clone()
                else:
                    masked_obs[key] = torch.tensor(final_obs[key], device=self.device)
            else:
                # Zero out missing observations
                if key in obs:
                    if isinstance(obs[key], torch.Tensor):
                        masked_obs[key] = torch.zeros_like(obs[key])
                    else:
                        masked_obs[key] = torch.zeros_like(torch.tensor(obs[key], device=self.device))
                else:
                    # Fallback: create zero tensor with expected shape
                    # This shouldn't happen in practice, but just in case
                    masked_obs[key] = torch.zeros(1, device=self.device)
        
        return masked_obs
    
    def _find_unprivileged_key(self, full_key: str, available_keys: dict) -> Optional[str]:
        """
        Find the unprivileged version of a key.
        
        Args:
            full_key: The full key to find unprivileged version for
            available_keys: Dictionary of available keys in current subset
            
        Returns:
            Optional[str]: The unprivileged key if found, None otherwise
        """
        # Look for keys that start with 'full_key_unprivileged'
        unprivileged_pattern = f"{full_key}_unprivileged"
        
        for key in available_keys:
            if key.startswith(unprivileged_pattern):
                return key
        
        return None
    
    def get_action_and_value(self, obs, lstm_state=None, done=None, action=None):
        """
        Get action and value from the student policy, along with expert actions for distillation.
        
        Args:
            obs: Full observations (should contain all keys)
            lstm_state: LSTM state (only for PPO-RNN)
            done: Done flags
            action: Actions (for evaluation)
            
        Returns:
            tuple: (action, logprob, entropy, value, new_lstm_state, expert_actions)
        """
        # Get current configuration
        config_name, eval_keys = self.get_current_config()
        
        # Mask observations based on current configuration
        # Student receives full_keys but with masked values
        masked_obs = self._mask_observations(obs, eval_keys)
        
        # Get student action and value using masked observations
        if self.student_policy_type == "ppo_rnn":
            # PPO-RNN expects lstm_state and done
            action, logprob, entropy, value, new_lstm_state = self.base_agent.get_action_and_value(
                masked_obs, lstm_state, done, action
            )
        else:
            # PPO doesn't use lstm_state
            action, logprob, entropy, value = self.base_agent.get_action_and_value(
                masked_obs, action
            )
            new_lstm_state = None
        
        # Get expert actions for distillation
        # Experts receive filtered observations (as they were trained)
        expert_actions = {}
        if self.expert_manager.expert_policies:
            for subset_name in self.expert_manager.expert_policies.keys():
                try:
                    expert_logits, _ = self.expert_manager.get_expert_action(
                        subset_name, obs, lstm_state
                    )
                    expert_actions[subset_name] = expert_logits
                except Exception as e:
                    print(f"Warning: Failed to get expert action for {subset_name}: {e}")
                    # Continue with other experts
        
        return action, logprob, entropy, value, new_lstm_state, expert_actions
    
    def compute_distillation_loss(self, student_logits: torch.Tensor, expert_actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute distillation loss between student and expert action distributions.
        
        Args:
            student_logits: Action logits from the student policy
            expert_actions: Action logits from expert policies
            
        Returns:
            torch.Tensor: Distillation loss
        """
        config_name, _ = self.get_current_config()
        
        if not expert_actions or config_name not in expert_actions:
            return torch.tensor(0.0, device=self.device)
        
        expert_logits = expert_actions[config_name]
        
        # Convert logits to probabilities for both student and expert
        student_probs = F.softmax(student_logits, dim=-1)
        expert_probs = F.softmax(expert_logits, dim=-1)
        
        # KL divergence loss: KL(student || expert)
        # This encourages the student to match the expert's action distribution
        distill_loss = F.kl_div(
            student_probs.log(), expert_probs, 
            reduction='batchmean'
        )
        
        return distill_loss
    
    def get_value(self, obs, lstm_state=None, done=None):
        """Get value from the student policy."""
        config_name, eval_keys = self.get_current_config()
        masked_obs = self._mask_observations(obs, eval_keys)
        
        if self.student_policy_type == "ppo_rnn":
            return self.base_agent.get_value(masked_obs, lstm_state, done)
        else:
            return self.base_agent.get_value(masked_obs)
    
    def get_states(self, obs, lstm_state=None, done=None):
        """Get states from the student policy."""
        config_name, eval_keys = self.get_current_config()
        masked_obs = self._mask_observations(obs, eval_keys)
        
        if self.student_policy_type == "ppo_rnn":
            return self.base_agent.get_states(masked_obs, lstm_state, done)
        else:
            return self.base_agent.get_states(masked_obs)
    
    def get_action_logits(self, obs, lstm_state=None, done=None):
        """Get action logits from the student policy."""
        config_name, eval_keys = self.get_current_config()
        masked_obs = self._mask_observations(obs, eval_keys)
        
        if self.student_policy_type == "ppo_rnn":
            return self.base_agent.get_action_logits(masked_obs, lstm_state, done)
        else:
            return self.base_agent.get_action_logits(masked_obs)
    
    def get_initial_lstm_state(self):
        """Get initial LSTM state (only for PPO-RNN)."""
        if self.student_policy_type == "ppo_rnn":
            return self.base_agent.get_initial_lstm_state()
        else:
            return None 