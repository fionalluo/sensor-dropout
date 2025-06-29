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
        import re  # Move import to top of function
        
        if subset_name not in self.expert_policies:
            raise ValueError(f"Expert policy {subset_name} not found")
        
        expert_agent = self.expert_policies[subset_name]
        eval_keys = self.expert_eval_keys[subset_name]
        
        # Filter observations based on the expert's eval_keys
        filtered_obs = {}
        
        # Helper function to check if a key matches a pattern
        def matches_pattern(key, pattern):
            if pattern == '.*':
                return True
            elif pattern == '^$':
                return False
            else:
                # Use re.match instead of re.search for better pattern matching
                return re.match(pattern, key) is not None
        
        # For evaluation: implement proper substitution logic
        # For every full_key: if it's in eval_keys, use it; if not, search for key_unprivileged; if not found, substitute 0s
        
        # First, get all the keys that match the eval_keys pattern
        pattern_matched_keys = set()
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if matches_pattern(key, eval_keys['mlp_keys']):
                pattern_matched_keys.add(key)
            if matches_pattern(key, eval_keys['cnn_keys']):
                pattern_matched_keys.add(key)
        
        # Now implement the substitution logic
        # For each key that the expert expects (based on eval_keys pattern), find the best substitute
        for key in pattern_matched_keys:
            if key in obs:
                # Key is directly available
                filtered_obs[key] = obs[key]
            else:
                # Key not available, this shouldn't happen since we're iterating over matched keys
                print(f"❌ WARNING: Key {key} matched pattern but not in obs")
        
        # For keys that the expert expects but aren't in pattern_matched_keys, 
        # we need to find unprivileged substitutes
        # This is the key insight: the expert expects unprivileged keys but we might have privileged keys
        
        # Get all possible keys the expert might expect based on the pattern
        pattern = eval_keys['mlp_keys']
        
        # Dynamically parse the pattern to extract expected keys
        # Remove word boundaries and split by | to get individual key patterns
        pattern_clean = pattern.replace('\\b', '').replace('(', '').replace(')', '')
        expected_keys = pattern_clean.split('|')
        
        # For each expected key, find the best substitute
        for expected_key in expected_keys:
            if expected_key not in filtered_obs:
                if expected_key in obs:
                    # Direct match
                    filtered_obs[expected_key] = obs[expected_key]
                else:
                    # Look for privileged version
                    privileged_key = expected_key.replace('_unprivileged', '')
                    if privileged_key in obs:
                        filtered_obs[expected_key] = obs[privileged_key]
                    else:
                        # Neither available, will substitute with zeros later
                        print(f"❌ WARNING: Neither {expected_key} nor {privileged_key} available")
        
        # Verify we have the expected number of features
        total_features = 0
        for key, value in filtered_obs.items():
            if isinstance(value, torch.Tensor):
                if value.dim() >= 2:
                    total_features += value.shape[1]
                else:
                    total_features += 1
            elif isinstance(value, np.ndarray):
                if value.ndim >= 2:
                    total_features += value.shape[1]
                else:
                    total_features += 1
            else:
                total_features += 1
        
        # Check if we have enough features for the expert
        if hasattr(expert_agent, 'total_mlp_size'):
            expected_features = expert_agent.total_mlp_size
            if total_features != expected_features:
                print(f"❌ WARNING: Feature mismatch! Expected {expected_features}, got {total_features}")
                
                # If we don't have enough features, we need to add zero tensors for missing keys
                if total_features < expected_features:
                    print(f"🔧 Adding zero tensors for missing features...")
                    # This is a simplified approach - in practice, we'd need to know the exact shapes
                    # For now, we'll just ensure we have all expected keys
                    for expected_key in expected_keys:
                        if expected_key not in filtered_obs:
                            # Create a zero tensor with appropriate shape
                            # This is a placeholder - we'd need to know the exact shape from the expert
                            print(f"🔧 Adding zero tensor for missing key: {expected_key}")
                            # We'll handle this in the tensor conversion step
        
        # Debug: Check what keys the expert agent expects
        # print(f"🔍 Expert eval_keys pattern: {eval_keys}")
        # print(f"🔍 Keys we're passing: {list(filtered_obs.keys())}")
        
        # Convert to tensor and add batch dimension
        obs_tensor = {}
        for key in expected_keys:  # Use expected_keys instead of filtered_obs.keys()
            if key in filtered_obs:
                value = filtered_obs[key]
                try:
                    if isinstance(value, np.ndarray):
                        obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
                    elif isinstance(value, torch.Tensor):
                        # Value is already a tensor, just ensure it's on the right device
                        # If it already has batch dimension, use as is
                        if value.dim() >= 2:
                            obs_tensor[key] = value.to(self.device)
                        else:
                            # Add batch dimension if needed
                            obs_tensor[key] = value.unsqueeze(0).to(self.device)
                    else:
                        # Scalar value, convert to tensor with batch dimension
                        obs_tensor[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
                except Exception as e:
                    print(f"❌ ERROR converting {key}: {e}")
                    print(f"❌ Value type: {type(value)}")
                    print(f"❌ Value shape: {getattr(value, 'shape', 'no shape')}")
                    raise RuntimeError(f"Failed to convert observation key {key}: {e}")
            else:
                # Key is missing, create zero tensor with appropriate shape
                # We need to infer the shape from the expert's expected input size
                if hasattr(expert_agent, 'mlp_key_sizes') and key in expert_agent.mlp_key_sizes:
                    # Use the expert's expected size for this key
                    expected_size = expert_agent.mlp_key_sizes[key]
                    obs_tensor[key] = torch.zeros(4, expected_size, device=self.device)
                else:
                    # Fallback: create a small zero tensor
                    obs_tensor[key] = torch.zeros(4, 1, device=self.device)
        # Only log after obs_tensor is fully constructed
        # Remove the excessive debug print that was printing keys on every call
        # print(f"{subset_name}: {list(obs_tensor.keys())}")
        
        # Debug: Show feature count for each key
        total_features_debug = 0
        for key, tensor in obs_tensor.items():
            if tensor.dim() >= 2:
                features = tensor.shape[1]
            else:
                features = 1
            total_features_debug += features
        # print(f"🔍 Total features: {total_features_debug}")
        
        # Get expert action logits for distillation
        with torch.no_grad():
            try:
                # Handle different expert agent types
                if hasattr(expert_agent, 'get_states'):
                    # LSTM-based agent (PPO-RNN)
                    # Ensure done tensor is properly shaped for the expert agent
                    if lstm_state is None:
                        # Create default LSTM states for evaluation
                        batch_size = obs_tensor[list(obs_tensor.keys())[0]].shape[0]
                        lstm_state = (
                            torch.zeros(expert_agent.lstm.num_layers, batch_size, expert_agent.lstm.hidden_size, device=self.device),
                            torch.zeros(expert_agent.lstm.num_layers, batch_size, expert_agent.lstm.hidden_size, device=self.device)
                        )
                    
                    # Create done tensor with proper shape
                    batch_size = obs_tensor[list(obs_tensor.keys())[0]].shape[0]
                    done_tensor = torch.zeros(batch_size, device=self.device)
                    
                    hidden, new_lstm_state = expert_agent.get_states(obs_tensor, lstm_state, done_tensor)
                else:
                    # Non-LSTM agent (PPO)
                    # The expert agent's mlp_keys are wrong - it shows privileged keys
                    # but the expert was trained with unprivileged keys
                    # So we bypass encode_observations and directly call the neural networks
                    
                    # Process MLP observations directly
                    mlp_features = []
                    batch_size = None
                    for key in obs_tensor.keys():
                        if key not in ['image']:  # Skip CNN keys for now
                            if batch_size is None:
                                batch_size = obs_tensor[key].shape[0]
                            # Ensure the observation is flattened
                            if obs_tensor[key].dim() > 2:
                                mlp_features.append(obs_tensor[key].view(batch_size, -1))
                            else:
                                mlp_features.append(obs_tensor[key])
                    
                    if mlp_features:
                        mlp_features = torch.cat(mlp_features, dim=1)
                        
                        # Call the expert's MLP encoder directly
                        if hasattr(expert_agent, 'mlp_encoder') and expert_agent.mlp_encoder is not None:
                            mlp_encoded = expert_agent.mlp_encoder(mlp_features)
                            
                            # Call the expert's latent projector directly
                            if hasattr(expert_agent, 'latent_projector'):
                                hidden = expert_agent.latent_projector(mlp_encoded)
                            else:
                                hidden = mlp_encoded
                        else:
                            # Fallback to encode_observations if direct approach fails
                            hidden = expert_agent.encode_observations(obs_tensor)
                    else:
                        # Fallback to encode_observations if no MLP features
                        hidden = expert_agent.encode_observations(obs_tensor)
                    
                    new_lstm_state = None
                
                # Get action logits from the actor network
                expert_logits = expert_agent.actor(hidden)
                
                # print(f"✅ Successfully got expert logits for {subset_name}: {expert_logits.shape}")
                return expert_logits, new_lstm_state
                
            except Exception as e:
                print(f"❌ ERROR in expert action computation for {subset_name}: {e}")
                print(f"❌ Expert agent type: {type(expert_agent)}")
                print(f"❌ Has get_states: {hasattr(expert_agent, 'get_states')}")
                print(f"❌ Obs tensor keys: {list(obs_tensor.keys())}")
                for key, tensor in obs_tensor.items():
                    print(f"❌ {key}: {tensor.shape}, {tensor.dtype}")
                
                # Additional debugging: let's see what the expert agent's encode_observations method does
                if hasattr(expert_agent, 'encode_observations'):
                    print(f"❌ Expert agent encode_observations method exists")
                    print(f"❌ Expert agent mlp_keys: {getattr(expert_agent, 'mlp_keys', 'N/A')}")
                    print(f"❌ Expert agent cnn_keys: {getattr(expert_agent, 'cnn_keys', 'N/A')}")
                    
                    # Try to understand what keys the expert agent will actually use
                    expert_mlp_keys = getattr(expert_agent, 'mlp_keys', [])
                    expert_cnn_keys = getattr(expert_agent, 'cnn_keys', [])
                    
                    keys_expert_will_use = []
                    for key in obs_tensor.keys():
                        if key in expert_mlp_keys or key in expert_cnn_keys:
                            keys_expert_will_use.append(key)
                    
                    print(f"❌ Keys expert will actually use: {keys_expert_will_use}")
                    
                    # Calculate features expert will actually get
                    actual_features = 0
                    for key in keys_expert_will_use:
                        if key in obs_tensor:
                            tensor = obs_tensor[key]
                            if tensor.dim() >= 2:
                                actual_features += tensor.shape[1]
                            else:
                                actual_features += 1
                    
                    print(f"❌ Features expert will actually get: {actual_features}")
                
                raise RuntimeError(f"Failed to compute expert action for {subset_name}: {e}")
    
    def get_all_expert_actions(self, obs: Dict, lstm_states: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Get action logits from all expert policies for distillation.
        
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
        # Remove excessive debug prints - only log when there's an actual change
        old_config_name = self.config_names[self.current_config_idx]
        
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
                if not callable(attr_value) or attr_name not in ['get_action_and_value', 'get_value', 'get_states', 'get_action_logits']:
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
        
        # Only print detailed logging when configuration actually changes
        if old_config_name != self.current_config_name:
            # Remove excessive debug prints
            # print(f"\n🔄 Cycling to configuration: {self.current_config_name}")
            # print(f"📋 Expert policy to mimic: {self.current_config_name}")
            # print(f"🔍 Observation filtering patterns:")
            # print(f"   MLP keys: {self.current_eval_keys['mlp_keys']}")
            # print(f"   CNN keys: {self.current_eval_keys['cnn_keys']}")
            # print("-" * 50)
            pass
    
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
        if current_time - self._last_mask_log_time > 60.0:  # Log every 60 seconds
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
            tuple: (action, logprob, entropy, value, new_lstm_state, expert_actions, student_logits)
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
        
        # Get student logits for distillation loss computation
        if self.student_policy_type == "ppo_rnn":
            hidden, _ = self.base_agent.get_states(masked_obs, lstm_state, done)
        else:
            hidden = self.base_agent.encode_observations(masked_obs)
        
        student_logits = self.base_agent.actor(hidden)
        
        # Get expert actions for distillation
        # Experts receive filtered observations (as they were trained)
        expert_actions = {}
        if self.expert_manager.expert_policies:
            try:
                expert_actions, _ = self.expert_manager.get_all_expert_actions(obs, {})
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Failed to get expert actions: {e}")
                print(f"❌ This is a critical error - expert actions are required for distillation!")
                print(f"❌ Available expert policies: {list(self.expert_manager.expert_policies.keys())}")
                print(f"❌ Current config: {config_name}")
                print(f"❌ Observation keys: {list(obs.keys())}")
                raise RuntimeError(f"Failed to get expert actions: {e}")
        else:
            print("❌ CRITICAL ERROR: No expert policies loaded!")
            print("❌ This means no distillation can occur!")
            raise RuntimeError("No expert policies loaded - distillation cannot proceed")
        
        # Verify we have expert actions for the current config
        if not expert_actions:
            print("❌ CRITICAL ERROR: No expert actions obtained!")
            raise RuntimeError("No expert actions obtained - distillation cannot proceed")
        
        if config_name not in expert_actions:
            print(f"❌ CRITICAL ERROR: Current config '{config_name}' not found in expert actions!")
            print(f"❌ Available expert actions: {list(expert_actions.keys())}")
            raise RuntimeError(f"Current config '{config_name}' not found in expert actions")
        
        return action, logprob, entropy, value, new_lstm_state, expert_actions, student_logits
    
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
        
        if not expert_actions:
            print("❌ CRITICAL ERROR: No expert actions provided for distillation loss!")
            raise RuntimeError("No expert actions provided for distillation loss")
        
        if config_name not in expert_actions:
            print(f"❌ CRITICAL ERROR: Current config '{config_name}' not found in expert actions!")
            print(f"❌ Available expert actions: {list(expert_actions.keys())}")
            raise RuntimeError(f"Current config '{config_name}' not found in expert actions")
        
        expert_logits = expert_actions[config_name]
        
        # Validate expert logits
        if expert_logits is None:
            print(f"❌ CRITICAL ERROR: Expert logits for {config_name} is None!")
            raise RuntimeError(f"Expert logits for {config_name} is None")
        
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