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
        Only select the keys from the raw obs dict that match the expert's eval_keys patterns. Raise an error if any expected key is missing.
        """
        import re
        if subset_name not in self.expert_policies:
            raise ValueError(f"Expert policy {subset_name} not found")
        expert_agent = self.expert_policies[subset_name]
        eval_keys = self.expert_eval_keys[subset_name]

        # Only select the keys that match the expert's eval_keys patterns
        filtered_obs = {}
        # Helper function to check if a key matches a pattern
        def matches_pattern(key, pattern):
            if pattern == '.*':
                return True
            elif pattern == '^$':
                return False
            else:
                return re.match(pattern, key) is not None

        # Get all keys that match the eval_keys pattern
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if matches_pattern(key, eval_keys['mlp_keys']) or matches_pattern(key, eval_keys['cnn_keys']):
                filtered_obs[key] = value

        # Dynamically parse the pattern to extract expected keys
        pattern = eval_keys['mlp_keys']
        pattern_clean = pattern.replace('\\b', '').replace('(', '').replace(')', '')
        expected_keys = pattern_clean.split('|')
        # Check that all expected keys are present
        for expected_key in expected_keys:
            if expected_key and expected_key not in filtered_obs:
                raise KeyError(f"Expected expert key '{expected_key}' not found in filtered_obs. Available keys: {list(filtered_obs.keys())}")

        # Print filtered_obs structure every 10 calls (debug)
        if not hasattr(self, '_expert_obs_debug_counter'):
            self._expert_obs_debug_counter = 0
        self._expert_obs_debug_counter += 1
        if self._expert_obs_debug_counter % 10 == 0:
            # print(f"[Expert Obs Debug] subset_name: {subset_name}")
            for k, v in filtered_obs.items():
                if isinstance(v, torch.Tensor):
                    shape = tuple(v.shape)
                    sample = v[0].cpu().numpy() if v.dim() > 0 else v.cpu().numpy()
                elif isinstance(v, np.ndarray):
                    shape = v.shape
                    sample = v[0] if v.ndim > 0 else v
                else:
                    shape = type(v)
                    sample = v
                # print(f"  key: {k}, shape: {shape}, sample: {sample}")

        # Convert to tensor and add batch dimension (existing logic)
        obs_tensor = {}
        for key in expected_keys:
            if key in filtered_obs:
                value = filtered_obs[key]
                try:
                    if isinstance(value, np.ndarray):
                        obs_tensor[key] = torch.tensor(value, device=self.device)
                    elif isinstance(value, torch.Tensor):
                        obs_tensor[key] = value.to(self.device)
                    else:
                        obs_tensor[key] = torch.tensor(value, device=self.device)
                except Exception as e:
                    print(f"❌ ERROR converting key {key} to tensor: {e}")
                    raise

        # Forward through expert agent
        if hasattr(expert_agent, 'get_action_logits'):
            expert_logits = expert_agent.get_action_logits(obs_tensor)
            return expert_logits, None
        elif hasattr(expert_agent, 'actor') and hasattr(expert_agent, 'encode_observations'):
            # Standard PPO agent
            hidden = expert_agent.encode_observations(obs_tensor)
            expert_logits = expert_agent.actor(hidden)
            return expert_logits, None
        elif hasattr(expert_agent, 'get_action_and_value'):
            # PPO-RNN or similar
            result = expert_agent.get_action_and_value(obs_tensor, lstm_state)
            # result: (action, logprob, entropy, value, new_lstm_state) or (action, logprob, entropy, value, new_lstm_state, logits)
            if len(result) == 6:
                # (action, logprob, entropy, value, new_lstm_state, logits)
                return result[-1], result[4]
            else:
                # Fallback: try to reconstruct logits
                if hasattr(expert_agent, 'get_states'):
                    hidden, _ = expert_agent.get_states(obs_tensor, lstm_state)
                    expert_logits = expert_agent.actor(hidden)
                    return expert_logits, result[4]
                else:
                    raise RuntimeError(f"Expert agent for {subset_name} get_action_and_value did not return logits and cannot reconstruct.")
        else:
            raise RuntimeError(f"Expert agent for {subset_name} does not have a recognized method to get logits.")
    
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
        
        if not hasattr(self, '_last_mask_debug_time'):
            self._last_mask_debug_time = 0
        
        current_time = time.time()
        if current_time - self._last_mask_debug_time > 30.0:  # Log every 30 seconds
            print(f"🎭 _mask_observations debug for {self.current_config_name}:")
            print(f"  MLP pattern: {mlp_pattern}")
            print(f"  CNN pattern: {cnn_pattern}")
            print(f"  Available keys: {list(obs.keys())}")
            self._last_mask_debug_time = current_time
        
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
        zeroed_keys = []
        for key in all_keys:
            if final_obs[key] is not None:
                if isinstance(final_obs[key], torch.Tensor):
                    # Ensure tensor is float32
                    if final_obs[key].dtype != torch.float32:
                        masked_obs[key] = final_obs[key].float()
                    else:
                        masked_obs[key] = final_obs[key].clone()
                else:
                    # Convert to tensor and ensure float32 dtype
                    masked_obs[key] = torch.tensor(final_obs[key], device=self.device, dtype=torch.float32)
            else:
                # Zero out missing observations
                zeroed_keys.append(key)
                if key in obs:
                    if isinstance(obs[key], torch.Tensor):
                        # Create zero tensor with same shape and ensure float32
                        masked_obs[key] = torch.zeros_like(obs[key], dtype=torch.float32)
                    else:
                        # Create zero tensor with expected shape and float32
                        masked_obs[key] = torch.zeros_like(torch.tensor(obs[key], device=self.device, dtype=torch.float32))
                else:
                    # Fallback: create zero tensor with expected shape and float32
                    # This shouldn't happen in practice, but just in case
                    masked_obs[key] = torch.zeros(1, device=self.device, dtype=torch.float32)
        
        # Debug: Show zeroed keys
        if zeroed_keys and current_time - self._last_mask_debug_time > 30.0:
            print(f"🎭 Zeroed keys for {self.current_config_name}: {zeroed_keys}")
        
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
        Every 100 steps, print the action chosen by the student and the expert for the first environment in the batch.
        """
        # Get current configuration
        config_name, eval_keys = self.get_current_config()
        
        # Mask observations based on current configuration
        masked_obs = self._mask_observations(obs, eval_keys)
        
        # Get student action and value using masked observations
        if self.student_policy_type == "ppo_rnn":
            action, logprob, entropy, value, new_lstm_state = self.base_agent.get_action_and_value(
                masked_obs, lstm_state, done, action
            )
        else:
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
        
        # Create comprehensive observations for all expert policies
        # This ensures that all expert policies get the keys they need
        comprehensive_obs = self._create_comprehensive_observations(obs)
        
        # Get expert actions for distillation
        expert_actions = {}
        if self.expert_manager.expert_policies:
            try:
                expert_actions, _ = self.expert_manager.get_all_expert_actions(comprehensive_obs, {})
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Failed to get expert actions: {e}")
                print(f"❌ This is a critical error - expert actions are required for distillation!")
                print(f"❌ Available expert policies: {list(self.expert_manager.expert_policies.keys())}")
                print(f"❌ Current config: {config_name}")
                print(f"❌ Original observation keys: {list(obs.keys())}")
                print(f"❌ Comprehensive observation keys: {list(comprehensive_obs.keys())}")
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

        # Print student and expert actions and logits for the first environment in the batch every 100 steps
        if not hasattr(self, '_action_debug_counter'):
            self._action_debug_counter = 0
        self._action_debug_counter += 1
        if self._action_debug_counter % 100 == 0:
            # Student action (first env)
            if hasattr(action[0], 'numel') and action[0].numel() == 1:
                student_action = action[0].item()
            elif hasattr(action[0], 'tolist'):
                student_action = action[0].tolist()
            else:
                student_action = action[0]
            # Expert logits and action (first env, current config)
            expert_logits = expert_actions[config_name]
            if hasattr(expert_logits[0], 'argmax'):
                expert_action = expert_logits[0].argmax().item() if expert_logits[0].numel() > 1 else expert_logits[0].item()
            elif hasattr(expert_logits[0], 'tolist'):
                expert_action = expert_logits[0].tolist()
            else:
                expert_action = expert_logits[0]
            # Print logits for both student and expert
            student_logits_first = student_logits[0].tolist() if hasattr(student_logits[0], 'tolist') else student_logits[0]
            expert_logits_first = expert_logits[0].tolist() if hasattr(expert_logits[0], 'tolist') else expert_logits[0]
            print(f"[Action Debug] Student action: {student_action}, Expert action: {expert_action}")
            print(f"[Logits Debug] Student logits: {student_logits_first}, Expert logits: {expert_logits_first}")
        
        return action, logprob, entropy, value, new_lstm_state, expert_actions, student_logits
    
    def _create_comprehensive_observations(self, obs: Dict) -> Dict:
        """
        Create comprehensive observations that include all keys that any expert policy might need.
        This ensures that expert policies get access to their required observation keys.
        
        Args:
            obs: Original observations from environment
            
        Returns:
            Dict: Comprehensive observations with all possible keys
        """
        comprehensive_obs = {}
        
        # Convert ALL observations to float32 tensors, not just the ones the student agent expects
        # Expert policies might need different keys than the student agent
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                comprehensive_obs[key] = value
                continue
                
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float32:
                    comprehensive_obs[key] = value.float()
                else:
                    comprehensive_obs[key] = value
            else:
                comprehensive_obs[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
        
        # DO NOT create fake observations! Only use what the environment provides.
        # If expert policies need unprivileged keys, the environment should provide them.
        # If they're missing, that's a problem that needs to be fixed at the environment level.
        
        return comprehensive_obs
    
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

        # Debug: Print mean/std of logits
        # print(f"[Distill Debug] Student logits mean: {student_logits.mean().item():.4f}, std: {student_logits.std().item():.4f}")
        # print(f"[Distill Debug] Expert logits mean: {expert_logits.mean().item():.4f}, std: {expert_logits.std().item():.4f}")

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        expert_probs = F.softmax(expert_logits, dim=-1)
        distill_loss = F.kl_div(
            student_log_probs, expert_probs,
            reduction='batchmean'
        )
        # print(f"[Distill Debug] Distillation loss: {distill_loss.item():.6f}")
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