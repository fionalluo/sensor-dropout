#!/usr/bin/env python3
"""
Utility script to load trained subset policies and use them for inference.
This makes it easy to deploy trained policies without retraining.
Supports both PPO and PPO-RNN policies.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import embodied
from embodied import wrappers
# Import both PPO and PPO-RNN agents
from baselines.ppo.agent import PPOAgent
from baselines.ppo_rnn.ppo_rnn import PPORnnAgent
from baselines.shared.eval_utils import filter_observations_by_keys
from baselines.shared.policy_utils import (
    load_policy_with_metadata, 
    find_policy_files, 
    load_metadata_from_dir,
    convert_obs_to_tensor
)

class SubsetPolicyLoader:
    """Class to load and use trained subset policies."""
    
    def __init__(self, policy_dir, device='cpu'):
        """
        Initialize the policy loader.
        
        Args:
            policy_dir: Directory containing the trained policies
            device: Device to load the policy on
        """
        self.policy_dir = policy_dir
        self.device = device
        self.policies = {}
        self.metadata = None
        self.policy_type = None
        
        # Load metadata
        self.metadata = load_metadata_from_dir(policy_dir)
        if self.metadata:
            self.policy_type = self.metadata.get('policy_type', 'ppo_rnn')
        
        # Load all available policies
        self._load_policies()
    
    def _load_policies(self):
        """Load all available policies from the directory."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Policy directory not found: {self.policy_dir}")
        
        # Find all policy files using shared utility
        self.policies = find_policy_files(self.policy_dir)
        
        print(f"Loaded {len(self.policies)} {self.policy_type} policies: {list(self.policies.keys())}")
    
    def _get_agent_class(self, policy_type=None):
        """Get the appropriate agent class based on policy type."""
        if policy_type is None:
            policy_type = self.policy_type
        
        if policy_type == 'ppo':
            return PPOAgent
        elif policy_type == 'ppo_rnn':
            return PPORnnAgent
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def load_policy(self, subset_name):
        """
        Load a specific policy.
        
        Args:
            subset_name: Name of the subset (e.g., 'env1', 'env2')
            
        Returns:
            tuple: (agent, config, eval_keys)
        """
        if subset_name not in self.policies:
            raise ValueError(f"Policy {subset_name} not found. Available: {list(self.policies.keys())}")
        
        policy_path = self.policies[subset_name]
        
        # Get the appropriate agent class
        agent_class = self._get_agent_class()
        
        # Use shared utility to load policy
        agent, config, metadata = load_policy_with_metadata(
            policy_path, agent_class, self.device
        )
        
        # Extract eval_keys from metadata
        eval_keys = metadata.get('eval_keys', {})
        
        return agent, config, eval_keys
    
    def get_action(self, subset_name, obs, lstm_state=None):
        """
        Get action from a specific policy.
        
        Args:
            subset_name: Name of the subset policy to use
            obs: Observation dictionary
            lstm_state: LSTM state (optional, only for PPO-RNN)
            
        Returns:
            tuple: (action, lstm_state) - lstm_state is None for PPO
        """
        agent, config, eval_keys = self.load_policy(subset_name)
        
        # Filter observations based on the subset's eval_keys
        filtered_obs = filter_observations_by_keys(
            obs, eval_keys['mlp_keys'], eval_keys['cnn_keys']
        )
        
        # Convert to tensor using shared utility
        obs_tensor = convert_obs_to_tensor(filtered_obs, self.device)
        
        # Add batch dimension if needed
        for key in obs_tensor:
            if obs_tensor[key].dim() == 1:
                obs_tensor[key] = obs_tensor[key].unsqueeze(0)
        
        # Get action from agent based on policy type
        with torch.no_grad():
            if self.policy_type == 'ppo':
                # PPO doesn't use LSTM states
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                return action, None
            else:  # ppo_rnn
                # PPO-RNN uses LSTM states
                action, lstm_state = agent.get_action(obs_tensor, lstm_state)
                return action, lstm_state
    
    def list_policies(self):
        """List all available policies."""
        print(f"Available {self.policy_type} policies in {self.policy_dir}:")
        for subset_name in sorted(self.policies.keys()):
            print(f"  {subset_name}: {self.policies[subset_name]}")
        
        if self.metadata:
            print(f"\nTask: {self.metadata.get('task', 'Unknown')}")
            print(f"Policy type: {self.metadata.get('policy_type', 'Unknown')}")
            print(f"Number of eval configs: {self.metadata.get('num_eval_configs', 'Unknown')}")

def main():
    """Example usage of the policy loader."""
    parser = argparse.ArgumentParser(description='Load and use trained subset policies')
    parser.add_argument('--policy_dir', type=str, required=True,
                       help='Directory containing trained policies')
    parser.add_argument('--subset', type=str, default=None,
                       help='Specific subset to load (e.g., env1)')
    parser.add_argument('--list', action='store_true',
                       help='List available policies')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to load policies on')
    
    args = parser.parse_args()
    
    # Create policy loader
    loader = SubsetPolicyLoader(args.policy_dir, device=args.device)
    
    if args.list:
        loader.list_policies()
        return
    
    if args.subset:
        # Load specific policy
        agent, config, eval_keys = loader.load_policy(args.subset)
        print(f"Loaded {loader.policy_type} policy {args.subset}")
        print(f"MLP keys: {eval_keys['mlp_keys']}")
        print(f"CNN keys: {eval_keys['cnn_keys']}")
    else:
        # List all policies
        loader.list_policies()

if __name__ == "__main__":
    main() 