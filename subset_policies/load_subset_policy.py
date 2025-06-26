#!/usr/bin/env python3
"""
Utility script to load trained subset policies and use them for inference.
This makes it easy to deploy trained policies without retraining.
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
from baselines.ppo_rnn.ppo_rnn import PPORnnAgent
from baselines.shared.eval_utils import filter_observations_by_keys

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
        
        # Load metadata
        metadata_path = os.path.join(policy_dir, 'metadata.yaml')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = yaml.safe_load(f)
        
        # Load all available policies
        self._load_policies()
    
    def _load_policies(self):
        """Load all available policies from the directory."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Policy directory not found: {self.policy_dir}")
        
        # Find all policy directories
        for item in os.listdir(self.policy_dir):
            item_path = os.path.join(self.policy_dir, item)
            if os.path.isdir(item_path) and item.startswith('env'):
                policy_path = os.path.join(item_path, 'policy.pt')
                if os.path.exists(policy_path):
                    self.policies[item] = policy_path
        
        print(f"Loaded {len(self.policies)} policies: {list(self.policies.keys())}")
    
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
        checkpoint = torch.load(policy_path, map_location=self.device)
        
        # Extract components
        agent_state_dict = checkpoint['agent_state_dict']
        config = checkpoint['config']
        eval_keys = checkpoint['eval_keys']
        
        # Create environment to get observation space
        env = self._create_env(config)
        
        # Create agent
        agent = PPORnnAgent(
            env.obs_space,
            env.act_space,
            config,
            device=self.device
        )
        
        # Load state dict
        agent.load_state_dict(agent_state_dict)
        agent.eval()
        
        return agent, config, eval_keys
    
    def _create_env(self, config):
        """Create a single environment for getting observation space."""
        suite, task = config.task.split('_', 1)
        
        ctor = {
            'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
        }[suite]
        
        if isinstance(ctor, str):
            module, cls = ctor.split(':')
            module = __import__(module, fromlist=[cls])
            ctor = getattr(module, cls)
        
        kwargs = getattr(config.env, suite, {})
        env = ctor(task, **kwargs)
        return self._wrap_env(env, config)
    
    def _wrap_env(self, env, config):
        """Wrap environment with standard wrappers."""
        args = getattr(config, 'wrapper', {})
        for name, space in env.act_space.items():
            if name == 'reset':
                continue
            elif space.discrete:
                env = wrappers.OneHotAction(env, name)
            else:
                env = wrappers.NormalizeAction(env, name)

        env = wrappers.ExpandScalars(env)

        if hasattr(args, 'length') and args.length:
            env = wrappers.TimeLimit(env, args.length, getattr(args, 'reset', True))

        for name, space in env.act_space.items():
            if not space.discrete:
                env = wrappers.ClipAction(env, name)

        return env
    
    def get_action(self, subset_name, obs, lstm_state=None):
        """
        Get action from a specific policy.
        
        Args:
            subset_name: Name of the subset policy to use
            obs: Observation dictionary
            lstm_state: LSTM state (optional)
            
        Returns:
            tuple: (action, lstm_state)
        """
        agent, config, eval_keys = self.load_policy(subset_name)
        
        # Filter observations based on the subset's eval_keys
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
        
        # Get action from agent
        with torch.no_grad():
            action, lstm_state = agent.get_action(obs_tensor, lstm_state)
        
        return action, lstm_state
    
    def list_policies(self):
        """List all available policies."""
        print(f"Available policies in {self.policy_dir}:")
        for subset_name in sorted(self.policies.keys()):
            print(f"  {subset_name}: {self.policies[subset_name]}")
        
        if self.metadata:
            print(f"\nTask: {self.metadata.get('task', 'Unknown')}")
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
        print(f"Loaded policy {args.subset}")
        print(f"MLP keys: {eval_keys['mlp_keys']}")
        print(f"CNN keys: {eval_keys['cnn_keys']}")
    else:
        # List all policies
        loader.list_policies()

if __name__ == "__main__":
    main() 