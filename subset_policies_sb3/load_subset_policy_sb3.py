#!/usr/bin/env python3
"""
Utility script to load trained SB3 subset policies and use them for inference.
This makes it easy to deploy trained policies without retraining.
Supports SB3 PPO policies.

Adapted from the CleanRL version to work with SB3's PPO implementation.
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
# Import SB3 PPO
from stable_baselines3 import PPO
from baselines.shared.masking_utils import mask_observations_for_student

class SubsetPolicyLoader:
    """Class to load and use trained SB3 subset policies."""
    
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
        self.task_name = None
        self.loaded_agents = {}  # Cache for loaded agents
        
        # Set default policy type
        self.policy_type = 'ppo'
        self.task_name = 'unknown'
        
        # Load all available policies
        self._load_policies()
    

    
    def _find_policy_files(self, policy_dir):
        """Find all policy files in the directory."""
        policies = {}
        if not os.path.exists(policy_dir):
            return policies
        
        # Look for subdirectories (env1_policy, env2_policy, etc.)
        for item in os.listdir(policy_dir):
            item_path = os.path.join(policy_dir, item)
            if os.path.isdir(item_path):
                # Look for .zip files in the subdirectory
                for file in os.listdir(item_path):
                    if file.endswith('.zip') and file.startswith('policy_'):
                        # Extract subset name from directory name (remove _policy suffix)
                        subset_name = item.replace('_policy', '')
                        policies[subset_name] = os.path.join(item_path, file)
        
        return policies
    
    def _load_policies(self):
        """Load all available policies from the directory."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Policy directory not found: {self.policy_dir}")
        
        # Find all policy files
        self.policies = self._find_policy_files(self.policy_dir)
        
        print(f"Loaded {len(self.policies)} {self.policy_type} policies: {list(self.policies.keys())}")
    
    def load_policy(self, subset_name):
        """
        Load a specific SB3 policy.
        
        Args:
            subset_name: Name of the subset (e.g., 'env1', 'env2')
            
        Returns:
            tuple: (agent, config, eval_keys)
        """
        if subset_name not in self.policies:
            raise ValueError(f"Policy {subset_name} not found. Available: {list(self.policies.keys())}")
        
        # Check if already loaded
        if subset_name in self.loaded_agents:
            return self.loaded_agents[subset_name]
        
        policy_path = self.policies[subset_name]
        
        print(f"\nLoading {self.policy_type} policy for {subset_name}")
        
        # Load the SB3 model
        agent = PPO.load(policy_path, device=self.device)
        
        # Get eval_keys from the model's custom attributes
        eval_keys = getattr(agent, 'eval_keys', {'mlp_keys': '.*', 'cnn_keys': '.*'})
        subset_name_from_model = getattr(agent, 'subset_name', subset_name)
        
        print(f"  MLP keys: {eval_keys['mlp_keys']}")
        print(f"  CNN keys: {eval_keys['cnn_keys']}")
        print(f"  Subset: {subset_name_from_model}")
        
        config = {}  # We don't have the original config
        
        # Cache the loaded agent
        self.loaded_agents[subset_name] = (agent, config, eval_keys)
        
        return agent, config, eval_keys
    
    def get_action(self, subset_name, obs, deterministic=True):
        """
        Get action from a specific policy.
        
        Args:
            subset_name: Name of the subset policy to use
            obs: Observation dictionary (unfiltered - contains all keys)
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: The action from the policy
        """
        # Load or get cached agent
        agent, config, eval_keys = self.load_policy(subset_name)
        
        # Convert observations to tensors for masking
        obs_tensors = {}
        for key, value in obs.items():
            if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
        
        # Get available keys from the environment
        available_keys = list(obs_tensors.keys())
        
        # Parse teacher keys from eval_keys patterns
        import re
        teacher_mlp_keys = []
        teacher_cnn_keys = []
        
        if 'mlp_keys' in eval_keys:
            mlp_pattern = re.compile(eval_keys['mlp_keys'])
            teacher_mlp_keys = [k for k in available_keys if mlp_pattern.search(k)]
        
        if 'cnn_keys' in eval_keys:
            cnn_pattern = re.compile(eval_keys['cnn_keys'])
            teacher_cnn_keys = [k for k in available_keys if cnn_pattern.search(k)]
        
        teacher_keys = teacher_mlp_keys + teacher_cnn_keys
        
        # Parse student keys from config.keys patterns (if available)
        student_keys = []
        if hasattr(config, 'keys') and config.keys:
            if hasattr(config.keys, 'mlp_keys'):
                student_mlp_pattern = re.compile(config.keys.mlp_keys)
                student_keys.extend([k for k in available_keys if student_mlp_pattern.search(k)])
            
            if hasattr(config.keys, 'cnn_keys'):
                student_cnn_pattern = re.compile(config.keys.cnn_keys)
                student_keys.extend([k for k in available_keys if student_cnn_pattern.search(k)])
        
        # If no student keys found, use teacher keys (fallback)
        if not student_keys:
            student_keys = teacher_keys
        
        # Mask observations for the student
        masked_obs = mask_observations_for_student(
            obs_tensors, 
            student_keys, 
            teacher_keys, 
            device=self.device,
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
        
        # Get action from agent
        with torch.no_grad():
            action, _ = agent.predict(obs_batch, deterministic=deterministic)
        
        # Extract scalar action if it's a numpy array
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        
        return action
    
    def list_policies(self):
        """List all available policies."""
        print(f"Available {self.policy_type} policies in {self.policy_dir}:")
        for subset_name in sorted(self.policies.keys()):
            print(f"  {subset_name}: {self.policies[subset_name]}")

def main():
    """Example usage of the policy loader."""
    parser = argparse.ArgumentParser(description='Load and use trained SB3 subset policies')
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