#!/usr/bin/env python3
"""
Example usage of the subset policy loader.
This script demonstrates how to load and use trained subset policies.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from subset_policies.load_subset_policy import SubsetPolicyLoader
import embodied
from embodied import wrappers

def create_env(task_name):
    """Create a simple environment for testing."""
    suite, task = task_name.split('_', 1)
    
    ctor = {
        'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
    }[suite]
    
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = __import__(module, fromlist=[cls])
        ctor = getattr(module, cls)
    
    env = ctor(task)
    
    # Apply standard wrappers
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        else:
            env = wrappers.NormalizeAction(env, name)

    env = wrappers.ExpandScalars(env)
    
    return env

def example_list_policies():
    """Example of listing available policies."""
    print("Listing available policies...")
    
    # Update this path to your actual policy directory
    policy_dir = "~/policies/ppo_rnn/tigerdoorkey"  # Replace with actual path
    
    try:
        loader = SubsetPolicyLoader(policy_dir, device='cpu')
        loader.list_policies()
    except FileNotFoundError:
        print(f"Policy directory not found: {policy_dir}")
        print("Please run train_subset_policies.sh first to create policies.")

def example_load_specific_policy():
    """Example of loading a specific policy."""
    print("Loading specific policy...")
    
    # Update this path to your actual policy directory
    policy_dir = "~/policies/ppo_rnn/tigerdoorkey"  # Replace with actual path
    
    try:
        loader = SubsetPolicyLoader(policy_dir, device='cpu')
        
        # Load a specific policy
        agent, config, eval_keys = loader.load_policy('env1')
        
        print(f"Loaded policy env1")
        print(f"MLP keys pattern: {eval_keys['mlp_keys']}")
        print(f"CNN keys pattern: {eval_keys['cnn_keys']}")
        print(f"Task: {config.task}")
        
    except FileNotFoundError:
        print(f"Policy directory not found: {policy_dir}")
        print("Please run train_subset_policies.sh first to create policies.")
    except ValueError as e:
        print(f"Error loading policy: {e}")

def example_get_action():
    """Example of getting actions from a policy."""
    print("Getting action from policy...")
    
    # Update this path to your actual policy directory
    example_policy_dir = "~/policies/ppo_rnn/tigerdoorkey"
    
    try:
        loader = SubsetPolicyLoader(example_policy_dir, device='cpu')
        
        # Create a dummy observation (replace with real observations)
        dummy_obs = {
            'tiger': [1.0, 0.0, 0.0],  # Example tiger observation
            'door': [0.0, 1.0, 0.0],   # Example door observation
            'key': [0.0, 0.0, 1.0],    # Example key observation
        }
        
        # Get action from env1 policy
        action, lstm_state = loader.get_action('env1', dummy_obs)
        
        print(f"Action from env1 policy: {action}")
        print(f"LSTM state shape: {lstm_state[0].shape if lstm_state else 'None'}")
        
    except FileNotFoundError:
        print(f"Policy directory not found: {example_policy_dir}")
        print("Please run train_subset_policies.sh first to create policies.")

if __name__ == "__main__":
    print("Subset Policy Usage Examples")
    print("=" * 40)
    
    # Run examples
    example_list_policies()
    print()
    
    example_load_specific_policy()
    print()
    
    example_get_action()
    print()
    
    print("Examples completed!") 