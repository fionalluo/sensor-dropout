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
    
    # Example for PPO policies
    print("\n--- PPO Policies ---")
    ppo_policy_dir = "~/policies/ppo/tigerdoorkey"  # Replace with actual path
    
    try:
        ppo_loader = SubsetPolicyLoader(ppo_policy_dir, device='cpu')
        ppo_loader.list_policies()
    except FileNotFoundError:
        print(f"PPO policy directory not found: {ppo_policy_dir}")
        print("Please run 'POLICY_TYPE=ppo ./train_subset_policies.sh' first to create PPO policies.")
    
    # Example for PPO-RNN policies
    print("\n--- PPO-RNN Policies ---")
    ppo_rnn_policy_dir = "~/policies/ppo_rnn/tigerdoorkey"  # Replace with actual path
    
    try:
        ppo_rnn_loader = SubsetPolicyLoader(ppo_rnn_policy_dir, device='cpu')
        ppo_rnn_loader.list_policies()
    except FileNotFoundError:
        print(f"PPO-RNN policy directory not found: {ppo_rnn_policy_dir}")
        print("Please run './train_subset_policies.sh' first to create PPO-RNN policies.")

def example_load_specific_policy():
    """Example of loading a specific policy."""
    print("Loading specific policies...")
    
    # Example for PPO policy
    print("\n--- Loading PPO Policy ---")
    ppo_policy_dir = "~/policies/ppo/tigerdoorkey"  # Replace with actual path
    
    try:
        ppo_loader = SubsetPolicyLoader(ppo_policy_dir, device='cpu')
        
        # Load a specific policy
        agent, config, eval_keys = ppo_loader.load_policy('env1')
        
        print(f"Loaded {ppo_loader.policy_type} policy env1")
        print(f"MLP keys pattern: {eval_keys['mlp_keys']}")
        print(f"CNN keys pattern: {eval_keys['cnn_keys']}")
        print(f"Task: {config.task}")
        
    except FileNotFoundError:
        print(f"PPO policy directory not found: {ppo_policy_dir}")
        print("Please run 'POLICY_TYPE=ppo ./train_subset_policies.sh' first to create PPO policies.")
    except ValueError as e:
        print(f"Error loading PPO policy: {e}")
    
    # Example for PPO-RNN policy
    print("\n--- Loading PPO-RNN Policy ---")
    ppo_rnn_policy_dir = "~/policies/ppo_rnn/tigerdoorkey"  # Replace with actual path
    
    try:
        ppo_rnn_loader = SubsetPolicyLoader(ppo_rnn_policy_dir, device='cpu')
        
        # Load a specific policy
        agent, config, eval_keys = ppo_rnn_loader.load_policy('env1')
        
        print(f"Loaded {ppo_rnn_loader.policy_type} policy env1")
        print(f"MLP keys pattern: {eval_keys['mlp_keys']}")
        print(f"CNN keys pattern: {eval_keys['cnn_keys']}")
        print(f"Task: {config.task}")
        
    except FileNotFoundError:
        print(f"PPO-RNN policy directory not found: {ppo_rnn_policy_dir}")
        print("Please run './train_subset_policies.sh' first to create PPO-RNN policies.")
    except ValueError as e:
        print(f"Error loading PPO-RNN policy: {e}")

def example_get_action():
    """Example of getting actions from policies."""
    print("Getting actions from policies...")
    
    # Example for PPO policy
    print("\n--- PPO Policy Action ---")
    ppo_policy_dir = "~/policies/ppo/tigerdoorkey"
    
    try:
        ppo_loader = SubsetPolicyLoader(ppo_policy_dir, device='cpu')
        
        # Create a dummy observation (replace with real observations)
        dummy_obs = {
            'tiger': [1.0, 0.0, 0.0],  # Example tiger observation
            'door': [0.0, 1.0, 0.0],   # Example door observation
            'key': [0.0, 0.0, 1.0],    # Example key observation
        }
        
        # Get action from env1 policy (PPO doesn't use LSTM states)
        action, lstm_state = ppo_loader.get_action('env1', dummy_obs)
        
        print(f"Action from PPO env1 policy: {action}")
        print(f"LSTM state: {lstm_state} (None for PPO)")
        
    except FileNotFoundError:
        print(f"PPO policy directory not found: {ppo_policy_dir}")
        print("Please run 'POLICY_TYPE=ppo ./train_subset_policies.sh' first to create PPO policies.")
    
    # Example for PPO-RNN policy
    print("\n--- PPO-RNN Policy Action ---")
    ppo_rnn_policy_dir = "~/policies/ppo_rnn/tigerdoorkey"
    
    try:
        ppo_rnn_loader = SubsetPolicyLoader(ppo_rnn_policy_dir, device='cpu')
        
        # Create a dummy observation (replace with real observations)
        dummy_obs = {
            'tiger': [1.0, 0.0, 0.0],  # Example tiger observation
            'door': [0.0, 1.0, 0.0],   # Example door observation
            'key': [0.0, 0.0, 1.0],    # Example key observation
        }
        
        # Get action from env1 policy (PPO-RNN uses LSTM states)
        action, lstm_state = ppo_rnn_loader.get_action('env1', dummy_obs)
        
        print(f"Action from PPO-RNN env1 policy: {action}")
        print(f"LSTM state shape: {lstm_state[0].shape if lstm_state else 'None'}")
        
    except FileNotFoundError:
        print(f"PPO-RNN policy directory not found: {ppo_rnn_policy_dir}")
        print("Please run './train_subset_policies.sh' first to create PPO-RNN policies.")

def example_compare_policies():
    """Example of comparing different policy types."""
    print("Comparing PPO vs PPO-RNN policies...")
    
    # Example observation
    dummy_obs = {
        'tiger': [1.0, 0.0, 0.0],
        'door': [0.0, 1.0, 0.0],
        'key': [0.0, 0.0, 1.0],
    }
    
    # Try to load both policy types
    ppo_policy_dir = "~/policies/ppo/tigerdoorkey"
    ppo_rnn_policy_dir = "~/policies/ppo_rnn/tigerdoorkey"
    
    try:
        ppo_loader = SubsetPolicyLoader(ppo_policy_dir, device='cpu')
        ppo_action, _ = ppo_loader.get_action('env1', dummy_obs)
        print(f"PPO action: {ppo_action}")
    except FileNotFoundError:
        print("PPO policies not found")
    
    try:
        ppo_rnn_loader = SubsetPolicyLoader(ppo_rnn_policy_dir, device='cpu')
        ppo_rnn_action, lstm_state = ppo_rnn_loader.get_action('env1', dummy_obs)
        print(f"PPO-RNN action: {ppo_rnn_action}")
        print(f"PPO-RNN LSTM state: {lstm_state is not None}")
    except FileNotFoundError:
        print("PPO-RNN policies not found")

if __name__ == "__main__":
    print("Subset Policy Loader Examples")
    print("=" * 50)
    
    # Run examples
    example_list_policies()
    example_load_specific_policy()
    example_get_action()
    example_compare_policies()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run these examples with real policies:")
    print("1. Train PPO policies: POLICY_TYPE=ppo ./train_subset_policies.sh")
    print("2. Train PPO-RNN policies: ./train_subset_policies.sh")
    print("3. Update the policy_dir paths in this script to match your actual paths") 