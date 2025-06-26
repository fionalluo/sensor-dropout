#!/usr/bin/env python3
"""
Example script showing how to use trained subset policies.
This demonstrates loading policies and using them for inference.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from load_subset_policy import SubsetPolicyLoader
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

def example_usage():
    """Example of how to use trained subset policies."""
    
    # Example policy directory (adjust this path)
    policy_dir = "~/policies/gymnasium_tigerkeydoor_12345678"  # Replace with actual path
    
    print("Loading subset policies...")
    loader = SubsetPolicyLoader(policy_dir, device='cpu')
    
    # List available policies
    loader.list_policies()
    
    # Create environment for testing
    env = create_env("gymnasium_TigerDoorKey-v0")
    
    print("\nTesting policies with environment...")
    
    # Test each policy
    for subset_name in sorted(loader.policies.keys()):
        print(f"\n--- Testing {subset_name} ---")
        
        # Reset environment
        obs = env.reset()
        
        # Initialize LSTM state
        lstm_state = None
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            # Get action from policy
            action, lstm_state = loader.get_action(subset_name, obs, lstm_state)
            
            # Convert action back to environment format
            if isinstance(action, dict):
                # Handle one-hot actions
                for key, value in action.items():
                    if hasattr(env.act_space[key], 'discrete') and env.act_space[key].discrete:
                        action[key] = value.argmax(dim=-1).cpu().numpy()[0]
                    else:
                        action[key] = value.cpu().numpy()[0]
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step}: Action={action}, Reward={reward:.2f}")
            
            if done:
                break
        
        print(f"  Total reward: {total_reward:.2f}")

def example_load_specific_policy():
    """Example of loading a specific policy."""
    
    policy_dir = "~/policies/gymnasium_tigerkeydoor_12345678"  # Replace with actual path
    
    print("Loading specific policy...")
    loader = SubsetPolicyLoader(policy_dir, device='cpu')
    
    # Load a specific policy
    agent, config, eval_keys = loader.load_policy('env1')
    
    print(f"Loaded policy env1")
    print(f"Task: {config.task}")
    print(f"MLP keys pattern: {eval_keys['mlp_keys']}")
    print(f"CNN keys pattern: {eval_keys['cnn_keys']}")
    
    # You can now use the agent directly
    print(f"Agent type: {type(agent)}")

if __name__ == "__main__":
    print("Subset Policy Usage Examples")
    print("=" * 40)
    
    # Check if policy directory exists
    example_policy_dir = "~/policies/gymnasium_tigerkeydoor_12345678"
    if os.path.exists(os.path.expanduser(example_policy_dir)):
        example_usage()
        example_load_specific_policy()
    else:
        print(f"Example policy directory not found: {example_policy_dir}")
        print("Please run train_subset_policies.sh first to create policies.")
        print("\nTo use with your own policies:")
        print("1. Update the policy_dir path in this script")
        print("2. Run the script again") 