#!/usr/bin/env python3
"""
Test script to verify PPO policy loading works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from subset_policies.load_subset_policy import SubsetPolicyLoader

def test_ppo_policy_loading():
    """Test loading PPO policies from the policies/ppo directory."""
    
    print("Testing PPO policy loading...")
    print("=" * 50)
    
    # Path to PPO policies
    ppo_policy_dir = "policies/ppo/tigerdoorkey"
    
    if not os.path.exists(ppo_policy_dir):
        print(f"Error: PPO policy directory not found: {ppo_policy_dir}")
        print("Please run 'POLICY_TYPE=ppo ./train_subset_policies.sh' first to create PPO policies.")
        return False
    
    try:
        # Create policy loader
        print(f"Loading policies from: {ppo_policy_dir}")
        loader = SubsetPolicyLoader(ppo_policy_dir, device='cpu')
        
        # List available policies
        print("\nAvailable policies:")
        loader.list_policies()
        
        # Test loading each policy
        print("\nTesting policy loading:")
        for subset_name in ['env1', 'env2', 'env3', 'env4']:
            print(f"\nLoading {subset_name}...")
            agent, config, eval_keys = loader.load_policy(subset_name)
            
            print(f"  ✓ Successfully loaded {subset_name}")
            print(f"  Policy type: {loader.policy_type}")
            print(f"  MLP keys: {eval_keys['mlp_keys']}")
            print(f"  CNN keys: {eval_keys['cnn_keys']}")
            print(f"  Task: {config.task}")
        
        print("\n" + "=" * 50)
        print("✓ All PPO policies loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during policy loading: {e}")
        return False

if __name__ == "__main__":
    success = test_ppo_policy_loading()
    
    if success:
        print("\nTest completed successfully!")
        print("PPO policies can be loaded correctly.")
        print("Exiting as requested...")
        exit(0)
    else:
        print("\nTest failed!")
        print("There are issues with PPO policy loading.")
        exit(1) 