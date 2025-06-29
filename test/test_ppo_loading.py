#!/usr/bin/env python3
"""
Test script to verify PPO policy loading works correctly.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from subset_policies.load_subset_policy import SubsetPolicyLoader

def test_ppo_loading():
    """Test loading PPO policies from the policies/ppo/tigerdoorkey directory."""
    
    # Set device to CPU to avoid CUDA issues
    device = 'cpu'
    
    # Path to PPO policies
    policy_dir = "policies/ppo/tigerdoorkey"
    
    print(f"Testing PPO policy loading from: {policy_dir}")
    print(f"Using device: {device}")
    
    # Create policy loader
    loader = SubsetPolicyLoader(policy_dir, device=device)
    
    # List available policies
    print("\nAvailable policies:")
    loader.list_policies()
    
    # Test loading each policy
    for subset_name in loader.policies.keys():
        print(f"\n{'='*50}")
        print(f"Testing policy: {subset_name}")
        print(f"{'='*50}")
        
        # Load the policy
        agent, config, eval_keys = loader.load_policy(subset_name)
        print(f"✓ Successfully loaded {subset_name}")
        print(f"  MLP keys: {eval_keys['mlp_keys']}")
        print(f"  CNN keys: {eval_keys['cnn_keys']}")
        print(f"  Task: {getattr(config, 'task', 'Unknown')}")
    
    print(f"\n{'='*50}")
    print("Test completed!")
    print(f"{'='*50}")
    
    return True

if __name__ == "__main__":
    success = test_ppo_loading()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 