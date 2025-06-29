#!/usr/bin/env python3
"""
Test script to verify PPO policy loading works correctly.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines.shared.policy_utils import load_policy_like_subset_policies

def test_ppo_loading():
    """Test loading PPO policies from the policies/ppo/tigerdoorkey directory."""
    
    # Set device to CPU to avoid CUDA issues
    device = 'cpu'
    
    # Path to PPO policies
    policy_dir = "policies/ppo/tigerdoorkey"
    
    print(f"Testing PPO policy loading from: {policy_dir}")
    print(f"Using device: {device}")
    
    # Load all policies using the shared utility
    loaded_policies = load_policy_like_subset_policies(policy_dir, 'ppo', device)
    
    print(f"\nLoaded {len(loaded_policies)} policies: {list(loaded_policies.keys())}")
    
    # Test each loaded policy
    for subset_name, (agent, config, eval_keys) in loaded_policies.items():
        print(f"\n{'='*50}")
        print(f"Testing policy: {subset_name}")
        print(f"{'='*50}")
        
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