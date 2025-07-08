#!/usr/bin/env python3
"""
Test script to verify SB3 PPO policy loading works correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from subset_policies.load_subset_policy import SubsetPolicyLoader

def test_ppo_loading():
    """Test loading SB3 PPO policies from the policies/ppo/tigerdoorkey directory."""
    
    # Set device to CPU to avoid CUDA issues
    device = 'cpu'
    
    # Path to SB3 PPO policies
    policy_dir = "policies/ppo/tigerdoorkey"
    
    print(f"Testing SB3 PPO policy loading from: {policy_dir}")
    print(f"Using device: {device}")
    
    # Check if policy directory exists
    if not os.path.exists(policy_dir):
        print(f"❌ Policy directory not found: {policy_dir}")
        print("Please run train_subset_policies.sh first to generate policies")
        return False
    
    # Create policy loader
    try:
        loader = SubsetPolicyLoader(policy_dir, device=device)
    except Exception as e:
        print(f"❌ Failed to create policy loader: {e}")
        return False
    
    print(f"\nLoaded {len(loader.policies)} policies: {list(loader.policies.keys())}")
    
    # Test each loaded policy
    for subset_name in loader.policies.keys():
        print(f"\n{'='*50}")
        print(f"Testing policy: {subset_name}")
        print(f"{'='*50}")
        
        try:
            # Load the policy
            agent, config, eval_keys = loader.load_policy(subset_name)
            
            print(f"✓ Successfully loaded {subset_name}")
            print(f"  MLP keys: {eval_keys['mlp_keys']}")
            print(f"  CNN keys: {eval_keys['cnn_keys']}")
            print(f"  Subset name from model: {getattr(agent, 'subset_name', 'Not found')}")
            print(f"  Eval keys from model: {getattr(agent, 'eval_keys', 'Not found')}")
            print(f"  Policy path: {loader.policies[subset_name]}")
            
        except Exception as e:
            print(f"❌ Failed to load/test {subset_name}: {e}")
            return False
    
    print(f"\n{'='*50}")
    print("Test completed!")
    print(f"{'='*50}")
    
    return True

def test_ppo_specific_policy():
    """Test loading a specific SB3 PPO policy."""
    
    device = 'cpu'
    policy_dir = "policies/ppo/tigerdoorkey"
    subset_name = "env1"  # Test with env1
    
    print(f"\nTesting specific policy loading: {subset_name}")
    print(f"Policy directory: {policy_dir}")
    
    if not os.path.exists(policy_dir):
        print(f"❌ Policy directory not found: {policy_dir}")
        return False
    
    try:
        loader = SubsetPolicyLoader(policy_dir, device=device)
        
        if subset_name not in loader.policies:
            print(f"❌ Policy {subset_name} not found. Available: {list(loader.policies.keys())}")
            return False
        
        # Load specific policy
        agent, config, eval_keys = loader.load_policy(subset_name)
        
        print(f"✓ Successfully loaded {subset_name}")
        print(f"  Policy path: {loader.policies[subset_name]}")
        print(f"  MLP keys: {eval_keys['mlp_keys']}")
        print(f"  CNN keys: {eval_keys['cnn_keys']}")
        print(f"  Subset name from model: {getattr(agent, 'subset_name', 'Not found')}")
        print(f"  Eval keys from model: {getattr(agent, 'eval_keys', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load specific policy: {e}")
        return False

if __name__ == "__main__":
    print("Testing SB3 PPO policy loading...")
    
    # Test 1: Load all policies
    success1 = test_ppo_loading()
    
    # Test 2: Load specific policy
    success2 = test_ppo_specific_policy()
    
    if success1 and success2:
        print("\n🎉 All SB3 PPO tests passed!")
    else:
        print("\n❌ Some SB3 PPO tests failed!")
        sys.exit(1) 