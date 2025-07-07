#!/usr/bin/env python3
"""
Test script to verify PPO Distill SB3 implementation works correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines.ppo_distill_sb3.ppo_distill_sb3 import ExpertPolicyManager, ConfigurationScheduler

def test_expert_policy_manager():
    """Test loading expert policies."""
    
    # Set device to CPU to avoid CUDA issues
    device = 'cpu'
    
    # Path to expert policies (should exist after running train_subset_policies_sb3.sh)
    policy_dir = "policies/ppo_sb3/tigerdoorkey"
    
    print(f"Testing ExpertPolicyManager with: {policy_dir}")
    print(f"Using device: {device}")
    
    # Check if policy directory exists
    if not os.path.exists(policy_dir):
        print(f"❌ Policy directory not found: {policy_dir}")
        print("Please run train_subset_policies_sb3.sh first to generate expert policies")
        return False
    
    try:
        # Create expert policy manager
        expert_manager = ExpertPolicyManager(policy_dir, device=device)
        
        print(f"✓ Successfully loaded {len(expert_manager.expert_policies)} expert policies")
        print(f"  Expert policies: {list(expert_manager.expert_policies.keys())}")
        
        # Test getting expert actions
        dummy_obs = {
            'position': np.array([0.5, 0.5], dtype=np.float32),
            'has_key': np.array(0.0, dtype=np.float32),
            'neighbors': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            'door': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'doors_unlocked': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        
        # Test getting expert actions for each policy
        for subset_name in expert_manager.expert_policies.keys():
            try:
                expert_action = expert_manager.get_expert_action(subset_name, dummy_obs)
                print(f"  ✓ Got expert action for {subset_name}: {expert_action.shape}")
            except Exception as e:
                print(f"  ❌ Failed to get expert action for {subset_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create ExpertPolicyManager: {e}")
        return False

def test_configuration_scheduler():
    """Test configuration cycling."""
    
    # Create dummy eval_keys
    eval_keys = {
        'env1': {'mlp_keys': '.*', 'cnn_keys': '.*'},
        'env2': {'mlp_keys': 'position|has_key', 'cnn_keys': '^$'},
        'env3': {'mlp_keys': 'neighbors|door', 'cnn_keys': '^$'},
        'env4': {'mlp_keys': 'doors_unlocked', 'cnn_keys': '^$'},
    }
    
    print(f"Testing ConfigurationScheduler with {len(eval_keys)} configurations")
    
    try:
        # Create configuration scheduler
        scheduler = ConfigurationScheduler(eval_keys, cycle_mode='episode')
        
        print(f"✓ Successfully created ConfigurationScheduler")
        print(f"  Configurations: {list(eval_keys.keys())}")
        
        # Test getting current config
        current_config_name, current_eval_keys = scheduler.get_current_config()
        print(f"  Current config: {current_config_name}")
        print(f"  Current eval_keys: {current_eval_keys}")
        
        # Test cycling
        print(f"  Cycling configurations...")
        for i in range(5):
            scheduler.cycle_config(episode_done=True)
            current_config_name, _ = scheduler.get_current_config()
            print(f"    Step {i+1}: {current_config_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create ConfigurationScheduler: {e}")
        return False

def test_distillation_training():
    """Test that the distillation training function can be called."""
    
    # This is a basic test to ensure the function exists and can be imported
    try:
        from baselines.ppo_distill_sb3.ppo_distill_sb3 import train_ppo_distill_sb3
        print("✓ Successfully imported train_ppo_distill_sb3 function")
        return True
    except Exception as e:
        print(f"❌ Failed to import train_ppo_distill_sb3: {e}")
        return False

if __name__ == "__main__":
    print("Testing PPO Distill SB3 implementation...")
    
    # Test 1: Expert Policy Manager
    success1 = test_expert_policy_manager()
    
    # Test 2: Configuration Scheduler
    success2 = test_configuration_scheduler()
    
    # Test 3: Distillation Training Function
    success3 = test_distillation_training()
    
    if success1 and success2 and success3:
        print("\n🎉 All PPO Distill SB3 tests passed!")
        print("\nThe implementation is ready to use:")
        print("1. First run: ./train_subset_policies_sb3.sh")
        print("2. Then run: ./ppo_distill_sb3.sh")
    else:
        print("\n❌ Some PPO Distill SB3 tests failed!")
        sys.exit(1) 