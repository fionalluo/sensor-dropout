#!/usr/bin/env python3
"""
Test script for PPO Distill policy loading with different student agent types.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines.ppo_distill.agent import PPODistillAgent
from baselines.ppo_distill.ppo_distill import train_ppo_distill
from baselines.ppo_distill.train import make_envs, load_config

def test_ppo_distill_student_types():
    """Test that PPO Distill can work with both PPO and PPO-RNN student agents."""
    
    # Test configuration
    config = load_config(['gymnasium_tigerdoorkey'])
    config.num_envs = 2
    config.num_steps = 4
    config.total_timesteps = 1000
    
    # Create environments
    envs = make_envs(config, config.num_envs)
    
    # Test expert policy directory
    expert_policy_dir = "./policies/ppo/tigerdoorkey"
    
    if not os.path.exists(expert_policy_dir):
        print(f"Warning: Expert policy directory not found: {expert_policy_dir}")
        print("Skipping test. Please run train_subset_policies.sh first.")
        return
    
    print("Testing PPO Distill with different student agent types...")
    
    # Test PPO-RNN student agent
    print("\n1. Testing PPO-RNN student agent...")
    try:
        agent_ppo_rnn = PPODistillAgent(
            envs, config, expert_policy_dir, 
            device='cpu', student_policy_type="ppo_rnn"
        )
        print("✓ PPO-RNN student agent created successfully")
        
        # Test getting action and value
        obs = {}
        for key in agent_ppo_rnn.mlp_keys + agent_ppo_rnn.cnn_keys:
            if key in envs.obs_space:
                obs[key] = torch.randn(1, *envs.obs_space[key].shape)
        
        lstm_state = agent_ppo_rnn.get_initial_lstm_state()
        done = torch.zeros(1)
        
        action, logprob, entropy, value, new_lstm_state, expert_actions = agent_ppo_rnn.get_action_and_value(
            obs, lstm_state, done
        )
        print("✓ PPO-RNN agent action and value computation successful")
        
    except Exception as e:
        print(f"✗ PPO-RNN student agent failed: {e}")
        return
    
    # Test PPO student agent
    print("\n2. Testing PPO student agent...")
    try:
        agent_ppo = PPODistillAgent(
            envs, config, expert_policy_dir, 
            device='cpu', student_policy_type="ppo"
        )
        print("✓ PPO student agent created successfully")
        
        # Test getting action and value
        obs = {}
        for key in agent_ppo.mlp_keys + agent_ppo.cnn_keys:
            if key in envs.obs_space:
                obs[key] = torch.randn(1, *envs.obs_space[key].shape)
        
        action, logprob, entropy, value, new_lstm_state, expert_actions = agent_ppo.get_action_and_value(
            obs, None, None
        )
        print("✓ PPO agent action and value computation successful")
        
    except Exception as e:
        print(f"✗ PPO student agent failed: {e}")
        return
    
    # Test training function
    print("\n3. Testing training function with PPO-RNN student...")
    try:
        # Use a small number of iterations for testing
        trained_agent = train_ppo_distill(
            envs=envs,
            config=config,
            seed=42,
            expert_policy_dir=expert_policy_dir,
            student_policy_type="ppo_rnn",
            num_iterations=2  # Very small for testing
        )
        print("✓ PPO Distill training with PPO-RNN student successful")
        
    except Exception as e:
        print(f"✗ PPO Distill training with PPO-RNN student failed: {e}")
    
    print("\n4. Testing training function with PPO student...")
    try:
        # Use a small number of iterations for testing
        trained_agent = train_ppo_distill(
            envs=envs,
            config=config,
            seed=42,
            expert_policy_dir=expert_policy_dir,
            student_policy_type="ppo",
            num_iterations=2  # Very small for testing
        )
        print("✓ PPO Distill training with PPO student successful")
        
    except Exception as e:
        print(f"✗ PPO Distill training with PPO student failed: {e}")
    
    # Clean up
    envs.close()
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_ppo_distill_student_types() 