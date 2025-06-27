#!/usr/bin/env python3
"""
Test script to verify that the training_env parameter works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import embodied
from baselines.ppo_rnn.agent import PPORnnAgent
from baselines.ppo_rnn.train import make_envs, load_config

def test_training_env():
    """Test that the training_env parameter correctly filters observation keys."""
    
    # Load config for Tiger Door Key environment
    config = load_config(['gymnasium_tigerkeydoor'])
    
    # Create environment
    envs = make_envs(config, num_envs=2)
    
    print("Testing training environment parameter...")
    print("=" * 50)
    
    # Test with full_keys (default)
    print("\n1. Testing with full_keys (default):")
    agent_full = PPORnnAgent(envs, config)
    print(f"   MLP keys: {agent_full.mlp_keys}")
    print(f"   CNN keys: {agent_full.cnn_keys}")
    
    # Test with env1
    print("\n2. Testing with env1:")
    agent_env1 = PPORnnAgent(envs, config, training_env='env1')
    print(f"   MLP keys: {agent_env1.mlp_keys}")
    print(f"   CNN keys: {agent_env1.cnn_keys}")
    
    # Test with env2
    print("\n3. Testing with env2:")
    agent_env2 = PPORnnAgent(envs, config, training_env='env2')
    print(f"   MLP keys: {agent_env2.mlp_keys}")
    print(f"   CNN keys: {agent_env2.cnn_keys}")
    
    # Test with env3
    print("\n4. Testing with env3:")
    agent_env3 = PPORnnAgent(envs, config, training_env='env3')
    print(f"   MLP keys: {agent_env3.mlp_keys}")
    print(f"   CNN keys: {agent_env3.cnn_keys}")
    
    # Test with env4
    print("\n5. Testing with env4:")
    agent_env4 = PPORnnAgent(envs, config, training_env='env4')
    print(f"   MLP keys: {agent_env4.mlp_keys}")
    print(f"   CNN keys: {agent_env4.cnn_keys}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    # Clean up
    envs.close()

if __name__ == "__main__":
    test_training_env() 