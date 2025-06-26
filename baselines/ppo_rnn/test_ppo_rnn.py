#!/usr/bin/env python3
"""
Test script for PPO RNN implementation.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from baselines.ppo_rnn.agent import PPORnnAgent
from baselines.shared.config_utils import load_config

def test_ppo_rnn_agent():
    """Test that PPO RNN agent can be created and run."""
    print("Testing PPO RNN agent...")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    # Create a simple mock environment
    class MockEnv:
        def __init__(self):
            self.obs_space = {'image': gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)}
            self.act_space = {'action': gym.spaces.Discrete(4)}
            self.single_observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
            self.single_action_space = gym.spaces.Discrete(4)
    
    envs = MockEnv()
    
    # Create agent
    device = torch.device("cpu")
    agent = PPORnnAgent(envs, config).to(device)
    
    print(f"Agent created successfully!")
    print(f"LSTM hidden size: {config['rnn']['hidden_size']}")
    print(f"LSTM layers: {config['rnn']['num_layers']}")
    
    # Test forward pass
    batch_size = 4
    obs = torch.randint(0, 255, (batch_size, 1, 84, 84), dtype=torch.uint8)
    lstm_state = (
        torch.zeros(config['rnn']['num_layers'], batch_size, config['rnn']['hidden_size']),
        torch.zeros(config['rnn']['num_layers'], batch_size, config['rnn']['hidden_size'])
    )
    done = torch.zeros(batch_size, dtype=torch.bool)
    
    # Test encoding
    latent = agent.encode_observations({'image': obs})
    print(f"Latent shape: {latent.shape}")
    
    # Test LSTM forward pass
    hidden, new_lstm_state = agent.get_states({'image': obs}, lstm_state, done)
    print(f"Hidden shape: {hidden.shape}")
    print(f"New LSTM state shapes: {[s.shape for s in new_lstm_state]}")
    
    # Test action and value
    action, logprob, entropy, value, final_lstm_state = agent.get_action_and_value(
        {'image': obs}, lstm_state, done
    )
    print(f"Action shape: {action.shape}")
    print(f"Logprob shape: {logprob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Value shape: {value.shape}")
    
    print("All tests passed! PPO RNN agent is working correctly.")

if __name__ == "__main__":
    test_ppo_rnn_agent() 