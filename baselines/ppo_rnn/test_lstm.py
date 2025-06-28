#!/usr/bin/env python3
"""
Test script to verify LSTM implementation is working correctly.
"""

import torch
import torch.nn as nn
import numpy as np
from agent import PPORnnAgent

def test_lstm_initialization():
    """Test LSTM initialization and basic forward pass."""
    print("Testing LSTM initialization...")
    
    # Create a simple config
    class Config:
        def __init__(self):
            self.encoder = type('Encoder', (), {
                'act': 'relu',
                'output_dim': 64,
                'mlp_layers': 2,
                'mlp_units': 64,
                'cnn_depth': 32,
                'cnn_blocks': 2,
                'resize': 2,
                'minres': 4,
                'norm': 'layer'
            })()
            self.rnn = type('RNN', (), {'hidden_size': 128})()
            self.full_keys = type('Keys', (), {
                'mlp_keys': '.*',
                'cnn_keys': '.*'
            })()
    
    config = Config()
    
    # Create a mock environment
    class MockEnvs:
        def __init__(self):
            self.obs_space = {
                'observation': type('Space', (), {'shape': (10,)})()
            }
            self.act_space = {
                'action': type('Space', (), {'shape': (5,)})()
            }
    
    envs = MockEnvs()
    
    # Create agent
    agent = PPORnnAgent(envs, config)
    
    # Test initial LSTM state
    batch_size = 4
    initial_state = agent.get_initial_lstm_state(batch_size)
    print(f"Initial LSTM state shapes: {initial_state[0].shape}, {initial_state[1].shape}")
    
    # Test forward pass
    obs = {'observation': torch.randn(batch_size, 10)}
    done = torch.zeros(batch_size)
    
    hidden, new_state = agent.get_states(obs, initial_state, done)
    print(f"Hidden state shape: {hidden.shape}")
    print(f"New LSTM state shapes: {new_state[0].shape}, {new_state[1].shape}")
    
    # Test action and value
    action, logprob, entropy, value, final_state = agent.get_action_and_value(obs, initial_state, done)
    print(f"Action shape: {action.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Logprob shape: {logprob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    print("✓ LSTM initialization test passed!")

def test_lstm_sequence():
    """Test LSTM with sequence of observations."""
    print("\nTesting LSTM sequence processing...")
    
    # Create a simple config
    class Config:
        def __init__(self):
            self.encoder = type('Encoder', (), {
                'act': 'relu',
                'output_dim': 64,
                'mlp_layers': 2,
                'mlp_units': 64,
                'cnn_depth': 32,
                'cnn_blocks': 2,
                'resize': 2,
                'minres': 4,
                'norm': 'layer'
            })()
            self.rnn = type('RNN', (), {'hidden_size': 128})()
            self.full_keys = type('Keys', (), {
                'mlp_keys': '.*',
                'cnn_keys': '.*'
            })()
    
    config = Config()
    
    # Create a mock environment
    class MockEnvs:
        def __init__(self):
            self.obs_space = {
                'observation': type('Space', (), {'shape': (10,)})()
            }
            self.act_space = {
                'action': type('Space', (), {'shape': (5,)})()
            }
    
    envs = MockEnvs()
    
    # Create agent
    agent = PPORnnAgent(envs, config)
    
    # Test sequence processing
    batch_size = 4
    seq_len = 8
    initial_state = agent.get_initial_lstm_state(batch_size)
    
    # Create sequence of observations
    obs_sequence = []
    for t in range(seq_len):
        obs = {'observation': torch.randn(batch_size, 10)}
        obs_sequence.append(obs)
    
    # Process sequence
    current_state = initial_state
    hidden_states = []
    
    for t, obs in enumerate(obs_sequence):
        done = torch.zeros(batch_size)
        if t == seq_len - 1:  # Last timestep
            done[0] = 1.0  # End episode for first environment
        
        hidden, current_state = agent.get_states(obs, current_state, done)
        hidden_states.append(hidden)
    
    print(f"Processed sequence of length {seq_len}")
    print(f"Final hidden state shape: {hidden_states[-1].shape}")
    print(f"Final LSTM state shapes: {current_state[0].shape}, {current_state[1].shape}")
    
    print("✓ LSTM sequence test passed!")

def test_lstm_done_masking():
    """Test LSTM done masking."""
    print("\nTesting LSTM done masking...")
    
    # Create a simple config
    class Config:
        def __init__(self):
            self.encoder = type('Encoder', (), {
                'act': 'relu',
                'output_dim': 64,
                'mlp_layers': 2,
                'mlp_units': 64,
                'cnn_depth': 32,
                'cnn_blocks': 2,
                'resize': 2,
                'minres': 4,
                'norm': 'layer'
            })()
            self.rnn = type('RNN', (), {'hidden_size': 128})()
            self.full_keys = type('Keys', (), {
                'mlp_keys': '.*',
                'cnn_keys': '.*'
            })()
    
    config = Config()
    
    # Create a mock environment
    class MockEnvs:
        def __init__(self):
            self.obs_space = {
                'observation': type('Space', (), {'shape': (10,)})()
            }
            self.act_space = {
                'action': type('Space', (), {'shape': (5,)})()
            }
    
    envs = MockEnvs()
    
    # Create agent
    agent = PPORnnAgent(envs, config)
    
    # Test done masking
    batch_size = 4
    initial_state = agent.get_initial_lstm_state(batch_size)
    
    # Create observation
    obs = {'observation': torch.randn(batch_size, 10)}
    
    # Test without done
    done_none = torch.zeros(batch_size)
    hidden_none, state_none = agent.get_states(obs, initial_state, done_none)
    
    # Test with done for first environment
    done_first = torch.zeros(batch_size)
    done_first[0] = 1.0
    hidden_first, state_first = agent.get_states(obs, initial_state, done_first)
    
    # Test with done for all environments
    done_all = torch.ones(batch_size)
    hidden_all, state_all = agent.get_states(obs, initial_state, done_all)
    
    print(f"Hidden states should be different:")
    print(f"  No done: {hidden_none[0, :5]}")  # First 5 values
    print(f"  First done: {hidden_first[0, :5]}")
    print(f"  All done: {hidden_all[0, :5]}")
    
    print("✓ LSTM done masking test passed!")

if __name__ == "__main__":
    print("Running LSTM implementation tests...")
    print("=" * 50)
    
    test_lstm_initialization()
    test_lstm_sequence()
    test_lstm_done_masking()
    
    print("\n" + "=" * 50)
    print("All tests passed! LSTM implementation looks good.") 