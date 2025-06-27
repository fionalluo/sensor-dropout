#!/usr/bin/env python3
"""
Test script to demonstrate prefix-based substitution for unprivileged keys.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from baselines.shared.eval_utils import find_unprivileged_key, substitute_unprivileged_for_agent

def test_prefix_substitution():
    """Test the prefix-based substitution system."""
    
    # Simulate available observations from eval_keys.env2
    available_obs = {
        'neighbors_unprivileged_key': [1, 0, 0, 0],  # Key-specific neighbors
        'door_unprivileged': [0, 1, 0, 0],  # Generic unprivileged door
        'doors_unlocked': [1, 1, 0, 0],  # Privileged doors_unlocked (available)
        'position': [0, 0],  # Position (always available)
        'has_key': [1],  # Has key (always available)
    }
    
    # Agent expects these privileged keys
    agent_keys = ['neighbors', 'door', 'doors_unlocked', 'position', 'has_key']
    
    print("=== Prefix-Based Substitution Test ===")
    print(f"Available observations: {list(available_obs.keys())}")
    print(f"Agent expects: {agent_keys}")
    print()
    
    # Test individual key substitution
    print("Testing individual key substitution:")
    for key in agent_keys:
        unprivileged_key = find_unprivileged_key(key, available_obs)
        if unprivileged_key:
            print(f"  {key} -> {unprivileged_key}")
        elif key in available_obs:
            print(f"  {key} -> {key} (direct)")
        else:
            print(f"  {key} -> None (not found)")
    print()
    
    # Test full substitution
    print("Testing full substitution:")
    final_obs = substitute_unprivileged_for_agent(agent_keys, available_obs, {})
    
    for key in agent_keys:
        if final_obs[key] is not None:
            unprivileged_key = find_unprivileged_key(key, available_obs)
            if unprivileged_key and key not in available_obs:
                print(f"  {key}: {final_obs[key]} (substituted from {unprivileged_key})")
            else:
                print(f"  {key}: {final_obs[key]} (direct)")
        else:
            print(f"  {key}: None (zeroed out)")
    print()
    
    # Test with different unprivileged key patterns
    print("Testing different unprivileged key patterns:")
    test_cases = [
        {
            'name': 'Basic suffix',
            'available': {'neighbors_unprivileged': [1, 0, 0, 0]}
        },
        {
            'name': 'Custom suffix',
            'available': {'neighbors_unprivileged_key': [1, 0, 0, 0]}
        },
        {
            'name': 'Button suffix',
            'available': {'neighbors_unprivileged_button': [1, 0, 0, 0]}
        },
        {
            'name': 'Multiple options (should pick first)',
            'available': {
                'neighbors_unprivileged_key': [1, 0, 0, 0],
                'neighbors_unprivileged_button': [0, 1, 0, 0]
            }
        }
    ]
    
    for case in test_cases:
        print(f"  {case['name']}:")
        unprivileged_key = find_unprivileged_key('neighbors', case['available'])
        if unprivileged_key:
            print(f"    neighbors -> {unprivileged_key}")
        else:
            print(f"    neighbors -> None")
    print()
    
    print("Prefix-based substitution rules:")
    print("1. For privileged key 'key', look for 'key_unprivileged*'")
    print("2. Any suffix after 'key_unprivileged' is allowed")
    print("3. If multiple matches, picks the first one found")
    print("4. If no match, returns None (will be zeroed out)")

if __name__ == "__main__":
    test_prefix_substitution() 