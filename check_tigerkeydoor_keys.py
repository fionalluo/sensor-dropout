#!/usr/bin/env python3

import gymnasium as gym
import numpy as np

def check_tigerkeydoor_keys():
    """Check what observation keys are available in the TigerDoorKey environment."""
    
    # Create the environment
    env = gym.make('gymnasium_TigerDoorKey-v0')
    
    # Reset to get initial observation
    obs, info = env.reset()
    
    print("TigerDoorKey Environment Observation Keys:")
    print("=" * 50)
    
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {type(value)} = {value}")
    else:
        print(f"  Single observation: {type(obs)} = {obs}")
    
    print("\nObservation space:")
    print(f"  Type: {type(env.observation_space)}")
    if hasattr(env.observation_space, 'spaces'):
        print("  Keys:")
        for key, space in env.observation_space.spaces.items():
            print(f"    {key}: {space}")
    else:
        print(f"  Space: {env.observation_space}")
    
    # Take a step to see if keys change
    print("\nAfter one step:")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    env.close()

if __name__ == "__main__":
    check_tigerkeydoor_keys() 