"""
Shared policy loading utilities for all baselines.
This module provides functions to load trained policies from .pt files.
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from types import SimpleNamespace
import json
import re
from baselines.ppo_rnn.agent import PPORnnAgent

import embodied
from embodied import wrappers

# Add SimpleNamespace to safe globals for torch.load
torch.serialization.add_safe_globals([SimpleNamespace])


def load_policy_checkpoint(policy_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load a policy checkpoint from a .pt file.
    
    Args:
        policy_path: Path to the .pt file
        device: Device to load the checkpoint on
        
    Returns:
        Dict containing the checkpoint contents
    """
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    try:
        # Try loading with weights_only=True first (safer)
        checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
    except Exception as e:
        # If that fails, try with weights_only=False (less safe but more compatible)
        print(f"Warning: weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    
    return checkpoint


def create_env_from_config(config: SimpleNamespace) -> embodied.Env:
    """
    Create an environment from a config object.
    
    Args:
        config: Configuration object with task and env settings
        
    Returns:
        embodied.Env: Created environment
    """
    suite, task = config.task.split('_', 1)
    
    ctor = {
        'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
        'gym': 'embodied.envs.from_gym:FromGym',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'robopianist': 'embodied.envs.robopianist:RoboPianist'
    }.get(suite, 'embodied.envs.from_gymnasium:FromGymnasium')
    
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = __import__(module, fromlist=[cls])
        ctor = getattr(module, cls)
    
    kwargs = getattr(config.env, suite, {})
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env: embodied.Env, config: SimpleNamespace) -> embodied.Env:
    """
    Wrap environment with standard wrappers.
    
    Args:
        env: Environment to wrap
        config: Configuration object
        
    Returns:
        embodied.Env: Wrapped environment
    """
    args = getattr(config, 'wrapper', {})
    
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif hasattr(args, 'discretize') and args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)

    env = wrappers.ExpandScalars(env)

    if hasattr(args, 'length') and args.length:
        env = wrappers.TimeLimit(env, args.length, getattr(args, 'reset', True))
    if hasattr(args, 'checks') and args.checks:
        env = wrappers.CheckSpaces(env)

    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)

    return env


def load_agent_from_checkpoint(checkpoint: Dict[str, Any], agent_class, device: str = 'cpu'):
    """
    Load an agent from a checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        agent_class: Agent class to instantiate
        device: Device to load the agent on
        
    Returns:
        Loaded agent instance
    """
    agent_state_dict = checkpoint['agent_state_dict']
    config = checkpoint['config']
    
    # Create environment to get observation space
    env = create_env_from_config(config)
    
    # Create agent
    agent = agent_class(
        env.obs_space,
        env.act_space,
        config,
        device=device
    )
    
    # Load state dict
    agent.load_state_dict(agent_state_dict)
    agent.eval()
    
    return agent


def load_policy_with_metadata(policy_path: str, agent_class, device: str = 'cpu') -> Tuple[Any, SimpleNamespace, Dict[str, Any]]:
    """
    Load a policy with its metadata from a .pt file.
    
    Args:
        policy_path: Path to the .pt file
        agent_class: Agent class to instantiate
        device: Device to load the policy on
        
    Returns:
        Tuple of (agent, config, metadata)
    """
    checkpoint = load_policy_checkpoint(policy_path, device)
    
    # Extract components
    agent_state_dict = checkpoint['agent_state_dict']
    config = checkpoint['config']
    
    # Extract additional metadata if available
    metadata = {}
    for key in ['eval_keys', 'subset_name', 'task', 'seed']:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    
    # Create and load agent
    agent = load_agent_from_checkpoint(checkpoint, agent_class, device)
    
    return agent, config, metadata


def find_policy_files(policy_dir: str, pattern: str = "policy*.pt") -> Dict[str, str]:
    """
    Find all policy files in a directory.
    
    Args:
        policy_dir: Directory to search
        pattern: Pattern to match policy files
        
    Returns:
        Dict mapping policy names to file paths
    """
    policies = {}
    
    if not os.path.exists(policy_dir):
        return policies
    
    for item in os.listdir(policy_dir):
        item_path = os.path.join(policy_dir, item)
        if os.path.isdir(item_path):
            # Look for policy files in subdirectories
            for file in os.listdir(item_path):
                if file.endswith('.pt') and file.startswith('policy'):
                    policy_path = os.path.join(item_path, file)
                    policies[item] = policy_path
                    break
    
    return policies


def load_metadata_from_dir(policy_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a policy directory.
    
    Args:
        policy_dir: Directory containing policies
        
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = os.path.join(policy_dir, 'metadata.yaml')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def convert_obs_to_tensor(obs: Dict[str, Any], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Convert observation dictionary to tensor format.
    
    Args:
        obs: Observation dictionary
        device: Device to place tensors on
        
    Returns:
        Dict with tensor observations
    """
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_tensor[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, torch.Tensor):
            obs_tensor[key] = value.to(device)
        else:
            obs_tensor[key] = torch.tensor([value], dtype=torch.float32).to(device)
    
    return obs_tensor


def load_expert_policy(policy_path: str, envs, config, device: str = 'cpu') -> PPORnnAgent:
    """
    Load an expert policy from a checkpoint file.
    
    Args:
        policy_path: Path to the policy checkpoint
        envs: Environment instance
        config: Configuration object
        device: Device to load the policy on
        
    Returns:
        Loaded PPORnnAgent instance
    """
    try:
        # Load checkpoint
        checkpoint = load_policy_checkpoint(policy_path, device)
        
        # Create agent instance (without device parameter)
        agent = PPORnnAgent(envs, config)
        
        # Load state dict
        agent.load_state_dict(checkpoint['model'])
        
        # Move to device
        agent.to(device)
        
        # Set to evaluation mode
        agent.eval()
        
        return agent
        
    except Exception as e:
        print(f"Failed to load expert policy {policy_path}: {e}")
        return None 