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
import glob
import ruamel.yaml
import embodied
from embodied import wrappers
from baselines.shared.config_utils import dict_to_namespace

# Add SimpleNamespace to safe globals for torch.load
torch.serialization.add_safe_globals([SimpleNamespace])


def load_config_for_policy(policy_type: str, task_name: str, subset_name: str = None):
    """
    Load configuration for a specific policy type and task.
    
    Args:
        policy_type: 'ppo' or 'ppo_rnn'
        task_name: Task name (e.g., 'tigerdoorkey')
        subset_name: Subset name (e.g., 'env1') - if None, uses full config
        
    Returns:
        Configuration object
    """
    # Load the appropriate config file - fix the path to avoid double baselines
    if policy_type == 'ppo':
        config_path = Path(__file__).parent.parent / 'ppo/config.yaml'
    else:  # ppo_rnn
        config_path = Path(__file__).parent.parent / 'ppo_rnn/config.yaml'
    
    configs = ruamel.yaml.YAML(typ='safe').load(config_path.read_text())
    
    # Start with defaults
    config_dict = embodied.Config(configs['defaults'])
    
    # Apply task-specific config
    if task_name in configs:
        config_dict = config_dict.update(configs[task_name])
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    
    # If subset_name is provided, update keys for that subset
    if subset_name and hasattr(config, 'eval_keys') and hasattr(config.eval_keys, subset_name):
        env_keys = getattr(config.eval_keys, subset_name)
        config.keys = SimpleNamespace(
            mlp_keys=env_keys.mlp_keys,
            cnn_keys=env_keys.cnn_keys
        )
    
    return config


def make_envs(config, num_envs):
    """Create vectorized environments - same as subset_policies."""
    from functools import partial as bind
    
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(num_envs):
        ctor = lambda: make_env(config)
        if hasattr(config, 'envs') and hasattr(config.envs, 'parallel') and config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if hasattr(config, 'envs') and hasattr(config.envs, 'restart') and config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(hasattr(config, 'envs') and hasattr(config.envs, 'parallel') and config.envs.parallel != 'none'))


def make_env(config, **overrides):
    """Create a single environment - same as subset_policies."""
    import importlib
    from functools import partial as bind
    
    suite, task = config.task.split('_', 1)
    if "TrailEnv" in task or "GridBlindPick" or "LavaTrail" in task:
        import trailenv

    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'robopianist': 'embodied.envs.robopianist:RoboPianist'
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = getattr(config.env, suite, {})
    kwargs.update(overrides)
    if suite == 'robopianist':
        render_image = False
        if 'Pixel' in task:
            task = task.replace('Pixel', '')
        render_image = True
        kwargs.update({'render_image': render_image})

    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    """Wrap environment with standard wrappers - same as subset_policies."""
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


def make_envs_for_distillation(config, num_envs):
    """
    Create vectorized environments for PPO Distill with ALL observation keys.
    This ensures that all expert policies get access to their required observation keys.
    Uses regex patterns to match actual environment keys.
    
    Args:
        config: Configuration object
        num_envs: Number of environments to create
        
    Returns:
        Vectorized environment with all observation keys
    """
    import copy
    from types import SimpleNamespace
    import re

    # 1. Instantiate a temporary environment to get available keys
    temp_env = make_env(config)
    available_keys = list(temp_env.obs_space.keys())
    temp_env.close()

    # 2. Determine how many eval configs there are
    num_eval_configs = getattr(getattr(config, 'eval', SimpleNamespace()), 'num_eval_configs', 0)
    all_mlp_keys = set()
    all_cnn_keys = set()
    if hasattr(config, 'eval_keys') and num_eval_configs > 0:
        for i in range(1, num_eval_configs + 1):
            env_name = f'env{i}'
            if hasattr(config.eval_keys, env_name):
                env_config = getattr(config.eval_keys, env_name)
                # MLP keys
                if hasattr(env_config, 'mlp_keys'):
                    pattern = env_config.mlp_keys
                    for key in available_keys:
                        if re.search(pattern, key):
                            all_mlp_keys.add(key)
                # CNN keys
                if hasattr(env_config, 'cnn_keys'):
                    pattern = env_config.cnn_keys
                    for key in available_keys:
                        if re.search(pattern, key):
                            all_cnn_keys.add(key)

    # 3. Create a modified config that includes all keys
    modified_config = copy.deepcopy(config)
    if not hasattr(modified_config, 'full_keys'):
        modified_config.full_keys = SimpleNamespace()

    # 4. Create regex patterns that match all the keys we need
    all_mlp_pattern = '|'.join([f'\\b{key}\\b' for key in all_mlp_keys]) if all_mlp_keys else '^$'
    all_cnn_pattern = '|'.join([f'\\b{key}\\b' for key in all_cnn_keys]) if all_cnn_keys else '^$'
    modified_config.full_keys.mlp_keys = all_mlp_pattern
    modified_config.full_keys.cnn_keys = all_cnn_pattern

    # 5. Create environments with the modified config
    return make_envs(modified_config, num_envs)


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


def convert_obs_to_tensor(obs: Dict[str, Any], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Convert observation dictionary to tensor dictionary.
    
    Args:
        obs: Observation dictionary
        device: Device to place tensors on
        
    Returns:
        Dictionary of tensors
    """
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_tensor[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, list):
            obs_tensor[key] = torch.tensor(value, dtype=torch.float32).to(device)
        elif isinstance(value, torch.Tensor):
            obs_tensor[key] = value.to(device)
        else:
            obs_tensor[key] = torch.tensor([value], dtype=torch.float32).to(device)
    
    return obs_tensor


def find_policy_files(policy_dir: str, pattern: str = "policy*.pt") -> Dict[str, str]:
    """
    Find all policy files in a directory, selecting the most recent one per subset.
    
    Args:
        policy_dir: Directory to search
        pattern: Pattern to match policy files
        
    Returns:
        Dict mapping policy names to file paths (most recent file per subset)
    """
    policies = {}
    
    if not os.path.exists(policy_dir):
        return policies
    
    for item in os.listdir(policy_dir):
        item_path = os.path.join(policy_dir, item)
        if os.path.isdir(item_path):
            # Look for policy files in subdirectories
            policy_files = glob.glob(os.path.join(item_path, pattern))
            if policy_files:
                # Sort by modification time and select the most recent
                policy_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                policies[item] = policy_files[0]  # Most recent file
    
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
        try:
            # Try to load with safe_load first
            with open(metadata_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            # If that fails due to SimpleNamespace objects, try a custom approach
            print(f"Warning: Could not load metadata with safe_load: {e}")
            print("Attempting to load with custom parser...")
            
            try:
                # Read the file and extract only the simple fields we need
                with open(metadata_path, 'r') as f:
                    content = f.read()
                
                # Extract basic metadata fields using regex
                metadata = {}
                
                # Extract task
                task_match = re.search(r'task:\s*(.+)', content)
                if task_match:
                    metadata['task'] = task_match.group(1).strip()
                
                # Extract policy_type
                policy_type_match = re.search(r'policy_type:\s*(.+)', content)
                if policy_type_match:
                    metadata['policy_type'] = policy_type_match.group(1).strip()
                
                # Extract num_eval_configs
                num_eval_match = re.search(r'num_eval_configs:\s*(\d+)', content)
                if num_eval_match:
                    metadata['num_eval_configs'] = int(num_eval_match.group(1))
                
                # Extract policies dictionary
                policies_match = re.search(r'policies:\s*\n((?:\s+\w+:\s+[^\n]+\n?)+)', content)
                if policies_match:
                    policies_text = policies_match.group(1)
                    policies = {}
                    for line in policies_text.strip().split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            policies[key.strip()] = value.strip()
                    metadata['policies'] = policies
                
                return metadata
                
            except Exception as e2:
                print(f"Failed to load metadata with custom parser: {e2}")
                return None
    
    return None


def load_policy_like_subset_policies(policy_dir: str, policy_type: str, device: str = 'cpu'):
    """
    Load policies using the exact same approach as subset_policies.
    This is the working approach that matches how policies were trained.
    
    Args:
        policy_dir: Directory containing trained policies
        policy_type: 'ppo' or 'ppo_rnn'
        device: Device to load policies on
        
    Returns:
        dict: Dictionary mapping subset names to (agent, config, eval_keys) tuples
    """
    # Import here to avoid circular imports
    from subset_policies.load_subset_policy import SubsetPolicyLoader
    
    # Create policy loader using the working approach
    loader = SubsetPolicyLoader(policy_dir, device=device)
    
    # Load all policies
    loaded_policies = {}
    for subset_name in loader.policies.keys():
        agent, config, eval_keys = loader.load_policy(subset_name)
        loaded_policies[subset_name] = (agent, config, eval_keys)
    
    return loaded_policies
