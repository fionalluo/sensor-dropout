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

# Add SimpleNamespace to safe globals for torch.load
torch.serialization.add_safe_globals([SimpleNamespace])


def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


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


def initialize_agent_like_subset_policies(policy_type: str, task_name: str, subset_name: str, device: str = 'cpu', checkpoint_path: str = None):
    """
    Initialize an agent exactly like subset_policies does.
    
    Args:
        policy_type: 'ppo' or 'ppo_rnn'
        task_name: Task name (e.g., 'tigerdoorkey')
        subset_name: Subset name (e.g., 'env1')
        device: Device to load the agent on
        checkpoint_path: Optional path to checkpoint to load config from
        
    Returns:
        Initialized agent
    """
    print(f"\n{'='*60}")
    print(f"Initializing {policy_type} agent for {subset_name}")
    print(f"{'='*60}")
    
    # If checkpoint path is provided, load the exact config that was used during training
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading config from checkpoint: {checkpoint_path}")
        checkpoint = load_policy_checkpoint(checkpoint_path, device)
        if 'config' in checkpoint:
            print("Using config from checkpoint")
            config = checkpoint['config']
            # Override task_name from checkpoint if available
            if hasattr(config, 'task'):
                task_name = config.task.split('_', 1)[-1] if '_' in config.task else config.task
        else:
            print("No config in checkpoint, loading from file")
            config = load_config_for_policy(policy_type, task_name, subset_name)
    else:
        print("Loading config from file")
        config = load_config_for_policy(policy_type, task_name, subset_name)
    
    print(f"Base config task: {getattr(config, 'task', 'Unknown')}")
    print(f"Base config keys: {getattr(config, 'keys', 'Not set')}")
    print(f"Base config full_keys: {getattr(config, 'full_keys', 'Not set')}")
    
    # Get eval keys for this subset - this is crucial for model architecture
    if hasattr(config, 'eval_keys') and hasattr(config.eval_keys, subset_name):
        env_keys = getattr(config.eval_keys, subset_name)
        eval_keys = {
            'mlp_keys': env_keys.mlp_keys,
            'cnn_keys': env_keys.cnn_keys
        }
        print(f"Found eval_keys for {subset_name}: {eval_keys}")
    else:
        print(f"Warning: No eval_keys.{subset_name} found, using default patterns")
        eval_keys = {'mlp_keys': '.*', 'cnn_keys': '.*'}
    
    # Create subset-specific config exactly like subset_policies
    subset_config = SimpleNamespace()
    for attr in dir(config):
        if not attr.startswith('_'):
            setattr(subset_config, attr, getattr(config, attr))
    
    # Update keys for this subset - keep full_keys as original, only change keys
    # This is crucial because the model architecture depends on which keys are available
    subset_config.keys = SimpleNamespace(**eval_keys)
    # Keep the original full_keys for encoder building - this is important!
    if hasattr(config, 'full_keys'):
        subset_config.full_keys = config.full_keys
    
    # Update exp_name to include subset name for distinct logging
    if hasattr(subset_config, 'exp_name'):
        subset_config.exp_name = f"{subset_config.exp_name}_{subset_name}"
    else:
        subset_config.exp_name = f"subset_{subset_name}"
    
    print(f"\nFinal subset config:")
    print(f"  Task: {getattr(subset_config, 'task', 'Unknown')}")
    print(f"  MLP keys: {eval_keys['mlp_keys']}")
    print(f"  CNN keys: {eval_keys['cnn_keys']}")
    print(f"  Full keys: {getattr(subset_config, 'full_keys', 'Not set')}")
    print(f"  Exp name: {getattr(subset_config, 'exp_name', 'Not set')}")
    
    # Create environment exactly like subset_policies
    print(f"\nCreating environment...")
    envs = make_envs(config, num_envs=config.num_envs)
    print(f"Environment created successfully")
    
    # Create agent based on type
    print(f"\nCreating {policy_type} agent...")
    if policy_type == 'ppo':
        from baselines.ppo.agent import PPOAgent
        agent = PPOAgent(envs, subset_config)
    else:  # ppo_rnn
        from baselines.ppo_rnn.ppo_rnn import PPORnnAgent
        agent = PPORnnAgent(envs, subset_config)
    
    agent.to(device)
    print(f"Agent created successfully on {device}")
    
    # Print model architecture info for debugging
    print(f"\nModel architecture info:")
    if hasattr(agent, 'mlp_encoder') and hasattr(agent.mlp_encoder, '0'):
        print(f"  MLP encoder input size: {agent.mlp_encoder[0].in_features}")
    if hasattr(agent, 'latent_projector') and hasattr(agent.latent_projector, '0'):
        print(f"  Latent projector input size: {agent.latent_projector[0].in_features}")
    
    print(f"{'='*60}\n")
    
    return agent, subset_config, eval_keys


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


def load_agent_from_checkpoint(checkpoint: Dict[str, Any], agent_class, device: str = 'cpu'):
    """
    Load an agent from a checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        agent_class: Agent class to instantiate
        device: Device to load the agent on
        
    Returns:
        Loaded agent
    """
    agent_state_dict = checkpoint['agent_state_dict']
    config = checkpoint['config']
    
    # Create environment for agent initialization
    env = create_env_for_agent(config)
    
    # Create agent instance
    agent = agent_class(env, config)
    agent.load_state_dict(agent_state_dict)
    agent.to(device)
    
    return agent


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