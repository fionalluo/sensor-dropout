#!/usr/bin/env python3
"""
Training script for PPO and PPO RNN subset policies.

This script trains separate PPO or PPO RNN policies for different observation subsets
(e.g., env1, env2, env3, env4) and saves them to the policies directory with
the structure: policies/{policy_type}/{task_name}/{env_name}/policy_{timestamp}.pt

Supports both PPO (non-RNN) and PPO-RNN policies.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
import ruamel.yaml
import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from functools import partial as bind
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add embodied to path
import embodied
from embodied import wrappers

# Import both PPO and PPO-RNN training functions
from baselines.ppo.ppo import train_ppo
from baselines.ppo_rnn.ppo_rnn import train_ppo_rnn
from baselines.shared.eval_utils import get_eval_keys

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO or PPO RNN policies for each eval subset')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--configs', type=str, nargs='+', default=[], 
                       help='Which named configs to apply')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--cuda', action='store_true', default=None,
                       help='Use CUDA (overrides config)')
    parser.add_argument('--track', action='store_true', default=None,
                       help='Track with wandb (overrides config)')
    parser.add_argument('--output_dir', type=str, default='policies',
                       help='Output directory for policies')
    parser.add_argument('--policy_type', type=str, choices=['ppo', 'ppo_rnn'], default='ppo_rnn',
                       help='Type of policy to train (ppo or ppo_rnn)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()

def load_config(configs_names=None, policy_type='ppo_rnn'):
    """Load configuration from YAML file with support for named configs."""
    # Choose config file based on policy type
    if policy_type == 'ppo':
        config_path = embodied.Path(__file__).parent.parent / 'baselines/ppo/config.yaml'
    else:  # ppo_rnn
        config_path = embodied.Path(__file__).parent.parent / 'baselines/ppo_rnn/config.yaml'
    
    configs = ruamel.yaml.YAML(typ='safe').load(config_path.read())
    
    # Start with defaults
    config_dict = embodied.Config(configs['defaults'])

    # Apply named configs
    if configs_names:
        for name in configs_names:
            if name in configs:
                config_dict = config_dict.update(configs[name])
            else:
                print(f"Warning: Config '{name}' not found in config file")
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    
    return config

def make_envs(config, num_envs):
    """Create vectorized environments."""
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
    """Create a single environment."""
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
    """Wrap environment with standard wrappers."""
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

def train_subset_policy(config, subset_name, eval_keys, output_dir, device, policy_type='ppo_rnn', debug=False):
    """Train a PPO or PPO RNN policy for a specific subset of observations."""
    
    print(f"\n{'='*60}")
    print(f"Training {policy_type.upper()} policy for {subset_name}")
    print(f"MLP keys pattern: {eval_keys['mlp_keys']}")
    print(f"CNN keys pattern: {eval_keys['cnn_keys']}")
    print(f"{'='*60}")
    
    # Create subset-specific config
    subset_config = SimpleNamespace()
    for attr in dir(config):
        if not attr.startswith('_'):
            setattr(subset_config, attr, getattr(config, attr))
    
    # Update keys for this subset - set full_keys to the subset keys so the agent uses the correct keys
    subset_config.full_keys = SimpleNamespace(**eval_keys)
    # Also set keys for compatibility
    subset_config.keys = SimpleNamespace(**eval_keys)
    
    # Update exp_name to include subset name for distinct wandb logging
    if hasattr(subset_config, 'exp_name'):
        subset_config.exp_name = f"{subset_config.exp_name}_{subset_name}"
    else:
        subset_config.exp_name = f"subset_{subset_name}"
    
    # Create environment normally (no wrapper needed)
    envs = make_envs(config, num_envs=config.num_envs)
    
    # Calculate number of iterations
    num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)
    
    # Train the policy based on type
    print(f"Starting training for {subset_name}...")
    if policy_type == 'ppo':
        trained_agent = train_ppo(envs, subset_config, config.seed, num_iterations=num_iterations, skip_subset_eval=True)
    else:  # ppo_rnn
        trained_agent = train_ppo_rnn(envs, subset_config, config.seed, num_iterations=num_iterations, skip_subset_eval=True)
    
    # Save the policy with timestamp
    policy_dir = os.path.join(output_dir, subset_name)
    os.makedirs(policy_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_filename = f"policy_{timestamp}.pt"
    policy_path = os.path.join(policy_dir, policy_filename)
    
    # Check if policy already exists
    if os.path.exists(policy_path):
        print(f"Overwriting existing policy at {policy_path}")
    
    torch.save({
        'agent_state_dict': trained_agent.state_dict(),
        'config': subset_config,
        'eval_keys': eval_keys,
        'subset_name': subset_name,
        'policy_type': policy_type  # Add policy type to metadata
    }, policy_path)
    
    print(f"{policy_type.upper()} policy saved to {policy_path}")
    
    return policy_path

def main():
    """Main entry point for training subset policies."""
    args = parse_args()
    
    # Load config using only the configs argument and policy type
    config = load_config(args.configs, args.policy_type)
    
    # Override config with command line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.cuda is not None:
        config.cuda = args.cuda
    if args.track is not None:
        config.track = args.track
    
    # Set random seed
    set_seed(config.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Use output directory directly (shell script already provides the correct name)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training {args.policy_type.upper()} policies for task: {config.task}")
    print(f"Output directory: {output_dir}")
    
    # Get number of eval configs
    if not hasattr(config, 'eval') or not hasattr(config.eval, 'num_eval_configs'):
        print("Error: No eval.num_eval_configs found in config")
        return
    
    num_eval_configs = config.eval.num_eval_configs
    print(f"Number of eval configs: {num_eval_configs}")
    
    # Train policies for each subset in reverse order
    trained_policies = {}
    
    for subset_idx in range(num_eval_configs, 0, -1):  # Start from highest, go down to 1
        subset_name = f"env{subset_idx}"
        
        # Get eval keys for this subset
        if hasattr(config, 'eval_keys') and hasattr(config.eval_keys, subset_name):
            env_keys = getattr(config.eval_keys, subset_name)
            eval_keys = {
                'mlp_keys': env_keys.mlp_keys,
                'cnn_keys': env_keys.cnn_keys
            }
        else:
            print(f"Warning: No eval_keys.{subset_name} found, using default patterns")
            eval_keys = {'mlp_keys': '.*', 'cnn_keys': '.*'}
        
        # Train policy for this subset
        policy_path = train_subset_policy(
            config, subset_name, eval_keys, output_dir, device, 
            policy_type=args.policy_type, debug=args.debug
        )
        trained_policies[subset_name] = policy_path
    
    # Save metadata about all trained policies
    metadata = {
        'task': config.task,
        'num_eval_configs': num_eval_configs,
        'policies': trained_policies,
        'config': config,
        'policy_type': args.policy_type  # Add policy type to metadata
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"All policies saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nTrained policies:")
    for subset_name, policy_path in trained_policies.items():
        print(f"  {subset_name}: {policy_path}")

if __name__ == "__main__":
    main() 