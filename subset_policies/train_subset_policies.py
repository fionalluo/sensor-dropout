#!/usr/bin/env python3
"""
Script to train PPO RNN policies for each subset of eval_keys.
This creates separate policies for each environment subset (env1, env2, etc.)
that can be easily loaded and deployed later.
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

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add embodied to path
import embodied
from embodied import wrappers

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
    parser = argparse.ArgumentParser(description='Train PPO RNN policies for each eval subset')
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()

def load_config(configs_names=None):
    """Load configuration from YAML file with support for named configs."""
    configs = ruamel.yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent.parent / 'baselines/ppo_rnn/config.yaml').read())
    
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

def train_subset_policy(config, subset_name, eval_keys, output_dir, device, debug=False):
    """Train a PPO RNN policy for a specific subset of observations."""
    
    print(f"\n{'='*60}")
    print(f"Training policy for {subset_name}")
    print(f"MLP keys pattern: {eval_keys['mlp_keys']}")
    print(f"CNN keys pattern: {eval_keys['cnn_keys']}")
    print(f"{'='*60}")
    
    # Create subset-specific config
    subset_config = SimpleNamespace()
    for attr in dir(config):
        if not attr.startswith('_'):
            setattr(subset_config, attr, getattr(config, attr))
    
    # Update keys for this subset - keep full_keys as original, only change keys
    subset_config.keys = SimpleNamespace(**eval_keys)
    # Keep the original full_keys for encoder building
    # subset_config.full_keys remains unchanged from the original config
    
    # Update exp_name to include subset name for distinct wandb logging
    if hasattr(subset_config, 'exp_name'):
        subset_config.exp_name = f"{subset_config.exp_name}_{subset_name}"
    else:
        subset_config.exp_name = f"subset_{subset_name}"
    
    # Create environment normally (no wrapper needed)
    envs = make_envs(config, num_envs=config.num_envs)
    
    # Calculate number of iterations
    num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)
    
    # Train the policy
    print(f"Starting training for {subset_name}...")
    trained_agent = train_ppo_rnn(envs, subset_config, config.seed, num_iterations=num_iterations)
    
    # Save the policy
    policy_dir = os.path.join(output_dir, subset_name)
    os.makedirs(policy_dir, exist_ok=True)
    
    # Save the trained agent
    policy_path = os.path.join(policy_dir, 'policy.pt')
    
    # Check if policy already exists
    if os.path.exists(policy_path):
        print(f"Overwriting existing policy at {policy_path}")
    
    torch.save({
        'agent_state_dict': trained_agent.state_dict(),
        'config': subset_config,
        'eval_keys': eval_keys,
        'subset_name': subset_name
    }, policy_path)
    
    print(f"Policy saved to {policy_path}")
    
    return policy_path

def main():
    """Main entry point for training subset policies."""
    args = parse_args()
    
    # Load config using only the configs argument
    config = load_config(args.configs)
    
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
    
    # Create output directory
    task_name = config.task.replace('gymnasium_', '').replace('-v0', '').lower()
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training policies for task: {config.task}")
    print(f"Output directory: {output_dir}")
    
    # Get number of eval configs
    if not hasattr(config, 'eval') or not hasattr(config.eval, 'num_eval_configs'):
        print("Error: No eval.num_eval_configs found in config")
        return
    
    num_eval_configs = config.eval.num_eval_configs
    print(f"Number of eval configs: {num_eval_configs}")
    
    # Train policies for each subset
    trained_policies = {}
    
    for subset_idx in range(1, num_eval_configs + 1):
        subset_name = f"env{subset_idx}"
        
        # Get eval keys for this subset
        if hasattr(config, 'eval_keys') and hasattr(config.eval_keys, subset_name):
            eval_keys = get_eval_keys(config, subset_name)
        else:
            print(f"Warning: No eval_keys.{subset_name} found, using default patterns")
            eval_keys = {'mlp_keys': '.*', 'cnn_keys': '.*'}
        
        # Train policy for this subset
        policy_path = train_subset_policy(
            config, subset_name, eval_keys, output_dir, device, debug=args.debug
        )
        trained_policies[subset_name] = policy_path
    
    # Save metadata about all trained policies
    metadata = {
        'task': config.task,
        'num_eval_configs': num_eval_configs,
        'policies': trained_policies,
        'config': config
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