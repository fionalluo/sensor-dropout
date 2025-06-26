#!/usr/bin/env python3
"""
Training script for PPO RNN agent.
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
from pathlib import Path
from types import SimpleNamespace
from functools import partial as bind

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

from baselines.ppo_rnn.ppo_rnn import train_ppo_rnn

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
    parser = argparse.ArgumentParser(description='Train PPO RNN agent')
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for evaluation')
    return parser.parse_args()

def load_config(argv=None):
    """Load configuration from YAML file with support for named configs."""
    configs = ruamel.yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'config.yaml').read())
    
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config_dict = embodied.Config(configs['defaults'])

    for name in parsed.configs:
        config_dict = config_dict.update(configs[name])
    config_dict = embodied.Flags(config_dict).parse(other)
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    
    # Print config in a more readable format
    def print_config_recursive(obj, indent=0):
        for key, value in vars(obj).items():
            if isinstance(value, SimpleNamespace):
                print("  " * indent + f"{key}:")
                print_config_recursive(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print("\nConfiguration:")
    print("-" * 50)
    print_config_recursive(config)
    print("-" * 50)

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

def main():
    """Main entry point for PPO RNN training."""
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    args = parse_args()
    config = load_config(argv)
    
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
    
    # Create output directories
    os.makedirs("runs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Create environment
    envs = make_envs(config, num_envs=config.num_envs)
    
    # Calculate number of iterations
    num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)
    
    # Run training
    print(f"Starting PPO RNN training on {config.task} with {config.num_envs} environments")
    print(f"Training for {num_iterations} iterations")
    if hasattr(config, 'use_wandb') and config.use_wandb:
        print(f"Wandb logging enabled - project: {getattr(config, 'wandb_project', 'sensor-dropout')}")
    else:
        print("Wandb logging disabled")
    
    trained_agent = train_ppo_rnn(envs, config, config.seed, num_iterations=num_iterations)
    print("Training completed!")

if __name__ == "__main__":
    main() 