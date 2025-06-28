#!/usr/bin/env python3
"""
Entry point script for PPO LSTM training that can be run directly.
This script handles the import issues and provides a clean interface.
"""

import os
import sys
import time
import random
import argparse
import warnings
import pathlib
import importlib
from functools import partial as bind
from types import SimpleNamespace

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

# Import PPO LSTM training function
from baselines.ppo_lstm.ppo import train_ppo_lstm

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=[], help="Which named configs to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Override seed manually.")
    parser.add_argument('--task', type=str, default="gymnasium_BanditPathEnv5-v0", help="Environment task.")
    parser.add_argument('--num_envs', type=int, default=8, help="Number of parallel environments.")
    parser.add_argument('--num_steps', type=int, default=2048, help="Number of steps per rollout.")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--num_iterations', type=int, default=1000, help="Number of training iterations.")
    parser.add_argument('--use_wandb', action='store_true', help="Enable wandb logging.")
    parser.add_argument('--no_wandb', action='store_true', help="Disable wandb logging.")
    parser.add_argument('--wandb_project', type=str, default="sensor-dropout", help="Wandb project name.")
    return parser.parse_args()

def load_config(argv=None):
    """Load configuration from YAML file."""
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
        'atari': 'embodied.envs.atari:Atari',
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
        # kwargs.update({
        # 'record': config.run.script == 'eval_only'  # record in eval only for now (single environment)
        # })
        render_image = False
        if 'Pixel' in task:
            task = task.replace('Pixel', '')
        render_image = True
        kwargs.update({'render_image': render_image})

    env = ctor(task, **kwargs)
    return wrap_env(env, config)

def wrap_env(env, config):
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

def main(argv=None):
    """Main training function."""
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args()
    config = load_config(argv)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Create environment
    envs = make_envs(config, num_envs=config.num_envs)

    # Calculate number of iterations like in thesis
    num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)

    # Run training
    print(f"Starting PPO LSTM training on {config.task} with {config.num_envs} environments")
    print(f"Training for {num_iterations} iterations")
    if config.use_wandb:
        print(f"Wandb logging enabled - project: {config.wandb_project}")
    else:
        print("Wandb logging disabled")
    
    trained_agent = train_ppo_lstm(envs, config, seed, num_iterations=num_iterations)
    
    print("Training completed!")
    return trained_agent

if __name__ == "__main__":
    main() 