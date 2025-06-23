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

# Add thesis directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

# Import unified PPO implementation
from ppo import main as ppo_main

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

# --- Config loading ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=[], help="Which named configs to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Override seed manually.")
    return parser.parse_args()

def load_config(argv=None):
    configs = ruamel.yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'config.yaml').read())
    
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config_dict = embodied.Config(configs['defaults'])

    for name in parsed.configs:
        config_dict = config_dict.update(configs[name])
    config_dict = embodied.Flags(config_dict).parse(other)
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    print(config)

    return config

# --- Environment creation ---
def make_envs(config):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(config.num_envs):
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
        kwargs.update({
        'record': config.run.script == 'eval_only'  # record in eval only for now (single environment)
        })
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

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="thesis/config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=['defaults'], help="List of config names to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Optional override seed")
    return parser.parse_args(argv)

# --- PPO Training ---
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args(argv)
    config = load_config(argv)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Create environment
    env = make_env(config)
    envs = make_envs(config)

    # Run training with unified PPO
    ppo_main(envs, config, seed)

if __name__ == "__main__":
    main()