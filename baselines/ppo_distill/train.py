#!/usr/bin/env python3
"""
Training script for PPO Distill baseline.
Trains a student policy that learns from multiple expert subset policies.
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
import copy

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

from baselines.ppo_distill.ppo_distill import train_ppo_distill

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO Distill agent")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default='config.yaml', 
        help='Path to configuration file'
    )
    parser.add_argument(
        "--configs", 
        type=str, 
        nargs='+', 
        default=['gymnasium_tigerdoorkey'],
        help="Which named configs to apply"
    )
    parser.add_argument(
        "--expert_policy_dir", 
        type=str, 
        required=True,
        help="Directory containing expert subset policies"
    )
    parser.add_argument(
        "--student_policy_type", 
        type=str, 
        default="ppo_rnn",
        choices=["ppo", "ppo_rnn"],
        help="Type of student agent to distill into (ppo or ppo_rnn)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--cuda", 
        action="store_true",
        default=None,
        help="Use CUDA (overrides config)"
    )
    parser.add_argument(
        "--track", 
        action="store_true",
        default=None,
        help="Track with wandb (overrides config)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()

def load_config(argv=None):
    """Load configuration from YAML file with support for named configs."""
    configs = ruamel.yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'config.yaml').read())
    
    # First, parse the config names and any other flags
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config_dict = embodied.Config(configs['defaults'])

    # Apply the named configs
    for name in parsed.configs:
        config_dict = config_dict.update(configs[name])
    
    # Only parse remaining flags if there are any
    if other:
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

def make_envs_ppo_distill(config, num_envs):
    """Create vectorized environments for PPO Distill with ALL observation keys."""
    print("🔧 Creating PPO Distill environments with ALL observation keys...")
    
    # Collect ALL observation keys from ALL config.env keys
    all_mlp_keys = set()
    all_cnn_keys = set()
    
    # Get keys from all env configurations
    if hasattr(config, 'eval_keys'):
        for env_name in ['env1', 'env2', 'env3', 'env4']:
            if hasattr(config.eval_keys, env_name):
                env_config = getattr(config.eval_keys, env_name)
                if hasattr(env_config, 'mlp_keys'):
                    # Parse the regex pattern to get actual keys
                    import re
                    pattern = env_config.mlp_keys
                    # Remove word boundaries and split by | to get individual keys
                    pattern_clean = pattern.replace('\\b', '').replace('(', '').replace(')', '')
                    keys = pattern_clean.split('|')
                    for key in keys:
                        all_mlp_keys.add(key)
                
                if hasattr(env_config, 'cnn_keys'):
                    pattern = env_config.cnn_keys
                    if pattern != '^$':  # Not empty
                        # Parse CNN keys similarly
                        pattern_clean = pattern.replace('\\b', '').replace('(', '').replace(')', '')
                        keys = pattern_clean.split('|')
                        for key in keys:
                            all_cnn_keys.add(key)
    
    print(f"🔧 All MLP keys needed: {sorted(all_mlp_keys)}")
    print(f"🔧 All CNN keys needed: {sorted(all_cnn_keys)}")
    
    # Create a modified config that includes all keys
    modified_config = copy.deepcopy(config)
    
    # Set the full_keys to include all keys from all envs
    if not hasattr(modified_config, 'full_keys'):
        modified_config.full_keys = SimpleNamespace()
    
    # Create regex patterns that match all the keys we need
    all_mlp_pattern = '|'.join([f'\\b{key}\\b' for key in all_mlp_keys])
    all_cnn_pattern = '|'.join([f'\\b{key}\\b' for key in all_cnn_keys]) if all_cnn_keys else '^$'
    
    modified_config.full_keys.mlp_keys = all_mlp_pattern
    modified_config.full_keys.cnn_keys = all_cnn_pattern
    
    # Create environments with the modified config
    suite, task = modified_config.task.split('_', 1)
    ctors = []
    for index in range(num_envs):
        ctor = lambda: make_env(modified_config)
        if hasattr(modified_config, 'envs') and hasattr(modified_config.envs, 'parallel') and modified_config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, modified_config.envs.parallel)
        if hasattr(modified_config, 'envs') and hasattr(modified_config.envs, 'restart') and modified_config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    
    print(f"🔧 Environment created with {len(envs[0].obs_space.keys())} observation keys")
    return embodied.BatchEnv(envs, parallel=(hasattr(modified_config, 'envs') and hasattr(modified_config.envs, 'parallel') and modified_config.envs.parallel != 'none'))

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
    """Main training function."""
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    args = parse_args()
    
    # Filter out boolean flags from argv before passing to embodied config system
    filtered_argv = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ['--cuda', '--track', '--debug']:
            # Skip boolean flags
            i += 1
        elif arg == '--expert_policy_dir' and i + 1 < len(argv):
            # Skip expert_policy_dir and its value
            i += 2
        elif arg == '--student_policy_type' and i + 1 < len(argv):
            # Skip student_policy_type and its value
            i += 2
        else:
            filtered_argv.append(arg)
            i += 1
    
    # Load configuration
    config = load_config(filtered_argv)
    
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
    
    # Verify expert policy directory exists
    if not os.path.exists(args.expert_policy_dir):
        print(f"Error: Expert policy directory not found: {args.expert_policy_dir}")
        print("Please run train_subset_policies.sh first to create expert policies.")
        return
    
    print(f"Expert policy directory: {args.expert_policy_dir}")
    
    # Create environments
    envs = make_envs_ppo_distill(config, num_envs=config.num_envs)
    
    # Calculate number of iterations
    num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)
    
    print(f"Training PPO Distill agent...")
    print(f"Task: {config.task}")
    print(f"Student policy type: {args.student_policy_type}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Number of environments: {config.num_envs}")
    print(f"Steps per iteration: {config.num_steps}")
    
    # Train the agent
    trained_agent = train_ppo_distill(
        envs=envs,
        config=config,
        seed=config.seed,
        expert_policy_dir=args.expert_policy_dir,
        student_policy_type=args.student_policy_type,
        num_iterations=num_iterations
    )
    
    print("Training completed!")
    
    # Save the trained agent
    if hasattr(config, 'save_model') and config.save_model:
        save_path = f"ppo_distill_{config.task}_{config.seed}.pt"
        torch.save({
            'agent_state_dict': trained_agent.state_dict(),
            'config': config,
            'expert_policy_dir': args.expert_policy_dir
        }, save_path)
        print(f"Model saved to: {save_path}")
    
    # Clean up
    envs.close()


if __name__ == "__main__":
    main() 