#!/usr/bin/env python3
"""
Training script for SB3 PPO subset policies.

This script trains separate SB3 PPO policies for different observation subsets
(e.g., env1, env2, env3, env4) and saves them to the policies directory with
the structure: policies/ppo/{task_name}/{env_name}/policy_{timestamp}.zip

Adapted from the CleanRL version to work with SB3's PPO implementation.
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
import multiprocessing as mp
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

# Import SB3 PPO training function
from baselines.ppo.train import train_ppo
from baselines.shared.config_utils import dict_to_namespace
from stable_baselines3 import PPO

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train SB3 PPO policies for each eval subset')
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
    parser.add_argument('--policy_type', type=str, choices=['ppo'], default='ppo',
                       help='Type of policy to train (only ppo supported for SB3)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()

def load_config(configs_names=None, policy_type='ppo'):
    """Load configuration from YAML file with support for named configs."""
    # Choose config file based on policy type (SB3 uses ppo config)
    config_path = embodied.Path(__file__).parent.parent / 'baselines/ppo/config.yaml'
    
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

def train_subset_policy(config, subset_name, eval_keys, output_dir, device, policy_type='ppo', debug=False):
    """Train an SB3 PPO policy for a specific subset of observations."""
    
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
    
    # Update keys for this subset - set keys to the subset keys so the agent uses the correct keys
    subset_config.keys = SimpleNamespace(**eval_keys)
    
    # Update exp_name to include subset name for distinct wandb logging and best model paths
    if hasattr(subset_config, 'exp_name'):
        subset_config.exp_name = f"{subset_config.exp_name}_{subset_name}"
    else:
        subset_config.exp_name = f"subset_{subset_name}"
    
    print(f"Subset config exp_name: {subset_config.exp_name}")
    print(f"This ensures unique best model paths for each subset")
    print(f"Training with vanilla SB3 PPO (no dropout) on observation subset: {subset_name}")
    print(f"Note: Custom evaluation across subsets is disabled for subset policies")
    
    # Calculate number of iterations (optional, for logging)
    num_iterations = config.total_timesteps // (config.num_envs * 2048)  # SB3 default n_steps
    print(f"Training for approximately {num_iterations} iterations")
    
    # Train the policy using vanilla SB3 PPO (no dropout)
    print(f"Starting training for {subset_name}...")
    # Disable custom evaluation for subset policies since each policy is trained on a specific subset
    trained_agent = train_ppo(None, subset_config, config.seed, enable_custom_eval=False)
    
    # Load the best model instead of the final model
    # Each subset has a unique exp_name (e.g., "maze_ppo_env1", "maze_ppo_env2")
    best_model_path = f"./best_models/ppo-{subset_config.task}-{subset_config.exp_name}-seed{config.seed}/best_model.zip"
    print(f"Looking for best model at: {best_model_path}")
    if os.path.exists(best_model_path):
        print(f"✓ Loading best model from {best_model_path}")
        trained_agent = PPO.load(best_model_path)
    else:
        print(f"✗ Best model not found at {best_model_path}, using final model")
        print(f"  This means the final model was the best performing model")
    
    # Save the policy with timestamp and env number in path
    policy_dir = os.path.join(output_dir, f"{subset_name}_policy")
    os.makedirs(policy_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_filename = f"policy_{timestamp}.zip"
    policy_path = os.path.join(policy_dir, policy_filename)
    
    # Check if policy already exists
    if os.path.exists(policy_path):
        print(f"Overwriting existing policy at {policy_path}")
    
    # Save the SB3 model with eval_keys as custom attribute
    trained_agent.eval_keys = eval_keys  # Store eval_keys in the model
    trained_agent.subset_name = subset_name  # Store subset name in the model
    trained_agent.save(policy_path)
    
    print(f"{policy_type.upper()} policy saved to {policy_path}")
    print(f"  Eval keys: {eval_keys}")
    print(f"  Subset: {subset_name}")
    
    # Clean up temporary best model files
    best_model_dir = f"./best_models/ppo-{subset_config.task}-{subset_config.exp_name}-seed{config.seed}"
    if os.path.exists(best_model_dir):
        print(f"Cleaning up temporary best model directory: {best_model_dir}")
        shutil.rmtree(best_model_dir)
    
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
    if not hasattr(config, 'num_eval_configs'):
        print("Error: No num_eval_configs found in config")
        return
    
    num_eval_configs = config.num_eval_configs
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
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"All policies saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nTrained policies:")
    for subset_name, policy_path in trained_policies.items():
        print(f"  {subset_name}: {policy_path}")
    
    print(f"\nEach policy is the best performing version for its subset.")
    print(f"To load a specific subset policy, use:")
    print(f"  python subset_policies/load_subset_policy.py --policy_dir {output_dir} --subset env1")

if __name__ == "__main__":
    # Use 'spawn' for multiprocessing to avoid issues with libraries like wandb
    mp.set_start_method("spawn", force=True)
    main() 