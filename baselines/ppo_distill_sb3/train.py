#!/usr/bin/env python3
"""
Training script for PPO Distill SB3 baseline.
Trains a student policy that learns from multiple expert subset policies using SB3.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
import ruamel.yaml
import multiprocessing as mp
import trailenv
from pathlib import Path
from types import SimpleNamespace

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add embodied to path (adjust if needed)
import embodied

from baselines.ppo_distill_sb3.ppo_distill_sb3 import train_ppo_distill_sb3
from baselines.shared.config_utils import load_config

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
    parser = argparse.ArgumentParser(description="Train PPO Distill SB3 agent")
    
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

def load_config(configs_names=None, policy_type='ppo_distill_sb3'):
    """Load configuration from YAML file with support for named configs."""
    # Choose config file based on policy type (SB3 uses ppo_distill_sb3 config)
    config_path = embodied.Path(__file__).parent / 'config.yaml'
    
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
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(v) for v in d]
        else:
            return d
    
    config = dict_to_namespace(config_dict)
    
    return config

def main():
    """Main training function."""
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    args = parse_args()
    
    # Load configuration using the same logic as PPO
    config = load_config(args.configs, 'ppo_distill_sb3')
    
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
        print("Please run train_subset_policies_sb3.sh first to create expert policies.")
        return
    
    print(f"Expert policy directory: {args.expert_policy_dir}")
    
    # Calculate number of iterations
    num_iterations = config.total_timesteps // (config.num_envs * 2048)  # SB3 default n_steps
    
    print(f"Training PPO Distill SB3 agent...")
    print(f"Task: {config.task}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Number of environments: {config.num_envs}")
    print(f"Steps per iteration: 2048 (SB3 default)")
    
    # Train the agent
    trained_agent = train_ppo_distill_sb3(
        config=config,
        seed=config.seed,
        expert_policy_dir=args.expert_policy_dir,
        device=device
    )
    
    print("Training completed!")
    
    # Save the trained agent
    if hasattr(config, 'save_model') and config.save_model:
        save_path = f"ppo_distill_sb3_{config.task}_{config.seed}.zip"
        trained_agent.save(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    # Use 'spawn' for multiprocessing to avoid issues with libraries like wandb
    mp.set_start_method("spawn", force=True)
    main() 