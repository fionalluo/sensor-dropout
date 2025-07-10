#!/usr/bin/env python3
"""
CLI entry point for multi-teacher distillation training.
"""

import os
import sys
import argparse
from types import SimpleNamespace
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from baselines.ppo_distill.simple_imitation import train_simple_imitation
from baselines.shared.config_utils import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-teacher distillation training")
    
    parser.add_argument(
        "--configs", 
        type=str, 
        required=True,
        help="Configuration name (e.g., gymnasium_tigerdoorkey)"
    )
    
    parser.add_argument(
        "--expert_policy_dir", 
        type=str, 
        required=True,
        help="Directory containing expert policies"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device for training (cpu/cuda)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="baselines/ppo_distill/config.yaml",
        help="Path to config file"
    )
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create config object from command line arguments and config file."""
    # Create argv list for load_config (it expects command line args)
    argv = ['--configs', args.configs]
    
    # Load base config from file - keep it as-is!
    config = load_config(argv, args.config_file)
    
    # Only override essential runtime parameters
    config.seed = args.seed
    config.expert_policy_dir = args.expert_policy_dir
    
    # Ensure basic wandb config exists (minimal fallbacks)
    if not hasattr(config, 'wandb_project'):
        config.wandb_project = "ppo-distill-baseline"
    if not hasattr(config, 'wandb_entity'):
        config.wandb_entity = None
    if not hasattr(config, 'track'):
        config.track = True
        
    # Use experiment name from config file, or create one if not specified
    if not hasattr(config, 'exp_name') or not config.exp_name:
        config.exp_name = f"multi_teacher_{args.configs}_{args.seed}"
    
    print(f"🔧 Using experiment name: {config.exp_name}")
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    config = create_config_from_args(args)
    
    print("Simple Imitation Learning Training")
    print("=" * 50)
    print(f"Config: {args.configs}")
    print(f"Expert Policy Dir: {args.expert_policy_dir}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Debug: {args.debug}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Distillation Configuration:")
    print(f"  Learning rate: {config.distillation.learning_rate}")
    print(f"  Batch size: {config.distillation.batch_size}")
    print(f"Evaluation Configuration:")
    print(f"  Eval frequency: {config.eval.eval_freq}")
    print(f"  Eval episodes: {config.eval.n_eval_episodes}")
    print("=" * 50)
    
    # Check if expert policy directory exists
    if not os.path.exists(args.expert_policy_dir):
        print(f"Error: Expert policy directory not found: {args.expert_policy_dir}")
        print("Please run train_subset_policies.sh first to create expert policies.")
        sys.exit(1)
    
    # Train
    trained_policy = train_simple_imitation(
        config=config,
        expert_policy_dir=args.expert_policy_dir,
        device=args.device,
        debug=args.debug
    )
    
    print("Training completed successfully!")
    return trained_policy


if __name__ == "__main__":
    main() 