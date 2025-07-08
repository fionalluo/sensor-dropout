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

from baselines.ppo_distill_sb3_2.simple_imitation import train_simple_imitation
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
        default="baselines/ppo_distill_sb3_2/config.yaml",
        help="Path to config file"
    )
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create config object from command line arguments and config file."""
    # Create argv list for load_config (it expects command line args)
    argv = ['--configs', args.configs]
    
    # Load base config from file
    config = load_config(argv, args.config_file)
    
    # Override with command line arguments
    config.seed = args.seed
    
    # Add multi-teacher specific config
    config.expert_policy_dir = args.expert_policy_dir
    
    # Set distillation-specific parameters
    config.learning_rate = getattr(config, 'learning_rate', 3e-4)
    config.batch_size = getattr(config, 'batch_size', 64)
    config.steps_per_rollout = getattr(config, 'steps_per_rollout', 128)
    config.num_minibatches = getattr(config, 'num_minibatches', 4)
    config.update_epochs = getattr(config, 'update_epochs', 4)
    config.temperature = getattr(config, 'temperature', 1.0)
    config.distillation_loss_weight = getattr(config, 'distillation_loss_weight', 1.0)
    config.eval_freq = getattr(config, 'eval_freq', 10000)
    config.n_eval_episodes = getattr(config, 'n_eval_episodes', 5)
    
    # Ensure wandb config
    if not hasattr(config, 'wandb_project'):
        config.wandb_project = "multi-teacher-distillation"
    if not hasattr(config, 'wandb_entity'):
        config.wandb_entity = None
    if not hasattr(config, 'track'):
        config.track = True
        
    # Create experiment name
    config.exp_name = f"multi_teacher_{args.configs}_{args.seed}"
    
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
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps per rollout: {config.steps_per_rollout}")
    print("=" * 50)
    
    # Check if expert policy directory exists
    if not os.path.exists(args.expert_policy_dir):
        print(f"Error: Expert policy directory not found: {args.expert_policy_dir}")
        print("Please run train_subset_policies_sb3.sh first to create expert policies.")
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