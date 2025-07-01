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
from pathlib import Path
from types import SimpleNamespace

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add embodied to path (adjust if needed)
import embodied

from baselines.ppo_distill.ppo_distill import train_ppo_distill
from baselines.shared.policy_utils import make_envs_for_distillation
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

def make_envs_ppo_distill(config, num_envs):
    """Create vectorized environments for PPO Distill with ALL observation keys."""
    return make_envs_for_distillation(config, num_envs)

def main():
    """Main training function."""
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    args = parse_args()
    
    # Load configuration using the same logic as PPO
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(argv, config_path)
    
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