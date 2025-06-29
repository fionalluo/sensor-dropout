#!/usr/bin/env python3
"""
Script to inspect checkpoint contents and understand configuration mismatch.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from baselines.shared.policy_utils import load_policy_checkpoint

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file to understand its contents."""
    
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("="*60)
    
    # Load checkpoint
    checkpoint = load_policy_checkpoint(checkpoint_path, device='cpu')
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\nConfig from checkpoint:")
        print(f"  Task: {getattr(config, 'task', 'Unknown')}")
        print(f"  Keys: {getattr(config, 'keys', 'Not set')}")
        print(f"  Full keys: {getattr(config, 'full_keys', 'Not set')}")
        
        if hasattr(config, 'eval_keys'):
            print(f"  Eval keys:")
            for env_name in dir(config.eval_keys):
                if not env_name.startswith('_'):
                    env_keys = getattr(config.eval_keys, env_name)
                    print(f"    {env_name}: {env_keys}")
    
    if 'eval_keys' in checkpoint:
        print(f"\nEval keys from checkpoint: {checkpoint['eval_keys']}")
    
    if 'subset_name' in checkpoint:
        print(f"\nSubset name: {checkpoint['subset_name']}")
    
    if 'policy_type' in checkpoint:
        print(f"\nPolicy type: {checkpoint['policy_type']}")
    
    if 'agent_state_dict' in checkpoint:
        state_dict = checkpoint['agent_state_dict']
        print(f"\nAgent state dict keys: {list(state_dict.keys())}")
        
        # Look for key architectural components
        mlp_keys = [k for k in state_dict.keys() if 'mlp_encoder' in k]
        cnn_keys = [k for k in state_dict.keys() if 'cnn_encoder' in k or 'heavyweight_cnn_encoder' in k]
        latent_keys = [k for k in state_dict.keys() if 'latent_projector' in k]
        
        print(f"\nMLP encoder keys ({len(mlp_keys)}): {mlp_keys[:5]}...")
        print(f"CNN encoder keys ({len(cnn_keys)}): {cnn_keys[:5]}...")
        print(f"Latent projector keys ({len(latent_keys)}): {latent_keys[:5]}...")
        
        # Check specific layer sizes
        for key in mlp_keys:
            if '.weight' in key and '0.weight' in key:
                print(f"  {key}: {state_dict[key].shape}")
        
        for key in latent_keys:
            if '.weight' in key and '0.weight' in key:
                print(f"  {key}: {state_dict[key].shape}")

def main():
    """Main function to inspect checkpoints."""
    
    # Path to PPO policies
    policy_dir = "policies/ppo/tigerdoorkey"
    
    if not os.path.exists(policy_dir):
        print(f"Policy directory not found: {policy_dir}")
        return
    
    # Find all policy files
    for item in os.listdir(policy_dir):
        item_path = os.path.join(policy_dir, item)
        if os.path.isdir(item_path):
            # Look for policy files
            import glob
            policy_files = glob.glob(os.path.join(item_path, "policy*.pt"))
            if policy_files:
                # Sort by modification time and select the most recent
                policy_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_policy = policy_files[0]
                
                print(f"\n{'='*80}")
                print(f"Inspecting {item}: {latest_policy}")
                print(f"{'='*80}")
                
                try:
                    inspect_checkpoint(latest_policy)
                except Exception as e:
                    print(f"Error inspecting {latest_policy}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    main() 