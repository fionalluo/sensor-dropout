#!/usr/bin/env python3
"""
Helper script to extract the number of evaluation subsets from the config.
Used by slurm scripts to determine how many parallel jobs to submit.
"""

import os
import sys
import ruamel.yaml
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import embodied
from baselines.shared.config_utils import dict_to_namespace

def get_num_subsets(configs_names=None):
    """Get the number of evaluation subsets from the config."""
    # Choose config file (using ppo config)
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
                print(f"Warning: Config '{name}' not found in config file", file=sys.stderr)
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    
    # Get number of eval configs
    num_eval_configs = getattr(config, 'num_eval_configs', 4)
    
    return num_eval_configs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Get number of evaluation subsets from config')
    parser.add_argument('--configs', type=str, nargs='+', default=[], 
                       help='Which named configs to apply')
    args = parser.parse_args()
    
    num_subsets = get_num_subsets(args.configs)
    print(num_subsets) 