#!/usr/bin/env python3
"""
Cleanup script to delete all runs from a specific WandB project in the local wandb folder.
"""

import os
import argparse
import logging
import yaml
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_wandb_info(run_dir):
    """Extract entity and project names from the run's config.yaml (if present).
    
    Returns (entity, project) – either may be None if not found.
    """
    cfg_path = os.path.join(run_dir, 'files', 'config.yaml')
    if not os.path.isfile(cfg_path):
        return None, None

    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logging.debug(f"Failed to parse {cfg_path}: {e}")
        return None, None

    # config.yaml can store primitives or dicts with a 'value' field
    def _resolve(val):
        if isinstance(val, dict):
            return val.get('value')
        return val

    entity = _resolve(cfg.get('wandb_entity') or cfg.get('entity'))
    project = _resolve(cfg.get('wandb_project') or cfg.get('project'))
    return entity, project

def cleanup_wandb_project(wandb_dir, project_name, dry_run=False):
    """
    Delete all runs from a specific project in the wandb directory.
    
    Parameters:
    -----------
    wandb_dir : str
        Path to the wandb directory containing run folders
    project_name : str
        Name of the project to clean up
    dry_run : bool
        If True, only print what would be deleted without actually deleting
    """
    wandb_path = Path(wandb_dir)
    
    if not wandb_path.exists():
        logging.error(f"WandB directory does not exist: {wandb_dir}")
        return
    
    if not wandb_path.is_dir():
        logging.error(f"Path is not a directory: {wandb_dir}")
        return
    
    # Find all run directories
    run_dirs = [d for d in wandb_path.iterdir() if d.is_dir() and d.name.startswith('run-')]
    
    if not run_dirs:
        logging.info(f"No run directories found in {wandb_dir}")
        return
    
    logging.info(f"Found {len(run_dirs)} run directories to check")
    
    runs_to_delete = []
    
    for run_dir in run_dirs:
        run_id = run_dir.name.split('-')[-1]
        entity, project = extract_wandb_info(run_dir)
        
        if project == project_name:
            runs_to_delete.append((run_dir, run_id, entity))
            logging.info(f"Found matching run: {run_id} (project: {project}, entity: {entity})")
        else:
            logging.debug(f"Skipping run {run_id}: project '{project}' != '{project_name}'")
    
    if not runs_to_delete:
        logging.info(f"No runs found for project '{project_name}'")
        return
    
    logging.info(f"Found {len(runs_to_delete)} runs to delete for project '{project_name}'")
    
    if dry_run:
        logging.info("DRY RUN MODE - No files will be deleted")
        for run_dir, run_id, entity in runs_to_delete:
            logging.info(f"Would delete: {run_dir}")
    else:
        # Ask for confirmation
        response = input(f"Are you sure you want to delete {len(runs_to_delete)} runs from project '{project_name}'? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logging.info("Operation cancelled by user")
            return
        
        # Delete the runs
        deleted_count = 0
        for run_dir, run_id, entity in runs_to_delete:
            try:
                shutil.rmtree(run_dir)
                logging.info(f"Deleted run {run_id}: {run_dir}")
                deleted_count += 1
            except Exception as e:
                logging.error(f"Failed to delete run {run_id}: {e}")
        
        logging.info(f"Successfully deleted {deleted_count}/{len(runs_to_delete)} runs")

def main():
    parser = argparse.ArgumentParser(
        description="Clean up WandB runs from a specific project in the local wandb folder"
    )
    parser.add_argument(
        "--project_name",
        help="Name of the WandB project to clean up",
        required=True
    )
    parser.add_argument(
        "--wandb_dir",
        default="wandb",
        help="Path to the wandb directory (default: 'wandb')"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cleanup_wandb_project(args.wandb_dir, args.project_name, dry_run=args.dry_run)

if __name__ == "__main__":
    main() 