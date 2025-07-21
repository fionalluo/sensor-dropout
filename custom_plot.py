import os
import logging
import sys
from slurm_utils import run_cmd
from plot.plot import get_plot_filename

# --- Configuration ---
# The base directory where all experiment data is stored.
# This can be a local folder or a shared directory on a cluster.
# The `plot.py` script will scan this directory for `run-*` folders.
# The default in plot.py is `"/mnt/kostas-graid/datasets/vlongle/rl_sensors"`
# but we can override it here if needed.
# For local testing, you might use:
# DATA_DIR = "wandb" 
DATA_DIR = "/mnt/kostas-graid/datasets/vlongle/rl_sensors"

# The directory where all generated plots will be saved.
# This keeps your project root clean.
PLOT_DIR = "figures"

def get_project_name(configs_str, baseline="ppo_dropout", version=1):
    """Get project name from config names (masking strategy is now embedded in config)."""
    return f"{baseline}-{configs_str}-v{version}"

def plot_all(configs=["debug_tiger_cycle", "debug_tiger_adaptive_naive", "debug_tiger_adaptive_warmup_no_smoothing"], 
             filename=None, versions=[1, 1, 1], seeds=None, run_dir=DATA_DIR, plot_dir=PLOT_DIR):
    """Plot comparison of different configs (each has masking strategy embedded)."""
    projects = []
    
    # Add projects for each config (masking strategy is embedded in config name)
    for config, v in zip(configs, versions):
        project = get_project_name(config, version=v)
        projects.append(project)
    
    cmd = f"python plot/plot.py --projects {' '.join(projects)}"
    cmd += f" --run_dir {run_dir}"
    cmd += f" --plot_dir {plot_dir}"
    cmd += " --metrics full_eval_return/env_mean/mean_return"
    cmd += " --ymin 0"
    if "tiger" in configs[0]:
        cmd += " --ymax 10"
    else:
        cmd += " --ymax 100"
    if filename is not None:
        cmd += f" --filename {filename}"
    if seeds is not None:
        cmd += f" --seeds {' '.join(str(s) for s in seeds)}"
    
    success = run_cmd(cmd, conda_env="rl_sensor")
    
    # Figure out the filename that plot.py will generate
    metric = "full_eval_return/env_mean/mean_return"
    # If a custom filename was provided, use it. Otherwise, generate the expected one.
    if filename is None:
        filename = get_plot_filename('comparison', projects, [metric], plot_dir=plot_dir)
    
    if success:
        logging.info(f"Successfully generated plot: {filename}")
        return [filename]
    else:
        logging.error("Plot generation failed or did not produce the expected output.")
        return []

def plot_env_perf(configs, filename=None, run_dir=DATA_DIR, plot_dir=PLOT_DIR):
    """Plot individual environment performance for different configs."""
    # For debug tiger configs, we have 2 environments
    num_envs = 2

    wandb_projects = []
    for config in configs:
        wandb_projects.append(get_project_name(config))

    metrics = [f"full_eval_return/env{i}/mean_return" for i in range(1, num_envs + 1)]

    cmd = f"python plot/plot.py --projects {' '.join(wandb_projects)}"
    cmd += f" --run_dir {run_dir}"
    cmd += f" --plot_dir {plot_dir}"
    cmd += f" --metrics {' '.join(metrics)}"
    cmd += " --ymin 0"
    cmd += " --ymax 10"
    if filename is not None:
        cmd += f" --filename {filename}"

    success = run_cmd(cmd, conda_env="rl_sensor")

    # Figure out the filename that plot.py will generate
    if filename is None:
        filename = get_plot_filename('multiple_metrics', wandb_projects, metrics, plot_dir=plot_dir)

    if success:
        logging.info(f"Successfully generated plot: {filename}")
        return [filename]
    else:
        logging.error("Plot generation failed.")
        return []

def plot_env_rate(config="debug_tiger_adaptive_naive", use_empirical_prob=True, num_envs=2,
                  filename=None, version=1, run_dir=DATA_DIR, plot_dir=PLOT_DIR):
    """Plot environment selection rates for a specific config."""
    wandb_project = get_project_name(config, version=version)
    
    # Extract masking strategy from config name for metric naming
    if "adaptive" in config:
        masking_strategy = "adaptive"
    elif "cycle" in config:
        masking_strategy = "cycle"
    else:
        masking_strategy = "uniform"
    
    metrics = []
    for i in range(1, num_envs + 1):
        if use_empirical_prob:
            metrics.append(f"empirical_masking_strategy/{masking_strategy}_env{i}_empirical_prob")
        else:
            metrics.append(f"masking_strategy/{masking_strategy}_env{i}_prob")

    cmd = f"python plot/plot.py --projects {wandb_project}"
    cmd += f" --run_dir {run_dir}"
    cmd += f" --plot_dir {plot_dir}"
    cmd += f" --metrics {' '.join(metrics)}"
    cmd += " --ymin 0"
    cmd += " --ymax 1"
    if filename is not None:
        cmd += f" --filename {filename}"
    success = run_cmd(cmd, conda_env="rl_sensor")

    # Figure out the filename that plot.py will generate
    if filename is None:
        filename = get_plot_filename('multiple_metrics', [wandb_project], metrics, plot_dir=plot_dir)

    if success:
        logging.info(f"Successfully generated plot: {filename}")
        return [filename]
    else:
        logging.error("Plot generation failed.")
        return []

if __name__ == "__main__":
    # Use the new debug configs which have masking strategy embedded
    # configs = ["debug_tiger_cycle", "debug_tiger_adaptive_naive", "debug_tiger_adaptive_warmup_no_smoothing"]
    # version = [1, 1, 1]

    # configs = ["debug_tiger_cycle", "debug_tiger_cycle", "debug_tiger_adaptive_warmup_no_smoothing"]
    # version = [1, 2, 1]
    tiger_configs = ["debug_tiger_cycle", 
               "debug_tiger_adaptive_window1",
               "debug_tiger_adaptive_warmup_window1",
               "debug_tiger_adaptive_warmup_window100",
               "debug_tiger_adaptive_warmup_window100_ema"]

    version = 5
    configs = tiger_configs
    versions = [version] * len(tiger_configs)

    config_idx = 1

    # maze_configs = [
    #     "debug_maze_cycle",
    #     "debug_maze_adaptive_window1",
    # ]
    # version = [2] * len(maze_configs)
    # configs = maze_configs

    # Define a base name for the plots for this specific run
    base_plot_name = f"{configs[config_idx]}_v{versions[config_idx]}"
    
    # Centralize plot directory management
    # This keeps the plot directory consistent and easy to change.
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Custom filenames now include the plot directory
    comparison_filename = os.path.join(PLOT_DIR, f"{base_plot_name}_comparison.png")
    env_rate_filename = os.path.join(PLOT_DIR, f"{base_plot_name}_env_rate_empirical.png")
    env_perf_filename = os.path.join(PLOT_DIR, f"{base_plot_name}_env_perf_comparison.png")

    seeds = None
    
    # The plot functions will now use the global DATA_DIR and PLOT_DIR
    plot_all(
        [configs[config_idx], configs[0]], 
        versions=[versions[config_idx], versions[0]], 
        filename=comparison_filename, 
        seeds=seeds
    )

    plot_env_rate(
        configs[config_idx], 
        use_empirical_prob=True, 
        filename=env_rate_filename, 
        version=versions[config_idx]
    )
    
    plot_env_perf(
        [configs[0], configs[config_idx]], 
        filename=env_perf_filename
    )