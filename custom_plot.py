import os
import logging
import sys
from slurm_utils import run_cmd
from plot.plot import get_plot_filename


def get_project_name(configs_str, baseline="ppo_dropout", version=1):
    """Get project name from config names (masking strategy is now embedded in config)."""
    return f"{baseline}-{configs_str}-v{version}"

def plot_all(configs=["debug_tiger_cycle", "debug_tiger_adaptive_naive", "debug_tiger_adaptive_warmup_no_smoothing"], filename=None, 
             versions=[1, 1, 1], seeds=None):
    """Plot comparison of different configs (each has masking strategy embedded)."""
    projects = []
    
    # Add projects for each config (masking strategy is embedded in config name)
    for config, v in zip(configs, versions):
        project = get_project_name(config, version=v)
        projects.append(project)
    
    cmd = f"python plot/plot.py --projects {' '.join(projects)}"
    cmd += " --run_dir wandb"
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
    filename = get_plot_filename('comparison', projects, [metric])
    
    if success:
        logging.info(f"Successfully generated plot: {filename}")
        return [filename]
    else:
        logging.error("Plot generation failed or did not produce the expected output.")
        return []

def plot_env_perf(configs,
                  filename=None):
    """Plot individual environment performance for different configs."""
    # For debug tiger configs, we have 2 environments
    num_envs = 2

    wandb_projects = []
    for config in configs:
        wandb_projects.append(get_project_name(config))

    metrics = [f"full_eval_return/env{i}/mean_return" for i in range(1, num_envs + 1)]

    cmd = f"python plot/plot.py --projects {' '.join(wandb_projects)}"
    cmd += " --run_dir wandb"
    cmd += f" --metrics {' '.join(metrics)}"
    cmd += " --ymin 0"
    cmd += " --ymax 10"
    if filename is not None:
        cmd += f" --filename {filename}"

    success = run_cmd(cmd, conda_env="rl_sensor")

    # Figure out the filename that plot.py will generate
    filename = get_plot_filename('multiple_metrics', wandb_projects, metrics)

    if success:
        logging.info(f"Successfully generated plot: {filename}")
        return [filename]
    else:
        logging.error("Plot generation failed.")
        return []

def plot_env_rate(config="debug_tiger_adaptive_naive", use_empirical_prob=True, num_envs=2,
                  filename=None):
    """Plot environment selection rates for a specific config."""
    wandb_project = get_project_name(config)
    
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
    cmd += " --run_dir wandb"
    cmd += f" --metrics {' '.join(metrics)}"
    cmd += " --ymin 0"
    cmd += " --ymax 1"
    if filename is not None:
        cmd += f" --filename {filename}"
    success = run_cmd(cmd, conda_env="rl_sensor")

    # Figure out the filename that plot.py will generate
    filename = get_plot_filename('multiple_metrics', [wandb_project], metrics)

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

    version = [1] * len(tiger_configs)
    maze_configs = [
        "debug_maze_cycle",
        "debug_maze_adaptive_window1",
    ]
    version = [2] * len(maze_configs)
    configs = maze_configs

    # filename = "figures/debug_tiger_comparison.png"
    filename = "figures/debug_maze_comparison.png"
    # seeds = [
    #     # 42, 
    #     #      43, 
    #     #      44, 
    #          45
    #          ]
    seeds = None
    plot_all(configs, versions=version, filename=filename, seeds=seeds)


    # plot_env_rate(configs[1], 
    #               use_empirical_prob=True, filename="figures/env_rate.png")
    # plot_env_perf([configs[0], configs[-1]], 
    #               filename="figures/env_perf.png")