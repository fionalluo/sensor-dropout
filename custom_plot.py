import os
import logging
import sys
from slurm_utils import run_cmd
from plot.plot import get_plot_filename


def get_project_name( masking_strategy, configs_str, version=1,
                     baseline="ppo_dropout"):
    return f"{baseline}-{masking_strategy}-{configs_str}-v{version}"

def plot_all(env="gymnasium_tigerdoorkey", version=1):
    projects = []
    # for strat in ["cycle", "adaptive", "uniform"]:
    for strat in ["cycle", "adaptive"]:
        wandb_project = get_project_name(strat, 
                                         env, version)
        projects.append(wandb_project)
    
    cmd = f"python plot/plot.py --projects {' '.join(projects)}"
    cmd += " --run_dir wandb"
    cmd += " --metrics full_eval_return/env_mean/mean_return"
    cmd += " --ymin 0"
    # cmd += " --ymax 10"
    
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

def plot_env_perf(strats=["adaptive"], num_envs=2, env="gymnasium_tigerdoorkey", version=1):
    wandb_projects = []
    for strat in strats:
        wandb_projects.append(get_project_name(strat, env, version))

    metrics = [f"full_eval_return/env{i}/mean_return" for i in range(1, num_envs + 1)]

    cmd = f"python plot/plot.py --projects {' '.join(wandb_projects)}"
    cmd += " --run_dir wandb"
    cmd += f" --metrics {' '.join(metrics)}"
    cmd += " --ymin 0"
    # cmd += " --ymax 10"

    success = run_cmd(cmd, conda_env="rl_sensor")

    # Figure out the filename that plot.py will generate
    filename = get_plot_filename('multiple_metrics', wandb_projects, metrics)

    if success:
        logging.info(f"Successfully generated plot: {filename}")
        return [filename]
    else:
        logging.error("Plot generation failed.")
        return []



def plot_env_rate(strat="adaptive", use_empirical_prob=True, num_envs=4, env="gymnasium_tigerdoorkey", version=1):
    wandb_project = get_project_name(strat, env, version)
    
    metrics = []
    for i in range(1, num_envs + 1):
        if use_empirical_prob:
            metrics.append(f"masking_strategy/{strat}_env{i}_empirical_prob")
        else:
            metrics.append(f"masking_strategy/{strat}_env{i}_prob")

    cmd = f"python plot/plot.py --projects {wandb_project}"
    cmd += " --run_dir wandb"
    cmd += f" --metrics {' '.join(metrics)}"
    cmd += " --ymin 0"
    cmd += " --ymax 1"
    
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
    envs = [
        # "gymnasium_tigerdoorkey", 
        #     "gymnasium_maze", 
            "gymnasium_maze11"]
    # for env in envs:
    #     plot_all(env)

    # plot_env_perf(strats=["cycle", "adaptive"], 
    #               num_envs=4)
    # plot_env_perf(strats=["cycle"], 
    #               num_envs=4)

    # for strat in ["cycle", "adaptive"]:
    for env in envs:
        for strat in ["adaptive"]:
            # plot_env_perf(strat)
            for use_empirical_prob in [True, False]:
                plot_env_rate(strat, use_empirical_prob=use_empirical_prob)
