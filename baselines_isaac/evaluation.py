# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate all checkpoints in a folder and log results to wandb."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
# Only require --task as CLI argument, everything else comes from config.yaml
# (Remove CLI parsing and AppLauncher from top level)

"""Rest everything follows."""

import os
import torch
import math
import re
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import glob
import yaml
import random
from baselines_isaac.ppo_dropout_any.isaac_dropout_wrapper import (
    IsaacProbabilisticDropoutWrapper, DropoutScheduler, load_task_dropout_config
)
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rollout_policy(env, agent, num_eval_episodes, rl_device):
    # Collect the first episode to finish for each of the first num_eval_episodes env indices
    total_rewards = [None] * num_eval_episodes
    episode_lengths = [None] * num_eval_episodes
    collected = set()
    num_envs = env.unwrapped.num_envs

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_rewards = torch.zeros(num_envs, device=rl_device)
    episode_lengths_env = torch.zeros(num_envs, device=rl_device)

    def extract_array(val):
        if isinstance(val, dict):
            if 'policy' in val:
                return val['policy']
            return next(iter(val.values()))
        return val

    while len(collected) < num_eval_episodes:
        try:
            context = torch.inference_mode()
        except AttributeError:
            context = torch.no_grad()
        with context:
            obs_torch = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_torch, is_deterministic=getattr(agent, 'is_deterministic', True))
        result = env.step(actions)
        if len(result) == 5:
            obs, rewards, terminated, truncated, infos = result
            terminated_arr = extract_array(terminated)
            truncated_arr = extract_array(truncated)
            dones = torch.logical_or(torch.as_tensor(terminated_arr), torch.as_tensor(truncated_arr))
        else:
            obs, rewards, dones, infos = result

        episode_rewards += rewards
        episode_lengths_env += 1

        if torch.any(dones):
            done_indices = torch.nonzero(dones).squeeze(1)
            for idx in done_indices:
                idx_int = int(idx)
                if idx_int < num_eval_episodes and idx_int not in collected:
                    total_rewards[idx_int] = episode_rewards[idx].item()
                    episode_lengths[idx_int] = episode_lengths_env[idx].item()
                    collected.add(idx_int)
                episode_rewards[idx] = 0.0
                episode_lengths_env[idx] = 0.0
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

    # Filter out any None (shouldn't happen, but for safety)
    total_rewards = [r for r in total_rewards if r is not None]
    episode_lengths = [l for l in episode_lengths if l is not None]
    return total_rewards, episode_lengths

def evaluate_checkpoint(env, agent, checkpoint_path, task, num_eval_episodes, rl_device):
    # Reset the agent (to clear RNN state, etc.)
    agent.reset()
    total_rewards, episode_lengths = rollout_policy(env, agent, num_eval_episodes, rl_device)
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    avg_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
    print(f"[RESULT] Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"[RESULT] Avg Reward: {avg_reward:.2f}")
    print(f"[RESULT] Avg Episode Length: {avg_episode_length:.1f}")
    import gc
    gc.collect()
    return avg_reward, avg_episode_length

def find_dropout_wrapper(env):
    """Walk through .env chain to find a wrapper with set_dropout_probability."""
    current = env
    while current is not None:
        if hasattr(current, 'set_dropout_probability'):
            return current
        current = getattr(current, 'env', None)
    return None

def evaluate_all_checkpoints(task, checkpoint_folder, num_eval_episodes, num_envs, env=None, wandb_project=None, wandb_entity=None, wandb_run_name=None):
    """
    Evaluate all checkpoints in a folder. If env is provided, use it; otherwise, create a new environment.
    Only close the environment if it was created here.
    Deeply reset the environment before each checkpoint evaluation.
    Optionally log results to wandb if wandb_project, wandb_entity, and wandb_run_name are provided.
    """
    import re
    # Optionally import wandb
    wandb = None
    if wandb_project and wandb_entity and wandb_run_name:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, resume="allow")
    # Load agent config to get steps_per_epoch
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    steps_per_epoch = agent_cfg["params"].get("steps_num", 1)
    # Find all checkpoint files matching regex ep_\d+
    checkpoint_pattern = os.path.join(checkpoint_folder, "*.pth")
    all_checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files = [f for f in all_checkpoint_files if re.search(r'ep_\d+', os.path.basename(f))]
    checkpoint_files = sorted(checkpoint_files, key=os.path.basename)
    if not checkpoint_files:
        print(f"[WARNING] No checkpoint files found in {checkpoint_folder}")
        return
    print(f"[INFO] Found {len(checkpoint_files)} checkpoint files to evaluate")
    # Prepare dropout probabilities
    dropout_probs = [0.0, 0.1, 0.25, 0.5]
    # Load key indices for dropout wrapper
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    key_indices = load_task_dropout_config(config_path, task)
    # Create environment ONCE if not provided
    close_env = False
    if env is None:
        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        env_cfg = parse_env_cfg(
            task, device=agent_cfg["params"]["config"]["device"], num_envs=num_envs, use_fabric=True
        )
        base_env = gym.make(task, cfg=env_cfg, render_mode=None)
        if isinstance(base_env.unwrapped, DirectMARLEnv):
            base_env = multi_agent_to_single_agent(base_env)
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        close_env = True
    else:
        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        rl_device = agent_cfg["params"]["config"]["device"]
        base_env = env
        # Ensure clip_obs and clip_actions are set even if env is provided
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # --- Ensure there is a dropout wrapper ONCE ---
    dropout_env = find_dropout_wrapper(base_env)
    if dropout_env is not None:
        print("Found dropout wrapper in the environment chain.")
        wrapped_env_for_eval = base_env
    else:
        print("No dropout wrapper found, inserting it just above gym.make (underneath any RlGamesVecEnvWrapper if present).")
        from baselines_isaac.ppo_dropout_any.isaac_dropout_wrapper import IsaacProbabilisticDropoutWrapper
        from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
        if isinstance(base_env, RlGamesVecEnvWrapper):
            inner_env = base_env.env
            dropout_wrapped = IsaacProbabilisticDropoutWrapper(inner_env, task_name=task)
            wrapped_env_for_eval = RlGamesVecEnvWrapper(dropout_wrapped, rl_device, clip_obs, clip_actions)
        else:
            wrapped_env_for_eval = IsaacProbabilisticDropoutWrapper(base_env, task_name=task)

    for checkpoint_path in checkpoint_files:
        # --- Create agent ONCE per checkpoint ---
        import copy
        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint_path
        agent_cfg["params"]["config"]["num_actors"] = wrapped_env_for_eval.unwrapped.num_envs
        from rl_games.torch_runner import Runner
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(checkpoint_path)
        # --- End agent creation ---
        for prob in dropout_probs:
            dropout_env = find_dropout_wrapper(wrapped_env_for_eval)
            if dropout_env is not None:
                print("Setting dropout probability to", prob)
                dropout_env.set_dropout_probability(prob)
                wrapped_env = wrapped_env_for_eval
            else:
                raise RuntimeError("No dropout wrapper found in the environment, even after trying to create one!")

            # --- Only evaluation and logging below this line ---
            avg_reward, avg_episode_length = evaluate_checkpoint(
                wrapped_env,
                agent,
                checkpoint_path,
                task,
                num_eval_episodes,
                rl_device
            )
            # Extract epoch/step from checkpoint_path
            checkpoint_name = os.path.basename(checkpoint_path)
            epoch = None
            global_step = None
            match = re.search(r'ep_(\d+)', checkpoint_name)
            if match:
                epoch = int(match.group(1))
                global_step = epoch * steps_per_epoch
            print(f"[RESULT] Dropout {prob}: Checkpoint: {checkpoint_name}")
            print(f"[RESULT] Dropout {prob}: Avg Reward: {avg_reward:.2f}")
            print(f"[RESULT] Dropout {prob}: Avg Episode Length: {avg_episode_length:.1f}")
            if wandb is not None and avg_reward is not None:
                wandb.log({
                    f"eval/avg_reward_dropout_{prob}": avg_reward,
                    f"eval/avg_episode_length_dropout_{prob}": avg_episode_length,
                    "eval/checkpoint": checkpoint_name,
                    "eval/epoch": epoch,
                    "global_step": global_step
                })
        # Cleanup agent and runner only once per checkpoint
        del agent
        del runner
        import gc
        gc.collect()

    if wandb is not None:
        wandb.finish()
    if close_env:
        base_env.close()
    # simulation_app.close() is handled outside

def main():
    import argparse
    # CLI argument parsing and AppLauncher initialization only happen here
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a folder using config.yaml.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task to evaluate (must match key in config.yaml)")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Load config from baselines_isaac/config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r') as f:
        all_config = yaml.safe_load(f)

    task_cfg = all_config.get(args_cli.task, {})
    eval_cfg = task_cfg.get('eval', {})
    checkpoint_folder = eval_cfg.get('checkpoint_folder', None)
    num_eval_episodes = eval_cfg.get('num_eval_episodes', 100)
    num_envs = eval_cfg.get('num_envs', 100)

    if checkpoint_folder is None:
        raise ValueError(f"checkpoint_folder must be specified for task {args_cli.task} in config.yaml under eval")

    evaluate_all_checkpoints(args_cli.task, checkpoint_folder, num_eval_episodes, num_envs)

if __name__ == "__main__":
    main() 