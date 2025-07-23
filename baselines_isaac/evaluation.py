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
parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a folder.")
parser.add_argument("--checkpoint_folder", type=str, required=True, help="Folder containing checkpoint files")
parser.add_argument("--task", type=str, required=True, help="Name of the task to evaluate")
parser.add_argument("--num_eval_episodes", type=int, default=100, help="Number of episodes to evaluate per checkpoint")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

def rollout_policy(env, agent, num_eval_episodes, rl_device):
    total_rewards = []
    episode_lengths = []
    total_episodes = 0
    num_envs = env.unwrapped.num_envs
    # Ensure env.reset() is outside any torch context
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    episode_rewards = torch.zeros(num_envs, device=rl_device)
    episode_lengths_env = torch.zeros(num_envs, device=rl_device)
    while total_episodes < num_eval_episodes:
        # Only use inference_mode/no_grad for agent forward, not for env.step or env.reset
        try:
            context = torch.inference_mode()
        except AttributeError:
            context = torch.no_grad()
        with context:
            obs_torch = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_torch, is_deterministic=getattr(agent, 'is_deterministic', True))
        # env.step is outside the context!
        result = env.step(actions)
        if len(result) == 5:
            obs, rewards, terminated, truncated, infos = result
            dones = torch.logical_or(torch.as_tensor(terminated), torch.as_tensor(truncated))
        else:
            obs, rewards, dones, infos = result
        episode_rewards += rewards
        episode_lengths_env += 1
        if torch.any(dones):
            done_indices = torch.nonzero(dones).squeeze(1)
            for idx in done_indices:
                total_rewards.append(episode_rewards[idx].item())
                episode_lengths.append(episode_lengths_env[idx].item())
                total_episodes += 1
                episode_rewards[idx] = 0.0
                episode_lengths_env[idx] = 0.0
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
    return total_rewards, episode_lengths

def evaluate_checkpoint(env, checkpoint_path, task, num_eval_episodes, rl_device):
    import copy
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    from rl_games.torch_runner import Runner
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(checkpoint_path)
    agent.reset()
    # Ensure env.reset() is outside any torch context
    obs = env.reset()
    total_rewards, episode_lengths = rollout_policy(env, agent, num_eval_episodes, rl_device)
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    avg_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
    print(f"[RESULT] Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"[RESULT] Avg Reward: {avg_reward:.2f}")
    print(f"[RESULT] Avg Episode Length: {avg_episode_length:.1f}")
    # Cleanup agent and runner only
    del agent
    del runner
    import gc
    gc.collect()

def main():
    # Create environment ONCE
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    env_cfg = parse_env_cfg(
        args_cli.task, device=agent_cfg["params"]["config"]["device"], num_envs=args_cli.num_envs, use_fabric=True
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # Find all checkpoint files matching regex ep_\d+
    checkpoint_pattern = os.path.join(args_cli.checkpoint_folder, "*.pth")
    all_checkpoint_files = glob.glob(checkpoint_pattern)
    import re
    checkpoint_files = [f for f in all_checkpoint_files if re.search(r'ep_\d+', os.path.basename(f))]
    checkpoint_files = sorted(checkpoint_files, key=os.path.basename)
    if not checkpoint_files:
        print(f"[WARNING] No checkpoint files found in {args_cli.checkpoint_folder}")
        return
    print(f"[INFO] Found {len(checkpoint_files)} checkpoint files to evaluate")
    for checkpoint_path in checkpoint_files:
        evaluate_checkpoint(
            env,
            checkpoint_path,
            args_cli.task,
            args_cli.num_eval_episodes,
            rl_device
        )
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main() 