# ppo_agent.py

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import re


@dataclass
class Args:
    # Default args (not really used in fixed version yet)
    exp_name: str = "ppo_agent"
    seed: int = 1
    cuda: bool = True
    track: bool = False
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    clip_vloss: bool = True
    norm_adv: bool = True


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config

        # Prepare regex patterns
        self.mlp_keys_pattern = re.compile(config['full_keys']['mlp_keys'])
        self.cnn_keys_pattern = re.compile(config['full_keys']['cnn_keys'])

        # Determine observation input sizes
        self.mlp_obs_size = 0
        self.cnn_obs_shapes = {}
        for key, space in envs.obs_space.items():
            if self.mlp_keys_pattern.match(key):
                self.mlp_obs_size += int(np.prod(space.shape))
            if self.cnn_keys_pattern.match(key):
                self.cnn_obs_shapes[key] = space.shape

        # Action space
        self.action_space = envs.act_space["action"]
        self.is_discrete = self.action_space.discrete
        self.is_continuous = not self.is_discrete

        if self.is_discrete:
            self.num_actions = self.action_space.shape[0]
        else:
            self.num_actions = int(np.prod(self.action_space.shape))

        # Actor and Critic networks
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.mlp_obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.num_actions), std=0.01),
        )

        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(self.num_actions))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.mlp_obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _preprocess(self, obs_dict):
        out = {}
        for key, value in obs_dict.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
            else:
                tensor = value.float()
            if len(tensor.shape) >= 3 and tensor.dtype == torch.uint8:
                tensor = tensor / 255.0
            out[key] = tensor
        if 'is_terminal' in obs_dict:
            out['cont'] = 1.0 - obs_dict['is_terminal'].float()
        return out

    def _flatten_features(self, obs):
        parts = []
        for key, value in obs.items():
            if self.mlp_keys_pattern.match(key):
                parts.append(value.view(value.shape[0], -1))
        if not parts:
            raise ValueError("No observation keys matched MLP regex!")
        return torch.cat(parts, dim=-1)

    def get_value(self, obs_dict):
        obs = self._preprocess(obs_dict)
        mlp_input = self._flatten_features(obs)
        return self.critic(mlp_input)

    def get_action_and_value(self, obs_dict, action=None):
        obs = self._preprocess(obs_dict)
        mlp_input = self._flatten_features(obs)

        if self.is_discrete:
            logits = self.actor_mean(mlp_input)
            dist = Categorical(logits=logits)
        else:
            mean = self.actor_mean(mlp_input)
            std = torch.exp(self.actor_logstd)
            dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        if self.is_continuous:
            logprob = logprob.sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1)
        else:
            entropy = dist.entropy()

        value = self.critic(mlp_input)
        return action, logprob, entropy, value
