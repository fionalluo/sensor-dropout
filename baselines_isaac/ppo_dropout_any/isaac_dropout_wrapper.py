import gym
import torch
import yaml
import os
from types import SimpleNamespace

class DropoutScheduler:
    """Handles dropout probability scheduling (constant only)."""
    def __init__(self, schedule_cfg):
        self.schedule_type = schedule_cfg.get('schedule_type', 'constant')
        if self.schedule_type == 'constant':
            self.base_prob = float(schedule_cfg.get('base_probability', 0.0))
        else:
            raise ValueError(f"Only 'constant' schedule_type is supported, got: {self.schedule_type}")

    def get_prob(self, episode):
        return self.base_prob


def load_task_dropout_config(config_path, task_name):
    """Load the YAML config and return the key->indices mapping for the given task."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if task_name not in config:
        raise ValueError(f"Task {task_name} not found in config file {config_path}")
    task_cfg = config[task_name]
    key_indices = task_cfg['keys']
    return key_indices


import gym
import torch

class IsaacProbabilisticDropoutWrapper(gym.Wrapper):
    """
    Efficient dropout wrapper for IsaacLab: keeps dropout mask fixed during episodes,
    re-samples on reset or when an env is done. Fully vectorized for speed.
    """
    def __init__(self, env, key_indices, dropout_scheduler, seed=None):
        super().__init__(env)
        self.key_indices = key_indices
        self.keys = list(key_indices.keys())
        self.dropout_scheduler = dropout_scheduler
        self.rng = torch.Generator(device="cpu")  # Generator remains on CPU
        if seed is not None:
            self.rng.manual_seed(seed)

        self.current_masks = None

    def _generate_masks(self, num_envs, device):
        """Generate a dropout mask for each env in the batch."""
        prob = self.dropout_scheduler.get_prob(0)
        # Generate on CPU, then move to target device
        masks = {}
        for k in self.keys:
            drop_flags = (torch.rand(num_envs, generator=self.rng) >= prob).float().unsqueeze(1).to(device)
            masks[k] = drop_flags  # shape: [N, 1]
        return masks

    def _mask_obs(self, obs):
        if not isinstance(obs, dict) or "policy" not in obs:
            return obs

        policy_obs = obs["policy"]
        if self.current_masks is None:
            return obs

        for k, (start, end) in self.key_indices.items():
            if k not in self.current_masks:
                continue
            mask = self.current_masks[k]  # shape: [N, 1]
            policy_obs[:, start:end] *= mask

        return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})

        if isinstance(obs, dict) and "policy" in obs:
            num_envs = obs["policy"].shape[0]
            device = obs["policy"].device
        else:
            num_envs = obs.shape[0] if torch.is_tensor(obs) and obs.ndim > 1 else 1
            device = obs.device if torch.is_tensor(obs) else torch.device("cpu")

        self.current_masks = self._generate_masks(num_envs, device)
        return (self._mask_obs(obs), info) if isinstance(result, tuple) else self._mask_obs(obs)

    def step(self, action):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result if len(result) == 5 else (*result, None)
        done_flags = torch.logical_or(torch.as_tensor(terminated), torch.as_tensor(truncated)) if truncated is not None else torch.as_tensor(terminated)

        if self.current_masks is not None:
            done_indices = torch.nonzero(done_flags).squeeze(1)
            for k in self.current_masks:
                prob = self.dropout_scheduler.get_prob(0)
                mask = self.current_masks[k]
                new_flags = (torch.rand(len(done_indices), generator=self.rng) >= prob).float().unsqueeze(1).to(mask.device)
                mask[done_indices] = new_flags

        return (self._mask_obs(obs), reward, terminated, truncated, info) if truncated is not None else (self._mask_obs(obs), reward, terminated, info)
