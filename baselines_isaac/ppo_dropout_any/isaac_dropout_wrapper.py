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


class IsaacProbabilisticDropoutWrapper(gym.Wrapper):
    """
    For IsaacLab envs: for each observation, each key has a probability (schedule) of being zeroed out.
    Uses config file for key->indices mapping. Works for both single and vectorized envs.
    """
    def __init__(self, env, key_indices, dropout_scheduler, seed=None):
        super().__init__(env)
        self.key_indices = key_indices  # dict: key -> [start, end]
        self.keys = list(key_indices.keys())
        self.dropout_scheduler = dropout_scheduler
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def _mask_obs(self, obs):
        prob = self.dropout_scheduler.get_prob(0)  # No step tracking, always use base prob
        # If obs is a dict with 'policy', mask that tensor
        if isinstance(obs, dict) and "policy" in obs:
            obs = obs.copy()
            policy_obs = obs["policy"].clone()
            for k in self.keys:
                rand_val = torch.rand(1, generator=self.rng).item()
                if rand_val < prob:
                    # print(f"[Dropout] Zeroing key: {k} (rand={rand_val:.3f} < prob={prob})")
                    start, end = self.key_indices[k]
                    policy_obs[..., start:end] = 0
            obs["policy"] = policy_obs
            return obs
        elif torch.is_tensor(obs):
            masked = obs.clone()
            for k in self.keys:
                rand_val = torch.rand(1, generator=self.rng).item()
                if rand_val < prob:
                    # print(f"[Dropout] Zeroing key: {k} (rand={rand_val:.3f} < prob={prob})")
                    start, end = self.key_indices[k]
                    masked[..., start:end] = 0
            return masked
        else:
            return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self._mask_obs(obs), info
        else:
            return self._mask_obs(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._mask_obs(obs), reward, terminated, truncated, info
        else:
            obs, reward, done, info = result
            return self._mask_obs(obs), reward, done, info 