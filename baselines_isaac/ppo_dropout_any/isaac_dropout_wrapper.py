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
    Efficient dropout wrapper for IsaacLab: keeps dropout mask fixed during episodes,
    re-samples on reset or when an env is done. Fully vectorized for speed.
    For each key, all indices for that key are zeroed together per environment, with probability given by the dropout scheduler.
    """
    def __init__(self, env, task_name, seed=None, dropout_prob=None):
        super().__init__(env)
        config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
        config_path = os.path.abspath(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if task_name not in config:
            raise ValueError(f"Task {task_name} not found in config file {config_path}")
        task_cfg = config[task_name]
        key_indices = task_cfg['keys']
        # Force dropout probability to 0.25 for all keys
        dropout_cfg = {'schedule_type': 'constant', 'base_probability': 0.25}
        print("[DropoutWrapper] Dropout probability forcibly set to 0.25 for all keys.")
        self.key_indices = key_indices
        self.keys = list(key_indices.keys())
        self.dropout_scheduler = DropoutScheduler(dropout_cfg)
        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(seed)
        self.current_masks = None

    def _generate_masks(self, num_envs, device):
        prob = self.dropout_scheduler.get_prob(0)
        masks = {}
        for k in self.keys:
            # Simple, robust mask generation
            keep_flags = (torch.rand(num_envs, device=device) >= prob).float().unsqueeze(1)
            masks[k] = keep_flags  # shape: [N, 1]
        return masks

    def _mask_obs(self, obs):
        # Only mask if self.current_masks is set
        if self.current_masks is None:
            # Try to infer batch size/device from obs
            if isinstance(obs, dict) and len(obs) > 0:
                first = next(iter(obs.values()))
                num_envs = first.shape[0]
                device = first.device
            elif torch.is_tensor(obs):
                num_envs = obs.shape[0] if obs.ndim > 1 else 1
                device = obs.device
            else:
                return obs
            self.current_masks = self._generate_masks(num_envs, device)

        # Remove debug: print obs before masking (first env only)
        # print("[DropoutWrapper] Before masking (env 0):")
        # if isinstance(obs, dict):
        #     for k, v in obs.items():
        #         if hasattr(v, '__getitem__'):
        #             print(f"  {k}: {v[0]}")
        #         else:
        #             print(f"  {k}: {v}")
        # elif torch.is_tensor(obs):
        #     print(obs[0] if obs.ndim > 1 else obs)
        # else:
        #     print(obs)

        zeroed_keys = []
        if isinstance(obs, dict):
            obs = obs.copy()
            for k, (start, end) in self.key_indices.items():
                if k not in self.current_masks:
                    continue
                if k in obs:
                    # before = obs[k][0].clone() if hasattr(obs[k], 'clone') else obs[k]
                    obs[k][:, ...] *= self.current_masks[k]
                    # after = obs[k][0] if hasattr(obs[k], '__getitem__') else obs[k]
                    # Only print if env 0 is zeroed
                    if self.current_masks[k][0] == 0:
                        zeroed_keys.append(k)
                        # print(f"[DropoutWrapper] Key '{k}' zeroed for env 0. Before: {before}, After: {after}")
                elif 'policy' in obs:
                    # before = obs['policy'][0, start:end].clone()
                    obs['policy'][:, start:end] *= self.current_masks[k]
                    # after = obs['policy'][0, start:end]
                    if self.current_masks[k][0] == 0:
                        zeroed_keys.append(k)
                        # print(f"[DropoutWrapper] Key '{k}' (policy slice {start}:{end}) zeroed for env 0. Before: {before}, After: {after}")
            # if zeroed_keys:
            #     print(f"[DropoutWrapper] Zeroed keys for env 0 this step: {zeroed_keys}")
            # print("[DropoutWrapper] After masking (env 0):")
            # for k, v in obs.items():
            #     if hasattr(v, '__getitem__'):
            #         print(f"  {k}: {v[0]}")
            #     else:
            #         print(f"  {k}: {v}")
            return obs
        elif torch.is_tensor(obs):
            masked_obs = obs.clone()
            for k, (start, end) in self.key_indices.items():
                if k not in self.current_masks:
                    continue
                # before = masked_obs[0, start:end].clone()
                masked_obs[:, start:end] *= self.current_masks[k]
                # after = masked_obs[0, start:end]
                if self.current_masks[k][0] == 0:
                    zeroed_keys.append(k)
                    # print(f"[DropoutWrapper] Key '{k}' (tensor slice {start}:{end}) zeroed for env 0. Before: {before}, After: {after}")
            # if zeroed_keys:
            #     print(f"[DropoutWrapper] Zeroed keys for env 0 this step: {zeroed_keys}")
            # print("[DropoutWrapper] After masking (env 0):")
            # print(masked_obs[0] if masked_obs.ndim > 1 else masked_obs)
            return masked_obs
        else:
            return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # result can be obs or (obs, info)
        if isinstance(result, tuple):
            obs, *rest = result
        else:
            obs = result
            rest = []
        # Determine num_envs/device for mask generation
        if isinstance(obs, dict) and len(obs) > 0:
            first = next(iter(obs.values()))
            num_envs = first.shape[0]
            device = first.device
        elif torch.is_tensor(obs):
            num_envs = obs.shape[0] if obs.ndim > 1 else 1
            device = obs.device
        else:
            num_envs = 1
            device = torch.device("cpu")
        self.current_masks = self._generate_masks(num_envs, device)
        masked_obs = self._mask_obs(obs)
        if rest:
            return (masked_obs, *rest)
        else:
            return masked_obs

    def _regenerate_masks_for_done(self, done_indices):
        """Regenerate dropout masks for environments that are done. Only print debug info for env 0."""
        if self.current_masks is not None and len(done_indices) > 0:
            done_indices_list = done_indices.tolist()
            for k in self.current_masks:
                prob = self.dropout_scheduler.get_prob(0)
                mask = self.current_masks[k]
                # Simple, robust mask regeneration
                new_flags = (torch.rand(len(done_indices), device=mask.device) >= prob).float().unsqueeze(1)
                mask[done_indices] = new_flags
                # Only print debug info if env 0 is among the done indices
                # if 0 in done_indices_list:
                #     idx0_pos = done_indices_list.index(0)
                #     flag0 = new_flags[idx0_pos].item()
                #     print(f"[DropoutWrapper] Regenerated mask for key '{k}' at env 0: mask={flag0}")

    def step(self, action):
        print("[DropoutWrapper] step() called, applying masking...")
        result = self.env.step(action)
        # result can be (obs, reward, terminated, truncated, info) or similar
        if isinstance(result, tuple) and len(result) >= 3:
            obs = result[0]
            masked_obs = self._mask_obs(obs)
            # Regenerate mask for done episodes if possible
            terminated = result[2]
            truncated = result[3] if len(result) >= 4 else None
            def extract_array(val):
                if isinstance(val, dict):
                    if 'policy' in val:
                        return val['policy']
                    return next(iter(val.values()))
                return val
            terminated_arr = extract_array(terminated)
            if truncated is not None:
                truncated_arr = extract_array(truncated)
                done_flags = torch.logical_or(torch.as_tensor(terminated_arr), torch.as_tensor(truncated_arr))
            else:
                done_flags = torch.as_tensor(terminated_arr)
            done_indices = torch.nonzero(done_flags).squeeze(1)
            # Use helper for mask regeneration and debug
            self._regenerate_masks_for_done(done_indices)
            # Return same structure as self.env
            return (masked_obs,) + result[1:]
        else:
            # Fallback: just mask first element if possible
            if isinstance(result, tuple) and len(result) > 0:
                obs = result[0]
                masked_obs = self._mask_obs(obs)
                return (masked_obs,) + result[1:]
            else:
                return self._mask_obs(result)

    def set_dropout_probability(self, prob):
        if hasattr(self.dropout_scheduler, 'base_prob'):
            self.dropout_scheduler.base_prob = float(prob)
        else:
            raise ValueError("DropoutScheduler does not support changing probability dynamically.")
