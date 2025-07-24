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
    
    NOTE: The reason there is a lot of conditional logic around the observation lengths and return types, 
    is because we designed the wrapper to be able to wrap around the BASE environment but also the 
    VECTORIZED environment, which have different observation lengths and return types. 
    
    This logic can be simplified in the future -- if we use a wrapper around every environment
    and dynamically change the wrapper properties instead! 
    """
    def __init__(self, env, task_name, seed=None, dropout_prob=None):
        super().__init__(env)
        # Fixed config path: always relative to this file
        config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
        config_path = os.path.abspath(config_path)
        # Load key indices and dropout config for the task
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if task_name not in config:
            raise ValueError(f"Task {task_name} not found in config file {config_path}")
        task_cfg = config[task_name]
        key_indices = task_cfg['keys']
        # If dropout_prob is provided, override config
        if dropout_prob is not None:
            dropout_cfg = {'schedule_type': 'constant', 'base_probability': dropout_prob}
            print("During initialization, dropout probability set to", dropout_prob)
        else:
            dropout_cfg = task_cfg.get('dropout', {'schedule_type': 'constant', 'base_probability': 0.0})
            print("Dropout probability set to", dropout_cfg['base_probability'])
        self.key_indices = key_indices
        self.keys = list(key_indices.keys())
        self.dropout_scheduler = DropoutScheduler(dropout_cfg)
        self.rng = torch.Generator(device="cpu")  # Generator remains on CPU
        if seed is not None:
            self.rng.manual_seed(seed)

    def _generate_masks(self, num_envs, device):
        """Generate a dropout mask for each key, for each env in the batch. Each env gets an independent mask for each key."""
        prob = self.dropout_scheduler.get_prob(0)
        masks = {}
        # Create a generator on the correct device for torch.rand
        rng = torch.Generator(device=device)
        # Optionally, seed the generator for reproducibility if self.rng has a seed
        if hasattr(self.rng, 'initial_seed'):
            rng.manual_seed(self.rng.initial_seed())
        for k in self.keys:
            keep_flags = (torch.rand(num_envs, generator=rng, device=device) >= prob).float().unsqueeze(1)
            masks[k] = keep_flags  # shape: [N, 1]
        return masks

    def _mask_obs(self, obs):
        # If current_masks is None, generate a new mask based on the observation
        if self.current_masks is None:
            # Determine num_envs and device for mask generation
            if isinstance(obs, dict) and "policy" in obs:
                num_envs = obs["policy"].shape[0]
                device = obs["policy"].device
            else:
                num_envs = obs.shape[0] if torch.is_tensor(obs) and obs.ndim > 1 else 1
                device = obs.device if torch.is_tensor(obs) else torch.device("cpu")
            self.current_masks = self._generate_masks(num_envs, device)
            # Comment: Always generate a mask if missing, never return obs unmasked

        # If obs is a dict with 'policy', mask as before
        if isinstance(obs, dict) and "policy" in obs:
            policy_obs = obs["policy"]
            for k, (start, end) in self.key_indices.items():
                if k not in self.current_masks:
                    continue
                mask = self.current_masks[k]  # shape: [N, 1]
                policy_obs[:, start:end] *= mask
            return obs

        # If obs is a tensor, mask it directly
        elif torch.is_tensor(obs):
            masked_obs = obs.clone()
            for k, (start, end) in self.key_indices.items():
                if k not in self.current_masks:
                    continue
                mask = self.current_masks[k]  # shape: [N, 1]
                masked_obs[:, start:end] *= mask
            return masked_obs

        return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        # Determine num_envs and device for mask generation
        if isinstance(obs, dict) and "policy" in obs:
            num_envs = obs["policy"].shape[0]
            device = obs["policy"].device
        else:
            num_envs = obs.shape[0] if torch.is_tensor(obs) and obs.ndim > 1 else 1
            device = obs.device if torch.is_tensor(obs) else torch.device("cpu")
        self.current_masks = self._generate_masks(num_envs, device)
        if isinstance(result, tuple):
            masked_obs = self._mask_obs(result[0])
            # Ensure masked_obs is a dict with 'policy' key
            if not (isinstance(masked_obs, dict) and "policy" in masked_obs):
                masked_obs = {"policy": masked_obs}
            return (masked_obs, result[1])
        else:
            masked_obs = self._mask_obs(result)
            if not (isinstance(masked_obs, dict) and "policy" in masked_obs):
                masked_obs = {"policy": masked_obs}
            return masked_obs, {}

    def step(self, action):
        result = self.env.step(action)
        # print("STEP results")
        # for res in result:
        #     print(res)
        if isinstance(result, tuple):
            masked_obs = self._mask_obs(result[0])
            if not (isinstance(masked_obs, dict) and "policy" in masked_obs):
                masked_obs = {"policy": masked_obs}
            # Regenerate mask for done episodes if possible
            if len(result) >= 3:
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
                if self.current_masks is not None and len(done_indices) > 0:
                    for k in self.current_masks:
                        prob = self.dropout_scheduler.get_prob(0)
                        mask = self.current_masks[k]
                        rng = torch.Generator(device=mask.device)
                        if hasattr(self.rng, 'initial_seed'):
                            rng.manual_seed(self.rng.initial_seed())
                        new_flags = (torch.rand(len(done_indices), generator=rng, device=mask.device) >= prob).float().unsqueeze(1)
                        mask[done_indices] = new_flags
            # Always return 5 values: (obs, reward, terminated, truncated, info)
            if len(result) == 5:
                return (masked_obs,) + result[1:]
            elif len(result) == 4:
                obs, reward, terminated, info = result
                # Try to extract 'time_outs' from info, else use all False
                if isinstance(info, dict) and "time_outs" in info:
                    truncated = info["time_outs"]
                else:
                    if isinstance(terminated, torch.Tensor):
                        truncated = torch.zeros_like(terminated, dtype=torch.bool)
                    else:
                        truncated = [False] * len(terminated)
                return masked_obs, reward, terminated, truncated, info
            else:
                raise RuntimeError(f"Unexpected number of items returned from env.step: {len(result)}")
        else:
            masked_obs = self._mask_obs(result)
            if not (isinstance(masked_obs, dict) and "policy" in masked_obs):
                masked_obs = {"policy": masked_obs}
            # Not expected, but fallback to 5-item return
            return masked_obs, None, None, None, {}

    def set_dropout_probability(self, prob):
        """Set a new dropout probability for this wrapper."""
        if hasattr(self.dropout_scheduler, 'base_prob'):
            self.dropout_scheduler.base_prob = float(prob)
        else:
            # If the scheduler is not constant, this will error as intended
            raise ValueError("DropoutScheduler does not support changing probability dynamically.")
