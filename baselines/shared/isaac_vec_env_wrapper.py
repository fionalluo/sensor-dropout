import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

class IsaacVecEnvWrapper(VecEnv):
    """
    Wrapper for Isaac Gym environments to make them compatible with Stable Baselines3 VecEnv interface.

    Isaac Gym environments handle parallelism internally and return observations with
    the first dimension being the number of environments. This wrapper adapts them
    to the format expected by Stable Baselines3.
    """
    def __init__(self, env: gym.Env, device: str = "cuda:0"):
        self.env = env
        self.device = device
        if hasattr(env, 'num_envs'):
            self.num_envs = env.num_envs
        else:
            obs_space = env.observation_space
            if isinstance(obs_space, gym.spaces.Dict):
                first_key = list(obs_space.spaces.keys())[0]
                self.num_envs = obs_space.spaces[first_key].shape[0]
            else:
                self.num_envs = obs_space.shape[0]
        self._process_spaces()
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)
        self._first_episode_logged = False  # Track if we've logged the first episode
        super().__init__(self.num_envs, self.observation_space, self.action_space)

    def _process_spaces(self):
        # process observation space
        observation_space = self.env.observation_space
        if isinstance(observation_space, gym.spaces.Dict):
            processed_spaces = {}
            for key, space in observation_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    shape = space.shape
                    if shape[0] == self.num_envs:
                        # Remove batch dim for the space
                        shape = shape[1:]
                        low = space.low[0]
                        high = space.high[0]
                    else:
                        low = space.low
                        high = space.high
                    processed_spaces[key] = gym.spaces.Box(
                        low=low,
                        high=high,
                        shape=shape,
                        dtype=space.dtype
                    )
                else:
                    raise TypeError(f"Unsupported observation space type for key '{key}': {type(space)}")
            print("PROCESSED SPACES:")
            for k, v in processed_spaces.items():
                print(f"{k}: {type(v)} -> {v}")
            self.observation_space = gym.spaces.Dict(processed_spaces)
        # process action space
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Box):
            if len(action_space.shape) > 1 and action_space.shape[0] == self.num_envs:
                new_shape = action_space.shape[1:]
                self.action_space = gym.spaces.Box(
                    low=action_space.low[0] if action_space.low.ndim > 0 else action_space.low,
                    high=action_space.high[0] if action_space.high.ndim > 0 else action_space.high,
                    shape=new_shape,
                    dtype=action_space.dtype
                )
            else:
                self.action_space = action_space
        else:
            self.action_space = action_space

    def reset(self) -> VecEnvObs:
        if not self._first_episode_logged:
            print("[DEBUG] IsaacVecEnvWrapper.reset() called")
        obs = self.env.reset()
        # If obs is a tuple, extract the first element (the actual observation)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)
        processed_obs = self._process_obs(obs)
        if not self._first_episode_logged:
            print(f"[DEBUG] Reset: processed_obs type: {type(processed_obs)}, keys: {list(processed_obs.keys()) if isinstance(processed_obs, dict) else None}")
            if isinstance(processed_obs, dict):
                for k, v in processed_obs.items():
                    print(f"[DEBUG] Reset: obs[{k}] shape: {v.shape}, dtype: {v.dtype}")
        return processed_obs

    def step_async(self, actions: np.ndarray):
        if not self._first_episode_logged:
            print(f"[DEBUG] IsaacVecEnvWrapper.step_async() called with actions type: {type(actions)}, shape: {actions.shape if hasattr(actions, 'shape') else None}")
            # Log the first few actions for inspection
            if isinstance(actions, np.ndarray):
                for i in range(min(3, actions.shape[0])):
                    print(f"[DEBUG] step_async: action[{i}]: {actions[i]}")
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device=self.device, dtype=torch.float32)
        if not self._first_episode_logged:
            print(f"[DEBUG] step_async: actions converted to torch: {isinstance(actions, torch.Tensor)}, shape: {actions.shape if hasattr(actions, 'shape') else None}")
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        if not self._first_episode_logged:
            print("[DEBUG] IsaacVecEnvWrapper.step_wait() called")
        obs, rewards, terminated, truncated, info = self.env.step(self._async_actions)
        # Log reward and done for first 2 envs only during first episode
        if not self._first_episode_logged:
            if isinstance(rewards, (np.ndarray, torch.Tensor)):
                for i in range(min(2, len(rewards))):
                    print(f"[DEBUG] step_wait: reward[{i}]: {rewards[i]}")
            if isinstance(terminated, (np.ndarray, torch.Tensor)):
                for i in range(min(2, len(terminated))):
                    print(f"[DEBUG] step_wait: terminated[{i}]: {terminated[i]}")
            if isinstance(truncated, (np.ndarray, torch.Tensor)):
                for i in range(min(2, len(truncated))):
                    print(f"[DEBUG] step_wait: truncated[{i}]: {truncated[i]}")
        # If obs is a tuple, extract the first element (the actual observation)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = self._process_obs(obs)
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        elif isinstance(rewards, np.ndarray):
            rewards = rewards.copy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.detach().cpu().numpy()
        dones = terminated | truncated
        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        infos = self._process_info(info, obs, terminated, truncated, dones)
        done_indices = np.where(dones)[0]
        # Log episode return and length for first 2 envs that finish, but only during first episode
        for idx in done_indices:
            if idx < 2 and not self._first_episode_logged:
                print(f"[DEBUG] step_wait: Episode done for env {idx}: return={self._ep_rew_buf[idx]}, length={self._ep_len_buf[idx]}")
        # Mark first episode as logged if any episode ends
        if len(done_indices) > 0 and not self._first_episode_logged:
            self._first_episode_logged = True
            print("[DEBUG] First episode completed - stopping debug logs")
        self._ep_rew_buf[done_indices] = 0
        self._ep_len_buf[done_indices] = 0
        return obs, rewards, dones, infos

    def _process_obs(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(obs, dict):
            processed_obs = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = value
                # Ensure batch dim is present: shape should be (num_envs, ...)
                expected_shape = self.observation_space.spaces[key].shape
                # If shape is not (num_envs, ...) or (1, ...), raise error
                if arr.shape[1:] != expected_shape:
                    raise ValueError(f"IsaacVecEnvWrapper: For key '{key}', got obs shape {arr.shape}, expected (*, {expected_shape})")
                processed_obs[key] = arr
            # Debug print
            # print("[IsaacVecEnvWrapper] Processed obs shapes:", {k: v.shape for k, v in processed_obs.items()})
            return processed_obs
        elif isinstance(obs, torch.Tensor):
            arr = obs.detach().cpu().numpy()
            expected_shape = self.observation_space.shape
            if arr.shape[1:] != expected_shape:
                raise ValueError(f"IsaacVecEnvWrapper: Got obs shape {arr.shape}, expected (*, {expected_shape})")
            # print(f"[IsaacVecEnvWrapper] Processed obs shape: {arr.shape}")
            return arr
        elif isinstance(obs, np.ndarray):
            expected_shape = self.observation_space.shape
            if obs.shape[1:] != expected_shape:
                raise ValueError(f"IsaacVecEnvWrapper: Got obs shape {obs.shape}, expected (*, {expected_shape})")
            # print(f"[IsaacVecEnvWrapper] Processed obs shape: {obs.shape}")
            return obs
        else:
            return obs

    def _process_info(self, info: Dict[str, Any], obs: Any, terminated: np.ndarray, truncated: np.ndarray, dones: np.ndarray) -> List[Dict[str, Any]]:
        infos = [{} for _ in range(self.num_envs)]
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            infos[idx]["episode"] = {
                "r": float(self._ep_rew_buf[idx]),
                "l": int(self._ep_len_buf[idx])
            }
            if isinstance(obs, dict):
                terminal_obs = {key: value[idx] for key, value in obs.items()}
            else:
                terminal_obs = obs[idx]
            infos[idx]["terminal_observation"] = terminal_obs
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
        for key, value in info.items():
            if key not in ["episode", "terminal_observation", "TimeLimit.truncated"]:
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    for i in range(self.num_envs):
                        if isinstance(value, torch.Tensor):
                            infos[i][key] = value[i].detach().cpu().numpy()
                        else:
                            infos[i][key] = value[i]
                else:
                    for i in range(self.num_envs):
                        infos[i][key] = value
        return infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None):
        if indices is None:
            indices = list(range(self.num_envs))
        attr_value = getattr(self.env, attr_name)
        if isinstance(attr_value, torch.Tensor):
            return attr_value[indices].detach().cpu().numpy()
        else:
            return [attr_value] * len(indices)

    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        raise NotImplementedError("Setting attributes is not supported for Isaac Gym environments.")

    def env_method(self, method_name: str, *method_args, indices: Optional[List[int]] = None, **method_kwargs):
        if method_name == "render":
            return self.env.render()
        else:
            method = getattr(self.env, method_name)
            return method(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices: Optional[List[int]] = None):
        return [False] * self.num_envs

    def get_images(self):
        raise NotImplementedError("Getting images is not supported for Isaac Gym environments.")

    def seed(self, seed: Optional[int] = None):
        return [self.env.seed(seed)] * self.num_envs 