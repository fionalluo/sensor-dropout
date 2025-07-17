#!/usr/bin/env python3
"""
Evaluation utilities for SB3-based training scripts.
Contains custom evaluation callbacks and related functions.
"""

# ObservationFilterWrapper, VecObservationFilterWrapper, and CustomEvalCallback moved from baselines/ppo/train.py
import re
import numpy as np
import gymnasium as gym
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from baselines.shared.masking_utils import mask_observations_for_student


class ObservationFilterWrapper(gym.ObservationWrapper):
    """Wrapper to filter observations based on mlp_keys and cnn_keys patterns."""
    def __init__(self, env, mlp_keys: str = ".*", cnn_keys: str = ".*"):
        super().__init__(env)
        self.mlp_pattern = re.compile(mlp_keys)
        self.cnn_pattern = re.compile(cnn_keys)
        self._filter_observation_space()
    def _filter_observation_space(self):
        if hasattr(self.env, 'observation_space'):
            original_spaces = self.env.observation_space.spaces
        else:
            original_spaces = self.env.observation_space.spaces
        filtered_spaces = {}
        for key, space in original_spaces.items():
            is_image = len(space.shape) == 3 and space.shape[-1] == 3
            if is_image:
                if self.cnn_pattern.search(key):
                    filtered_spaces[key] = space
                else:
                    pass
            else:
                if self.mlp_pattern.search(key):
                    filtered_spaces[key] = space
                else:
                    pass
        self.observation_space = gym.spaces.Dict(filtered_spaces)
    def observation(self, obs):
        filtered_obs = {}
        for key, value in obs.items():
            if key in self.observation_space.spaces:
                filtered_obs[key] = value
        return filtered_obs
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "terminal_observation" in info:
            info["terminal_observation"] = self.observation(info["terminal_observation"])
        return self.observation(obs), reward, terminated, truncated, info
    def get_wrapper_attr(self, name):
        return getattr(self.env, name)

class VecObservationFilterWrapper:
    """Vectorized wrapper to filter observations based on mlp_keys and cnn_keys patterns.
    This works with VecEnv environments like IsaacVecEnvWrapper."""
    def __init__(self, env, mlp_keys: str = ".*", cnn_keys: str = ".*"):
        self.env = env
        self.mlp_pattern = re.compile(mlp_keys)
        self.cnn_pattern = re.compile(cnn_keys)
        self._filter_observation_space()
        self.num_envs = env.num_envs
        self.action_space = env.action_space
    def _filter_observation_space(self):
        if hasattr(self.env, 'observation_space'):
            original_spaces = self.env.observation_space.spaces
        else:
            original_spaces = self.env.observation_space.spaces
        filtered_spaces = {}
        for key, space in original_spaces.items():
            is_image = len(space.shape) == 3
            if is_image:
                if self.cnn_pattern.search(key):
                    filtered_spaces[key] = space
                else:
                    pass
            else:
                if self.mlp_pattern.search(key):
                    filtered_spaces[key] = space
                else:
                    pass
        self.observation_space = gym.spaces.Dict(filtered_spaces)
    def reset(self):
        obs = self.env.reset()
        return self._filter_obs(obs)
    def step_async(self, actions):
        return self.env.step_async(actions)
    def step_wait(self):
        obs, rewards, dones, infos = self.env.step_wait()
        filtered_obs = self._filter_obs(obs)
        for i, info in enumerate(infos):
            if "terminal_observation" in info:
                info["terminal_observation"] = self._filter_obs(info["terminal_observation"])
        return filtered_obs, rewards, dones, infos
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    def _filter_obs(self, obs):
        if isinstance(obs, dict):
            filtered_obs = {}
            for key, value in obs.items():
                if key in self.observation_space.spaces:
                    filtered_obs[key] = value
            return filtered_obs
        else:
            return obs
    def close(self):
        return self.env.close()
    def get_attr(self, attr_name, indices=None):
        return self.env.get_attr(attr_name, indices)
    def set_attr(self, attr_name, value, indices=None):
        return self.env.set_attr(attr_name, value, indices)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.env.env_method(method_name, *method_args, indices=indices, **method_kwargs)
    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.env.env_is_wrapped(wrapper_class, indices)
    def get_images(self):
        return self.env.get_images()
    def seed(self, seed=None):
        return self.env.seed(seed)


class CustomEvalCallback(BaseCallback):
    """Custom evaluation callback that evaluates across different observation subsets."""
    def __init__(
        self,
        eval_env,
        config,
        make_eval_env_func,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        debug=False,
        base_env_for_keys=None
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        self.make_eval_env_func = make_eval_env_func
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.debug = debug
        self.last_eval_step = -1  # Track last evaluation step to handle irregular increments
        self.base_env_for_keys = base_env_for_keys
        self.isaacgym_suite = config.task.split('_', 1)[0] == "isaacgym"
        # Get the number of eval configs
        self.num_eval_configs = getattr(config, 'num_eval_configs', 4)
        # Parse student keys from the agent's training keys
        if hasattr(config, 'keys') and config.keys:
            def _get_filtered_keys():
                if self.base_env_for_keys is not None:
                    env = self.base_env_for_keys
                    if hasattr(self.config, 'keys') and self.config.keys:
                        mlp_keys = getattr(self.config.keys, 'mlp_keys', '.*')
                        cnn_keys = getattr(self.config.keys, 'cnn_keys', '.*')
                        # Only wrap if not already filtered
                        if not isinstance(env, (VecObservationFilterWrapper, ObservationFilterWrapper)):
                            env = VecObservationFilterWrapper(env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
                else:
                    suite, task = config.task.split('_', 1)
                    env = gym.make(task)
                    if hasattr(config, 'keys') and config.keys:
                        mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
                        cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
                        env = ObservationFilterWrapper(
                            env, 
                            mlp_keys=mlp_keys,
                            cnn_keys=cnn_keys
                        )
                    obs, _ = env.reset()
                    env.close()
                    return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
            training_keys = _get_filtered_keys()
            self.student_keys = training_keys
            if self.debug:
                print(f"[EVAL CALLBACK] Training keys (student keys): {self.student_keys}")
        else:
            self.student_keys = []

    def _get_eval_env(self, filtered=True):
        if self.isaacgym_suite:
            env = self.base_env_for_keys
            if filtered:
                mlp_keys = getattr(self.config.keys, 'mlp_keys', '.*') if hasattr(self.config, 'keys') and self.config.keys else '.*'
                cnn_keys = getattr(self.config.keys, 'cnn_keys', '.*') if hasattr(self.config, 'keys') and self.config.keys else '.*'
                if not isinstance(env, VecObservationFilterWrapper):
                    env = VecObservationFilterWrapper(env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
            return env
        else:
            suite, task = self.config.task.split('_', 1)
            env = gym.make(task)
            if filtered and hasattr(self.config, 'keys') and self.config.keys:
                mlp_keys = getattr(self.config.keys, 'mlp_keys', '.*')
                cnn_keys = getattr(self.config.keys, 'cnn_keys', '.*')
                env = ObservationFilterWrapper(env, mlp_keys=mlp_keys, cnn_keys=cnn_keys)
            return env

    def _get_available_keys(self):
        reset_result = self.eval_env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result
        return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]

    def _parse_keys_from_pattern(self, pattern, available_keys):
        if pattern == '.*':
            return available_keys
        elif pattern == '^$':
            return []
        else:
            try:
                regex = re.compile(pattern)
                matched_keys = [k for k in available_keys if regex.search(k)]
                return matched_keys
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")
                return []

    def _on_step(self):
        if self.eval_freq > 0:
            current_step = self.num_timesteps
            if current_step == 0 and self.last_eval_step == -1:
                print(f"\nInitial evaluation at step {current_step}")
                self._run_comprehensive_evaluation()
                self.last_eval_step = current_step
            else:
                prev_eval_count = self.last_eval_step // self.eval_freq
                current_eval_count = current_step // self.eval_freq
                if current_eval_count > prev_eval_count:
                    next_eval_step = (prev_eval_count + 1) * self.eval_freq
                    print(f"\nEvaluation triggered: step {current_step} crossed boundary at {next_eval_step} (eval_freq={self.eval_freq})")
                    print(f"  Previous eval at step {self.last_eval_step}, current step {current_step}")
                    self._run_comprehensive_evaluation()
                    self.last_eval_step = current_step
        return True

    def _run_comprehensive_evaluation(self):
        if self.isaacgym_suite:
            print("[INFO] Skipping comprehensive evaluation for Isaac Gym environment.")
            return
        print(f"Starting comprehensive evaluation with {self.n_eval_episodes} episodes...")
        print(f"Number of eval configs: {self.num_eval_configs}")
        if not self.isaacgym_suite:
            self._evaluate_student_default()
        if not self.isaacgym_suite:
            env_metrics = {}
            for i in range(1, self.num_eval_configs + 1):
                env_name = f'env{i}'
                if hasattr(self.config.eval_keys, env_name):
                    eval_keys = getattr(self.config.eval_keys, env_name)
                    metrics = self._evaluate_environment(env_name, eval_keys)
                    env_metrics[env_name] = metrics
                else:
                    print(f"Warning: Missing eval_keys for {env_name}")
            self._log_mean_metrics(env_metrics)
        print("Evaluation complete!")

    def _evaluate_student_default(self):
        """Evaluate the student agent on a default environment configuration."""
        eval_env = self._get_eval_env()
        episode_returns = []
        episode_lengths = []
        for episode in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_return = 0
            episode_length = 0
            done = False
            truncated = False
            while not (done or truncated):
                # Convert observations to tensors for masking
                obs_tensors = {}
                for key, value in obs.items():
                    if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                        if isinstance(value, np.ndarray):
                            obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                        else:
                            obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
                
                # Mask observations for the student
                masked_obs = mask_observations_for_student(
                    obs_tensors, 
                    self.student_keys, 
                    self.student_keys, # Use student_keys for student evaluation
                    device=None,  # Use CPU for evaluation
                    debug=self.debug and episode == 0
                )
                
                # Debug logging for first episode
                if self.debug and episode == 0:
                    print(f"[EVAL DEBUG] Default env - Original obs keys: {list(obs_tensors.keys())}")
                    print(f"[EVAL DEBUG] Default env - Student keys: {self.student_keys}")
                    print(f"[EVAL DEBUG] Default env - Masked obs keys: {list(masked_obs.keys())}")
                
                # Convert masked observations to numpy for SB3
                masked_obs_numpy = {}
                for key, value in masked_obs.items():
                    if isinstance(value, torch.Tensor):
                        masked_obs_numpy[key] = value.cpu().numpy()
                    else:
                        masked_obs_numpy[key] = np.array(value)
                
                # Fix: Add batch dimension for SB3's MultiInputPolicy
                obs_batch = {k: np.expand_dims(v, axis=0) for k, v in masked_obs_numpy.items()}
                
                # Get action from student policy
                with torch.no_grad():
                    action, _ = self.model.predict(obs_batch, deterministic=self.deterministic)
                
                # Extract scalar action if it's a numpy array
                if isinstance(action, np.ndarray):
                    action = action.item() if action.size == 1 else action[0]
                
                # Step environment
                obs, reward, done, truncated, info = eval_env.step(action)
                
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        # Compute student metrics
        episode_returns = np.array(episode_returns)
        episode_lengths = np.array(episode_lengths)
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"  Student Default: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
        
        # Log to wandb if available
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record(f"student_eval_return/default/mean_return", mean_return)
            self.logger.record(f"student_eval/default/std_return", std_return)
            self.logger.record(f"student_eval/default/mean_length", mean_length)
            self.logger.record(f"student_eval/default/std_length", std_length)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                f"student_eval_return/default/mean_return": mean_return,
                f"student_eval/default/std_return": std_return,
                f"student_eval/default/mean_length": mean_length,
                f"student_eval/default/std_length": std_length
            })
        
        eval_env.close()

    def _evaluate_environment(self, env_name, eval_keys):
        """Evaluate the student agent on a specific environment configuration."""
        eval_env = self._get_eval_env()
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_return = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Convert observations to tensors for masking
                obs_tensors = {}
                for key, value in obs.items():
                    if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                        if isinstance(value, np.ndarray):
                            obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                        else:
                            obs_tensors[key] = torch.tensor([value], dtype=torch.float32)
                
                # Parse eval_keys patterns to get teacher_keys (list of keys)
                available_keys = list(obs.keys())
                mlp_keys_pattern = getattr(eval_keys, 'mlp_keys', '.*')
                cnn_keys_pattern = getattr(eval_keys, 'cnn_keys', '.*')
                teacher_mlp_keys = self._parse_keys_from_pattern(mlp_keys_pattern, available_keys)
                teacher_cnn_keys = self._parse_keys_from_pattern(cnn_keys_pattern, available_keys)
                teacher_keys = teacher_mlp_keys + teacher_cnn_keys
                # Mask observations for the student
                masked_obs = mask_observations_for_student(
                    obs_tensors, 
                    self.student_keys, 
                    teacher_keys, # Use parsed teacher_keys (list)
                    device=None,  # Use CPU for evaluation
                    debug=self.debug and episode == 0
                )
                
                # Debug logging for first episode
                if self.debug and episode == 0:
                    print(f"[EVAL DEBUG] {env_name} - Original obs keys: {list(obs_tensors.keys())}")
                    print(f"[EVAL DEBUG] {env_name} - Student keys: {self.student_keys}")
                    print(f"[EVAL DEBUG] {env_name} - Teacher keys: {eval_keys}")
                    print(f"[EVAL DEBUG] {env_name} - Masked obs keys: {list(masked_obs.keys())}")
                
                # Convert masked observations to numpy for SB3
                masked_obs_numpy = {}
                for key, value in masked_obs.items():
                    if isinstance(value, torch.Tensor):
                        masked_obs_numpy[key] = value.cpu().numpy()
                    else:
                        masked_obs_numpy[key] = np.array(value)
                
                # Fix: Add batch dimension for SB3's MultiInputPolicy
                obs_batch = {k: np.expand_dims(v, axis=0) for k, v in masked_obs_numpy.items()}
                
                # Get action from student policy
                with torch.no_grad():
                    action, _ = self.model.predict(obs_batch, deterministic=self.deterministic)
                
                # Extract scalar action if it's a numpy array
                if isinstance(action, np.ndarray):
                    action = action.item() if action.size == 1 else action[0]
                
                # Step environment
                obs, reward, done, truncated, info = eval_env.step(action)
                
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        # Compute student metrics
        episode_returns = np.array(episode_returns)
        episode_lengths = np.array(episode_lengths)
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"  Student {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
        
        # Log to wandb if available
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record(f"student_eval_return/{env_name}/mean_return", mean_return)
            self.logger.record(f"student_eval/{env_name}/std_return", std_return)
            self.logger.record(f"student_eval/{env_name}/mean_length", mean_length)
            self.logger.record(f"student_eval/{env_name}/std_length", std_length)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                f"student_eval_return/{env_name}/mean_return": mean_return,
                f"student_eval/{env_name}/std_return": std_return,
                f"student_eval/{env_name}/mean_length": mean_length,
                f"student_eval/{env_name}/std_length": std_length
            })
        
        eval_env.close()
        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "mean_length": mean_length,
            "std_length": std_length
        }

    def _log_mean_metrics(self, env_metrics):
        """Log mean metrics across all environments for comprehensive evaluation."""
        if not env_metrics:
            print("No environments to log comprehensive metrics.")
            return

        all_returns = [metrics["mean_return"] for metrics in env_metrics.values()]
        all_lengths = [metrics["mean_length"] for metrics in env_metrics.values()]

        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        mean_length = np.mean(all_lengths)
        std_length = np.std(all_lengths)

        print(f"\nComprehensive Evaluation Metrics:")
        print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"  Mean Length: {mean_length:.1f} ± {std_length:.1f}")

        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record("comprehensive_eval/mean_return", mean_return)
            self.logger.record("comprehensive_eval/std_return", std_return)
            self.logger.record("comprehensive_eval/mean_length", mean_length)
            self.logger.record("comprehensive_eval/std_length", std_length)

        if wandb.run is not None:
            wandb.log({
                "comprehensive_eval/mean_return": mean_return,
                "comprehensive_eval/std_return": std_return,
                "comprehensive_eval/mean_length": mean_length,
                "comprehensive_eval/std_length": std_length
            })
