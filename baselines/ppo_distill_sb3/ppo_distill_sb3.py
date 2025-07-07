#!/usr/bin/env python3
"""
Pure Distillation Training using SB3
Trains a student policy purely through imitation from expert policies, no RL components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import time
from collections import deque
import torch.optim as optim
import wandb
import os
import re
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from baselines.shared.masking_utils import mask_observations_for_student
from subset_policies_sb3.load_subset_policy_sb3 import SubsetPolicyLoader


class ExpertPolicyManager:
    """Manages loading and using expert policies for distillation."""
    
    def __init__(self, policy_dir: str, device: str = 'cpu'):
        """
        Initialize the expert policy manager.
        
        Args:
            policy_dir: Directory containing expert policies
            device: Device to load policies on
        """
        self.policy_dir = policy_dir
        self.device = device
        self.expert_policies = {}
        self.expert_eval_keys = {}
        
        # Load all expert policies using SB3 loader
        self._load_expert_policies()
    
    def _load_expert_policies(self):
        """Load all expert policies from the directory using SB3 loader."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Expert policy directory not found: {self.policy_dir}")
        
        # Use SB3 subset policy loader
        loader = SubsetPolicyLoader(self.policy_dir, device=self.device)
        
        # Store the loaded policies
        for subset_name in loader.policies.keys():
            agent, config, eval_keys = loader.load_policy(subset_name)
            self.expert_policies[subset_name] = agent
            self.expert_eval_keys[subset_name] = eval_keys
            
            print(f"Loaded expert policy: {subset_name}")
        
        print(f"Loaded {len(self.expert_policies)} expert policies")
    
    def get_expert_action_logits(self, subset_name: str, obs: Dict) -> torch.Tensor:
        """
        Get action logits from a specific expert policy for distillation.
        """
        if subset_name not in self.expert_policies:
            raise ValueError(f"Expert policy {subset_name} not found")
        
        expert_agent = self.expert_policies[subset_name]
        eval_keys = self.expert_eval_keys[subset_name]

        # Filter observations based on expert's eval_keys patterns
        filtered_obs = {}
        for key, value in obs.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            
            # Check if key matches either mlp_keys or cnn_keys pattern
            mlp_pattern = eval_keys['mlp_keys']
            cnn_pattern = eval_keys['cnn_keys']
            
            if re.search(mlp_pattern, key) or re.search(cnn_pattern, key):
                filtered_obs[key] = value

        # Convert to numpy for SB3 expert policy (SB3 expects numpy arrays)
        obs_numpy = {}
        for key, value in filtered_obs.items():
            if isinstance(value, torch.Tensor):
                obs_numpy[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                obs_numpy[key] = value
            else:
                obs_numpy[key] = np.array(value)

        # Get action logits from expert agent
        with torch.no_grad():
            # For SB3, we need to get the action logits from the policy
            if hasattr(expert_agent.policy, 'get_distribution'):
                # Get the policy's action distribution
                obs_tensor_flat = expert_agent.policy.obs_to_tensor(obs_numpy)[0]
                distribution = expert_agent.policy.get_distribution(obs_tensor_flat)
                if hasattr(distribution, 'distribution'):
                    # For discrete actions, get logits
                    if hasattr(distribution.distribution, 'logits'):
                        logits = distribution.distribution.logits
                        # Convert back to the target device
                        return logits.to(self.device)
                    # For continuous actions, get mean
                    elif hasattr(distribution.distribution, 'mean'):
                        mean = distribution.distribution.mean
                        return mean.to(self.device)
                return torch.zeros(expert_agent.policy.action_space.n, device=self.device)
            else:
                # Fallback: get action and convert to one-hot
                action, _ = expert_agent.predict(obs_numpy, deterministic=True)
                action_logits = torch.zeros(expert_agent.policy.action_space.n, device=self.device)
                action_logits[action] = 1.0
                return action_logits
    
    def get_all_expert_action_logits(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        Get action logits from all expert policies for distillation.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            dict: Dictionary of expert action logits for each subset
        """
        expert_logits = {}
        
        for subset_name in self.expert_policies.keys():
            expert_logits[subset_name] = self.get_expert_action_logits(subset_name, obs)
        
        return expert_logits


class ConfigurationScheduler:
    """Manages cycling through different observation configurations."""
    
    def __init__(self, eval_keys: Dict, cycle_mode: str = 'episode'):
        """
        Initialize the configuration scheduler.
        
        Args:
            eval_keys: Dictionary of eval_keys for each configuration
            cycle_mode: How to cycle configurations ('episode' or 'batch')
        """
        self.eval_keys = eval_keys
        self.cycle_mode = cycle_mode
        self.config_names = list(eval_keys.keys())
        self.current_config_idx = 0
        self.episode_count = 0
        
    def get_current_config(self) -> Tuple[str, Dict]:
        """
        Get the current configuration.
        
        Returns:
            tuple: (config_name, eval_keys_dict)
        """
        config_name = self.config_names[self.current_config_idx]
        return config_name, self.eval_keys[config_name]
    
    def cycle_config(self, episode_done: bool = False):
        """
        Cycle to the next configuration.
        
        Args:
            episode_done: Whether the current episode is done (for episode-level cycling)
        """
        if self.cycle_mode == 'episode' and episode_done:
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            self.episode_count += 1
            new_config_name = self.config_names[self.current_config_idx]
            print(f"✅ Cycled to configuration: {new_config_name} (episode {self.episode_count})")
        elif self.cycle_mode == 'batch':
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            new_config_name = self.config_names[self.current_config_idx]
            print(f"✅ Cycled to configuration: {new_config_name} (batch mode)")
        else:
            # For training step-based cycling
            self.current_config_idx = (self.current_config_idx + 1) % len(self.config_names)
            self.episode_count += 1
            new_config_name = self.config_names[self.current_config_idx]
            print(f"✅ Cycled to configuration: {new_config_name} (training step {self.episode_count})")


class DistillationTrainer:
    """Custom trainer that implements pure distillation without RL."""
    
    def __init__(self, student_model: PPO, expert_manager: ExpertPolicyManager, 
                 config_scheduler: ConfigurationScheduler, config, vec_env, device: str = 'cpu'):
        self.student_model = student_model
        self.expert_manager = expert_manager
        self.config_scheduler = config_scheduler
        self.config = config
        self.vec_env = vec_env  # Store reference to vectorized environment
        self.device = device
        self.optimizer = optim.Adam(student_model.policy.parameters(), lr=3e-4)
        self.distillation_losses = []
        
        # Get student policy keys from config
        self.student_mlp_keys = getattr(config.keys, 'mlp_keys', '.*') if hasattr(config, 'keys') else '.*'
        self.student_cnn_keys = getattr(config.keys, 'cnn_keys', '.*') if hasattr(config, 'keys') else '.*'
        
    def compute_distillation_loss(self, student_observations: List[Dict], full_observations: List[Dict], expert_config_name: str) -> torch.Tensor:
        """
        Compute distillation loss between student and expert policies.
        
        Args:
            student_observations: Filtered observations for student (from rollout buffer)
            full_observations: Full observations for teacher queries
            expert_config_name: Name of expert configuration to mimic
            
        Returns:
            torch.Tensor: Distillation loss
        """
        if expert_config_name not in self.expert_manager.expert_policies:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_obs = 0
        
        for student_obs, full_obs in zip(student_observations, full_observations):
            # Get expert action logits using full observations
            expert_logits = self.expert_manager.get_expert_action_logits(expert_config_name, full_obs)
            
            # Get student action logits using filtered observations (already filtered by wrapper)
            # Convert to numpy for SB3 student policy
            obs_numpy = {}
            for key, value in student_obs.items():
                if isinstance(value, torch.Tensor):
                    obs_numpy[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    obs_numpy[key] = value
                else:
                    obs_numpy[key] = np.array(value)
            
            # Get student policy distribution
            obs_flat = self.student_model.policy.obs_to_tensor(obs_numpy)[0]
            student_distribution = self.student_model.policy.get_distribution(obs_flat)
            
            if hasattr(student_distribution, 'distribution'):
                if hasattr(student_distribution.distribution, 'logits'):
                    student_logits = student_distribution.distribution.logits.to(self.device)
                elif hasattr(student_distribution.distribution, 'mean'):
                    student_logits = student_distribution.distribution.mean.to(self.device)
                else:
                    continue
            else:
                continue
            
            # Compute KL divergence loss
            if expert_logits.dim() == 1:
                expert_logits = expert_logits.unsqueeze(0)
            if student_logits.dim() == 1:
                student_logits = student_logits.unsqueeze(0)
            
            # KL divergence: KL(student || expert)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(expert_logits, dim=-1),
                reduction='batchmean'
            )
            
            total_loss += kl_loss
            num_obs += 1
        
        if num_obs > 0:
            return total_loss / num_obs
        else:
            return torch.tensor(0.0, device=self.device)
    
    def train_step(self, rollout_buffer: RolloutBuffer) -> float:
        """
        Perform one training step using pure distillation.
        
        Args:
            rollout_buffer: SB3 rollout buffer containing observations
            
        Returns:
            float: Distillation loss value
        """
        # Get current expert configuration
        current_config_name, _ = self.config_scheduler.get_current_config()
        
        # Extract observations from rollout buffer (dict of arrays) - these are already filtered for student
        obs_keys = list(rollout_buffer.observations.keys())
        num_steps = rollout_buffer.observations[obs_keys[0]].shape[0]
        
        # Get filtered observations for student (from rollout buffer)
        student_observations = []
        for i in range(num_steps):
            obs = {}
            for key in obs_keys:
                value = rollout_buffer.observations[key][i]
                # Convert to tensor on the correct device
                if isinstance(value, np.ndarray):
                    obs[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
                else:
                    obs[key] = torch.tensor([value], device=self.device, dtype=torch.float32)
            student_observations.append(obs)
        
        # For distillation, we need full observations for the expert policies
        # Since we can't easily get the actual full observations from the rollout,
        # we'll use a different approach: get a sample of full observations from a separate environment
        full_observations = self._get_sample_full_observations(num_steps)
        
        if not full_observations:
            print("Warning: Could not get full observations for distillation")
            return 0.0
        
        # Compute distillation loss
        distillation_loss = self.compute_distillation_loss(student_observations, full_observations, current_config_name)
        
        # Backpropagate distillation loss ONLY (no RL loss)
        self.optimizer.zero_grad()
        distillation_loss.backward()
        self.optimizer.step()
        
        # Store loss for logging
        self.distillation_losses.append(distillation_loss.item())
        
        return distillation_loss.item()
    
    def _get_sample_full_observations(self, num_steps: int) -> List[Dict]:
        """
        Get sample full observations from a separate environment instance.
        This is a simplified approach for distillation - in practice, you'd want
        to use the actual observations from the rollout.
        """
        full_observations = []
        
        try:
            # Create a temporary environment to get full observations
            suite, task = self.config.task.split('_', 1)
            temp_env = gym.make(task)
            
            # Get sample observations
            for i in range(min(num_steps, 10)):  # Limit to avoid too much computation
                obs, _ = temp_env.reset()
                
                # Convert to tensor format
                full_obs = {}
                for key, value in obs.items():
                    if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                        continue
                    if isinstance(value, np.ndarray):
                        full_obs[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
                    else:
                        full_obs[key] = torch.tensor([value], device=self.device, dtype=torch.float32)
                
                full_observations.append(full_obs)
            
            temp_env.close()
            
            # Repeat observations to match num_steps if needed
            while len(full_observations) < num_steps:
                full_observations.extend(full_observations[:min(len(full_observations), num_steps - len(full_observations))])
            
        except Exception as e:
            print(f"Error getting sample full observations: {e}")
            return []
        
        return full_observations[:num_steps]


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
        debug=False
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        self.make_eval_env_func = make_eval_env_func
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.debug = debug
        self.last_eval = 0
        
        # Get the number of eval configs
        self.num_eval_configs = getattr(config, 'num_eval_configs', 4)
        
        # Parse student keys from the agent's training keys
        if hasattr(config, 'keys') and config.keys:
            # Get student keys using the same logic as training
            def _get_filtered_keys():
                suite, task = config.task.split('_', 1)
                env = gym.make(task)
                
                # Apply observation filtering if keys are specified (same as training)
                if hasattr(config, 'keys') and config.keys:
                    mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
                    cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
                    
                    from baselines.ppo_sb3.train import ObservationFilterWrapper
                    env = ObservationFilterWrapper(
                        env, 
                        mlp_keys=mlp_keys,
                        cnn_keys=cnn_keys
                    )
                
                obs, _ = env.reset()
                env.close()
                return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
            
            # Get the actual keys the agent was trained on
            training_keys = _get_filtered_keys()
            self.student_keys = training_keys
            
            if self.debug:
                print(f"[EVAL CALLBACK] Training keys (student keys): {self.student_keys}")
        else:
            self.student_keys = []
    
    def _get_available_keys(self):
        """Get available keys from the evaluation environment."""
        obs, _ = self.eval_env.reset()
        return [k for k in obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
    
    def _parse_keys_from_pattern(self, pattern, available_keys):
        """Parse keys from regex pattern."""
        import re
        regex = re.compile(pattern)
        matched_keys = [k for k in available_keys if regex.search(k)]
        return matched_keys
    
    def _on_step(self):
        """Called after each step."""
        # Check if we should run evaluation
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            print(f"Running custom evaluation at step {self.num_timesteps}...")
            
            # Run evaluation for each environment configuration
            for subset_idx in range(1, self.num_eval_configs + 1):
                env_name = f"env{subset_idx}"
                if not hasattr(self.config.eval_keys, env_name):
                    print(f"Warning: Missing eval_keys for {env_name}")
                    continue
                
                eval_keys = getattr(self.config.eval_keys, env_name)
                mlp_keys_pattern = getattr(eval_keys, 'mlp_keys', '.*')
                cnn_keys_pattern = getattr(eval_keys, 'cnn_keys', '.*')
                
                # Parse teacher keys for this environment
                available_keys = self._get_available_keys()
                teacher_mlp_keys = self._parse_keys_from_pattern(mlp_keys_pattern, available_keys)
                teacher_cnn_keys = self._parse_keys_from_pattern(cnn_keys_pattern, available_keys)
                teacher_keys = teacher_mlp_keys + teacher_cnn_keys
                
                if self.debug:
                    print(f"[EVAL CALLBACK] {env_name} - Teacher keys: {teacher_keys}")
                
                # Run evaluation for this environment
                self._evaluate_environment(env_name, teacher_keys)
            
            self.last_eval = self.num_timesteps
            
        return True
    
    def _evaluate_environment(self, env_name, teacher_keys):
        """Evaluate the agent on a specific environment configuration."""
        # Create a fresh evaluation environment
        eval_env = self.make_eval_env_func()
        
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
                    teacher_keys, 
                    device=None,  # Use CPU for evaluation
                    debug=self.debug and episode == 0
                )
                
                # Debug logging for first episode
                if self.debug and episode == 0:
                    print(f"[EVAL DEBUG] {env_name} - Original obs keys: {list(obs_tensors.keys())}")
                    print(f"[EVAL DEBUG] {env_name} - Student keys: {self.student_keys}")
                    print(f"[EVAL DEBUG] {env_name} - Teacher keys: {teacher_keys}")
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
                
                # Get action from policy
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
        
        # Compute metrics
        episode_returns = np.array(episode_returns)
        episode_lengths = np.array(episode_lengths)
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"  {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
        
        # Log to wandb if available
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record(f"full_eval_return/{env_name}/mean_return", mean_return)
            self.logger.record(f"full_eval/{env_name}/std_return", std_return)
            self.logger.record(f"full_eval/{env_name}/mean_length", mean_length)
            self.logger.record(f"full_eval/{env_name}/std_length", std_length)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                f"full_eval_return/{env_name}/mean_return": mean_return,
                f"full_eval/{env_name}/std_return": std_return,
                f"full_eval/{env_name}/mean_length": mean_length,
                f"full_eval/{env_name}/std_length": std_length
            })
        
        eval_env.close()


class PureDistillationCallback(BaseCallback):
    """Custom callback for pure distillation training - no RL, only imitation."""
    
    def __init__(self, expert_manager, config_scheduler, config, device='cpu', verbose=1):
        super().__init__(verbose)
        self.expert_manager = expert_manager
        self.config_scheduler = config_scheduler
        self.config = config
        self.device = device
        self.episodes_completed = 0
        self.distillation_trainer = None
        self.training_step_count = 0
        self.last_dones = None
        
    def _on_training_start(self):
        """Called when training starts."""
        # Initialize distillation trainer
        self.distillation_trainer = DistillationTrainer(
            self.model, 
            self.expert_manager, 
            self.config_scheduler,
            self.config,
            self.model.env,  # Pass the vectorized environment
            self.device
        )
        print("🎯 Initialized pure distillation trainer")
        
    def _on_step(self):
        """Called after each step."""
        # Track episode completion without using get_attr
        # We'll track this in _on_rollout_end instead
        return True
    
    def _on_rollout_end(self):
        """Called at the end of a rollout - apply pure distillation loss."""
        if self.distillation_trainer is None:
            return True
            
        # Get current configuration
        current_config_name, current_eval_keys = self.config_scheduler.get_current_config()
        
        # Get expert policy for current configuration
        if current_config_name in self.expert_manager.expert_policies:
            if self.verbose > 0:
                print(f"🎯 Pure distillation: mimicking {current_config_name}")
            
            # Apply distillation loss (this replaces the standard PPO loss)
            try:
                rollout_buffer = self.model.rollout_buffer
                distillation_loss = self.distillation_trainer.train_step(rollout_buffer)
                
                # Log distillation loss
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    self.model.logger.record("distillation/loss", distillation_loss)
                    self.model.logger.record("distillation/expert_config", current_config_name)
                
                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        "distillation/loss": distillation_loss,
                        "distillation/expert_config": current_config_name,
                        "distillation/training_step": self.training_step_count
                    })
                
                self.training_step_count += 1
                
                # Cycle to next expert configuration
                self.config_scheduler.cycle_config()
                
            except Exception as e:
                print(f"❌ Error in distillation training: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"⚠️ Expert policy {current_config_name} not found")
            
        return True


class RewardZeroingWrapper(gym.RewardWrapper):
    """Wrapper that zeros out all rewards to prevent RL learning."""
    
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        """Always return zero reward."""
        return 0.0


def create_distillation_envs(config, seed):
    """Create environments for distillation training."""
    
    def _make_env():
        # Create the base environment using gymnasium
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        
        # Zero out rewards to prevent RL learning - PURE DISTILLATION ONLY
        env = RewardZeroingWrapper(env)
        
        # Apply simple observation filtering if keys are specified (same as ppo_sb3)
        if hasattr(config, 'keys') and config.keys:
            mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
            cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
            
            # Import the simple observation filter wrapper
            from baselines.ppo_sb3.train import ObservationFilterWrapper
            env = ObservationFilterWrapper(
                env, 
                mlp_keys=mlp_keys,
                cnn_keys=cnn_keys
            )
        
        env.reset(seed=seed)
        return env
    
    # Create vectorized environment
    if config.num_envs > 1:
        env_fns = [lambda i=i: _make_env() for i in range(config.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([_make_env])

    vec_env = VecMonitor(vec_env)
    return vec_env


def train_ppo_distill_sb3(config, seed: int, expert_policy_dir: str, device: str = 'cpu'):
    """
    Train a student policy using pure distillation (no RL) from expert policies.
    
    Args:
        config: Configuration object
        seed: Random seed
        expert_policy_dir: Directory containing expert subset policies
        device: Device to use
        
    Returns:
        PPO: Trained SB3 PPO model (student policy)
    """
    print("🚀 train_ppo_distill_sb3 function called!")
    print("🎯 Pure distillation mode: NO RL, only imitation learning")
    
    # Set random seed
    set_random_seed(seed)
    
    # Create expert policy manager
    print("🚀 Creating ExpertPolicyManager...")
    expert_manager = ExpertPolicyManager(expert_policy_dir, device)
    print(f"🚀 Loaded {len(expert_manager.expert_policies)} expert policies")
    
    # Create configuration scheduler
    print("🚀 Creating ConfigurationScheduler...")
    eval_keys = {}
    if hasattr(config, 'eval_keys'):
        for env_idx in range(1, config.num_eval_configs + 1):
            env_name = f"env{env_idx}"
            if hasattr(config.eval_keys, env_name):
                eval_keys[env_name] = {
                    'mlp_keys': getattr(config.eval_keys, env_name).mlp_keys,
                    'cnn_keys': getattr(config.eval_keys, env_name).cnn_keys
                }
    
    config_scheduler = ConfigurationScheduler(eval_keys, cycle_mode=getattr(config, 'cycle_mode', 'episode'))
    print(f"🚀 Created scheduler with {len(eval_keys)} configurations")
    
    # Create environments
    print("🚀 Creating environments...")
    vec_env = create_distillation_envs(config, seed)
    
    # Create evaluation environment with the same wrapper as training
    def _make_eval_env():
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        
        # Apply simple observation filtering if keys are specified (same as original ppo_sb3)
        if hasattr(config, 'keys') and config.keys:
            mlp_keys = getattr(config.keys, 'mlp_keys', '.*')
            cnn_keys = getattr(config.keys, 'cnn_keys', '.*')
            
            # Import the simple observation filter wrapper
            from baselines.ppo_sb3.train import ObservationFilterWrapper
            env = ObservationFilterWrapper(
                env, 
                mlp_keys=mlp_keys,
                cnn_keys=cnn_keys
            )
        
        env.reset(seed=seed)
        return env
    
    eval_env = _make_eval_env()
    
    # Create unfiltered evaluation environment for custom evaluation
    def _make_unfiltered_eval_env():
        suite, task = config.task.split('_', 1)
        env = gym.make(task)
        # No filtering - keep all observations for masking
        env.reset(seed=seed)
        return env
    
    unfiltered_eval_env = _make_unfiltered_eval_env()
    
    # Get evaluation settings from config
    eval_freq = getattr(config.eval, 'eval_freq', max(10000 // config.num_envs, 1))
    n_eval_episodes = getattr(config.eval, 'n_eval_episodes', 5)
    log_interval = getattr(config, 'log_interval', 1)
    
    print(f"Eval frequency: every {eval_freq} env.step() calls (~{eval_freq * config.num_envs} total env steps)")
    print(f"Log interval: every {log_interval} rollouts")
    print(f"Number of eval episodes: {n_eval_episodes}")

    # Initialize W&B if enabled
    run = None
    if config.use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            name=f"ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
            config=config,
            sync_tensorboard=True,
        )

    # Create standard SB3 PPO model (will be trained purely through distillation)
    print("🚀 Creating standard SB3 PPO model with pure distillation callback...")
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for dictionary observations
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./tb_logs/ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
    )

    # Create evaluation callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_models/ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
        log_path=f"./eval_logs/ppo_distill_sb3-{config.task}-{config.exp_name}-seed{seed}",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create pure distillation callback (no RL, only imitation)
    distill_callback = PureDistillationCallback(
        expert_manager, 
        config_scheduler,
        config,
        device=device
    )

    # Create custom evaluation callback
    custom_eval_callback = CustomEvalCallback(
        unfiltered_eval_env,  # Use unfiltered environment for masking
        config,
        _make_unfiltered_eval_env,  # Function to create unfiltered environments
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
        debug=False
    )

    # Prepare callbacks
    callbacks = [eval_callback, distill_callback, custom_eval_callback]
    
    # Note: WandbCallback is not available in this version of SB3
    # Wandb logging is handled manually in the distillation callback

    # Train the model using pure distillation (no RL loss)
    print("🚀 Starting pure distillation training (no RL, only imitation)...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
    )

    vec_env.close()
    eval_env.close()
    unfiltered_eval_env.close()
    
    if run is not None:
        run.finish()
    
    print("🚀 Pure distillation training completed!")
    print(f"🎯 Final configuration: {config_scheduler.get_current_config()[0]}")
    print(f"📊 Expert policies used: {list(expert_manager.expert_policies.keys())}")
    
    return model 