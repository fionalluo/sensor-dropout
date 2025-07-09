#!/usr/bin/env python3
"""
Simplified imitation learning for multi-teacher distillation.
Uses SB3 ONLY for model initialization, then pure PyTorch training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import wandb
import os
import re
import trailenv
import sys
from typing import Dict, List, Tuple, Any
from stable_baselines3 import PPO
from baselines.shared.masking_utils import mask_observations_for_student
from subset_policies.load_subset_policy import SubsetPolicyLoader


class SimpleImitationTrainer:
    """Simple imitation learning trainer using only PyTorch."""
    
    def __init__(self, config, expert_policy_dir: str, device: str = 'cpu', debug: bool = False):
        self.config = config
        self.expert_policy_dir = expert_policy_dir
        self.device = device
        self.debug = debug
        
        # Parse student keys
        self.student_keys = self._parse_student_keys(config)
        
        # Parse all teacher keys  
        self.teacher_keys_by_config = self._get_all_teacher_keys(config)
        self.teacher_configs = list(self.teacher_keys_by_config.keys())
        self.current_teacher_idx = 0  # For cycling through teachers
        
        # Initialize teacher manager
        self.policy_loader = SubsetPolicyLoader(expert_policy_dir, device)
        
        # Create environment and model
        self.env = self._create_environment()
        self.student_model, self.action_space = self._create_student_model()
        
        # Get training hyperparameters
        distillation_config = getattr(config, 'distillation', {})
        self.gradient_clip_norm = getattr(distillation_config, 'gradient_clip_norm', None)
        self.l2_regularization = getattr(distillation_config, 'l2_regularization', 0.0)
        
        # Initialize optimizer with weight decay for L2 regularization
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=config.learning_rate,
            weight_decay=self.l2_regularization
        )
        
        print(f"Simple Imitation Learning initialized")
        print(f"Student keys: {self.student_keys}")
        print(f"Teacher configs: {self.teacher_configs}")
        print(f"Action space: {self.action_space}")
    
    def _obs_to_tensors(self, obs: Dict, device: str = None) -> Dict[str, torch.Tensor]:
        """Convert observations dict to tensors, filtering out metadata keys."""
        obs_tensors = {}
        for key, value in obs.items():
            if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if isinstance(value, np.ndarray):
                    obs_tensors[key] = torch.tensor(value, dtype=torch.float32, device=device)
                else:
                    obs_tensors[key] = torch.tensor([value], dtype=torch.float32, device=device)
        return obs_tensors
    
    def _obs_to_numpy(self, obs: Dict) -> Dict[str, np.ndarray]:
        """Convert observations dict to numpy arrays for SB3 compatibility."""
        obs_numpy = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                obs_numpy[key] = value.cpu().numpy()
            else:
                obs_numpy[key] = np.array(value)
        return obs_numpy
    
    def _add_batch_dim(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add batch dimension to observations for SB3 model input."""
        return {k: np.expand_dims(v, axis=0) for k, v in obs.items()}

    def _parse_student_keys(self, config) -> List[str]:
        """Parse student keys from config."""
        suite, task = config.task.split('_', 1)
        temp_env = gym.make(task)
        sample_obs, _ = temp_env.reset()
        temp_env.close()
        
        available_keys = [k for k in sample_obs.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
        
        student_keys = []
        if hasattr(config.keys, 'mlp_keys'):
            mlp_pattern = re.compile(config.keys.mlp_keys)
            student_keys.extend([k for k in available_keys if mlp_pattern.search(k)])
        if hasattr(config.keys, 'cnn_keys'):
            cnn_pattern = re.compile(config.keys.cnn_keys)
            student_keys.extend([k for k in available_keys if cnn_pattern.search(k)])
        
        return student_keys
    
    def _get_all_teacher_keys(self, config):
        """Get all teacher keys from config."""
        teacher_keys_by_config = {}
        for i in range(1, config.num_eval_configs + 1):
            config_name = f'env{i}'
            if hasattr(config.eval_keys, config_name):
                teacher_keys_by_config[config_name] = getattr(config.eval_keys, config_name)
        
        if not teacher_keys_by_config:
            raise ValueError(f"No teacher configurations found! Expected env1 to env{config.num_eval_configs}")
        
        return teacher_keys_by_config
    
    def get_current_teacher_config(self):
        """Get the current teacher configuration name."""
        return self.teacher_configs[self.current_teacher_idx]
    
    def cycle_to_next_teacher(self):
        """Cycle to the next teacher."""
        self.current_teacher_idx = (self.current_teacher_idx + 1) % len(self.teacher_configs)
    
    def _create_environment(self):
        """Create simple environment."""
        suite, task = self.config.task.split('_', 1)
        env = gym.make(task)
        return env
    
    def _parse_keys_from_patterns(self, key_patterns, available_keys: List[str]) -> Tuple[List[str], List[str]]:
        """Parse keys from regex patterns."""
        mlp_keys = []
        cnn_keys = []
        
        # Handle both dict and SimpleNamespace objects
        if hasattr(key_patterns, 'mlp_keys'):
            mlp_pattern = re.compile(key_patterns.mlp_keys)
            mlp_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and mlp_pattern.search(k)]
        elif isinstance(key_patterns, dict) and 'mlp_keys' in key_patterns:
            mlp_pattern = re.compile(key_patterns['mlp_keys'])
            mlp_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and mlp_pattern.search(k)]
        
        if hasattr(key_patterns, 'cnn_keys'):
            cnn_pattern = re.compile(key_patterns.cnn_keys)
            cnn_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and cnn_pattern.search(k)]
        elif isinstance(key_patterns, dict) and 'cnn_keys' in key_patterns:
            cnn_pattern = re.compile(key_patterns['cnn_keys'])
            cnn_keys = [k for k in available_keys if k not in ['reward', 'is_first', 'is_last', 'is_terminal'] and cnn_pattern.search(k)]
        
        return mlp_keys, cnn_keys
    
    def _create_student_model(self):
        """Initialize student model using SB3, then extract PyTorch model."""
        # Extract student observation space from environment's existing observation space
        env_obs_space = self.env.observation_space
        
        if not isinstance(env_obs_space, gym.spaces.Dict):
            print(f"ERROR: Expected Dict observation space, got {type(env_obs_space)}")
            import sys
            sys.exit(1)
        
        # Extract only the student keys from environment's observation space
        student_obs_spaces = {}
        for student_key in self.student_keys:
            if student_key not in env_obs_space.spaces:
                print(f"ERROR: Student key '{student_key}' not found in environment observation space!")
                print(f"Available keys: {list(env_obs_space.spaces.keys())}")
                print(f"Student keys from config: {self.student_keys}")
                import sys
                sys.exit(1)
            
            # Use the environment's existing space definition for this key
            student_obs_spaces[student_key] = env_obs_space.spaces[student_key]
        
        # Create student observation space using environment's definitions
        student_obs_space = gym.spaces.Dict(student_obs_spaces)
        print(f"Student observation space: {student_obs_space}")
        
        # Create a temporary environment with student observation space
        # We need to do this because self.env has the full observation space and can't be passed in
        class TempEnv(gym.Env):
            def __init__(self, obs_space, action_space):
                self.observation_space = obs_space
                self.action_space = action_space
            
            def reset(self):
                # Return a dummy observation
                dummy_obs = {}
                for key, space in self.observation_space.spaces.items():
                    dummy_obs[key] = np.zeros(space.shape, dtype=space.dtype)
                return dummy_obs, {}
            
            def step(self, action):
                return self.reset()[0], 0.0, False, False, {}
        
        temp_env = TempEnv(student_obs_space, self.env.action_space)
        
        # Create SB3 policy with correct observation space
        temp_policy = PPO("MultiInputPolicy", temp_env, device=self.device, verbose=0)
        
        # Extract the actual PyTorch model
        student_model = temp_policy.policy
        action_space = self.env.action_space
        
        print(f"Extracted student model: {type(student_model)}")
        return student_model, action_space
    
    def get_teacher_probs(self, full_obs: Dict, teacher_config: str = None) -> torch.Tensor:
        """Get teacher action probabilities for given observation."""
        # Use current teacher if none specified
        if teacher_config is None:
            teacher_config = self.get_current_teacher_config()
        
        # Load teacher policy
        teacher_agent, _, _ = self.policy_loader.load_policy(teacher_config)
        
        # Get the exact keys the teacher policy expects (from its observation space)
        expected_teacher_keys = list(teacher_agent.policy.observation_space.spaces.keys())
        
        # Filter observations to only include keys the teacher expects
        teacher_obs = {}
        for expected_key in expected_teacher_keys:
            if expected_key in full_obs:
                # Direct match - use this observation
                teacher_obs[expected_key] = full_obs[expected_key]
            else:
                # This should never happen - if teacher expects a key that doesn't exist, 
                print(f"ERROR: Teacher {teacher_config} expects key '{expected_key}' but it's not in observation!")
                import sys
                sys.exit(1)
                
        # Convert to numpy for SB3 and add batch dimension
        obs_numpy = self._obs_to_numpy(teacher_obs)
        obs_batch = self._add_batch_dim(obs_numpy)
        
        # Get teacher probabilities directly
        with torch.no_grad():
            obs_tensor = teacher_agent.policy.obs_to_tensor(obs_batch)[0]
            distribution = teacher_agent.policy.get_distribution(obs_tensor)
            if hasattr(distribution.distribution, 'probs'):
                teacher_probs = distribution.distribution.probs.squeeze(0)
            else:
                # Fallback: compute probabilities from logits
                if hasattr(distribution.distribution, 'logits'):
                    teacher_probs = F.softmax(distribution.distribution.logits.squeeze(0), dim=-1)
                else:
                    # Fallback for continuous actions - uniform distribution
                    teacher_probs = torch.ones(self.action_space.n, device=self.device) / self.action_space.n
        
        return teacher_probs.to(self.device)
    
    def get_student_probs(self, full_obs: Dict, teacher_config: str = None) -> torch.Tensor:
        """Get student action probabilities for given observation (after masking)."""
        # Use current teacher if none specified
        if teacher_config is None:
            teacher_config = self.get_current_teacher_config()
        
        # Get teacher keys for this configuration
        teacher_keys = self.teacher_keys_by_config[teacher_config]
        
        # Parse teacher keys for masking
        teacher_mlp_keys, teacher_cnn_keys = self._parse_keys_from_patterns(
            teacher_keys, list(full_obs.keys())
        )
        all_teacher_keys = teacher_mlp_keys + teacher_cnn_keys
        
        # Apply masking to get what student should see
        masked_obs = mask_observations_for_student(
            full_obs,
            self.student_keys,
            all_teacher_keys,
            device=None,
            debug=False
        )
                
        # Convert to numpy for SB3 model and add batch dimension
        obs_numpy = self._obs_to_numpy(masked_obs)
        obs_batch = self._add_batch_dim(obs_numpy)
        
        # Get student probabilities directly
        obs_tensor = self.student_model.obs_to_tensor(obs_batch)[0]
        distribution = self.student_model.get_distribution(obs_tensor)
        student_probs = distribution.distribution.probs.squeeze(0)
        
        return student_probs
    
    def train_step(self, obs_batch: List[Dict], teacher_probs_batch: List[torch.Tensor]) -> float:
        """Single training step with batch of observations and teacher probabilities."""
        total_loss = 0.0
        
        for obs, teacher_probs in zip(obs_batch, teacher_probs_batch):
            # Get student probabilities directly
            student_probs = self.get_student_probs(obs)
            
            print(f"Student probs: {student_probs}")
            print(f"Teacher probs: {teacher_probs}")
            
            # Compute KL divergence loss: KL(teacher || student)
            # This measures how much the student distribution differs from teacher distribution
            loss = F.kl_div(
                torch.log(student_probs + 1e-8),  # log probabilities of student (add small epsilon for stability)
                teacher_probs,                    # target probabilities from teacher
                reduction='batchmean'
            )
            
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping if specified
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
        
        return total_loss / len(obs_batch)
    
    def collect_data(self, num_episodes: int = 10):
        """Collect observation-teacher action pairs by rolling out episodes with student."""
        observations = []
        teacher_probs = []
        
        # Track teacher usage for verification
        teacher_usage = {teacher: 0 for teacher in self.teacher_configs}
        
        for episode in range(num_episodes):
            # Get current teacher and cycle through all teachers
            current_teacher = self.get_current_teacher_config()
            teacher_usage[current_teacher] += 1
                        
            # Start episode
            obs, _ = self.env.reset()
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated) and episode_length < 100:  # Max 100 steps per episode
                # Convert observation to tensors
                obs_tensors = self._obs_to_tensors(obs)
                
                # Get teacher probabilities for this state (using current teacher)
                teacher_probs_sample = self.get_teacher_probs(obs_tensors, current_teacher)
                
                # Store the observation and teacher probabilities
                observations.append(obs_tensors.copy())
                teacher_probs.append(teacher_probs_sample)
                
                # Get student action to continue episode (using same teacher for consistency)
                with torch.no_grad():
                    student_probs = self.get_student_probs(obs_tensors, current_teacher)
                    # Use some exploration - not just greedy
                    action = torch.multinomial(student_probs, 1).item()
                
                # Step environment with student action
                obs, reward, done, truncated, info = self.env.step(action)
                episode_length += 1
                        
            # Cycle to next teacher for next episode
            self.cycle_to_next_teacher()
        
        print(f"  Training data collected using teachers: {teacher_usage}")
        return observations, teacher_probs
    
    def train(self):
        """Main training loop."""
        print(f"Starting simple imitation learning for {self.config.total_timesteps} steps")
        print(f"Training with multi-teacher distillation from: {self.teacher_configs}")
        
        step = 0
        iteration = 0
        episodes_per_iteration = getattr(self.config, 'episodes_per_iteration', 5)
        last_eval_step = -1  # Track last evaluation step to handle irregular increments
        
        # Set current step for evaluation logging
        self.current_step = step
        
        # Evaluate at step 0 (before training)
        print(f"\nInitial evaluation at step {step}")
        self.evaluate()
        last_eval_step = step
        
        while step < self.config.total_timesteps:
            iteration += 1
            print(f"\nIteration {iteration} (Step {step}/{self.config.total_timesteps})")
            
            # Collect data from episode rollouts
            observations, teacher_probs = self.collect_data(episodes_per_iteration)
            
            # Train on collected data
            loss = self.train_step(observations, teacher_probs)
            print(f"  Training loss: {loss:.6f}")
            
            # Update step count (approximate based on data collected)
            prev_step = step
            step += len(observations)
            self.current_step = step  # Update current step for evaluation logging
            
            # Log to wandb
            if self.config.track and wandb.run is not None:
                wandb.log({
                    'train/imitation_loss': loss,
                    'train/iteration': iteration,
                    'train/samples_collected': len(observations)
                }, step=step)
            
            # Check if we've crossed an evaluation boundary
            # Calculate how many eval_freq intervals we've passed since last evaluation
            prev_eval_count = last_eval_step // self.config.eval_freq
            current_eval_count = step // self.config.eval_freq
            
            if current_eval_count > prev_eval_count:
                # We've crossed at least one evaluation boundary
                next_eval_step = (prev_eval_count + 1) * self.config.eval_freq
                print(f"\nEvaluation triggered: step {step} crossed boundary at {next_eval_step} (eval_freq={self.config.eval_freq})")
                print(f"  Previous eval at step {last_eval_step}, current step {step}")
                self.evaluate()
                last_eval_step = step
        
        print("Simple imitation training completed!")
    
    def evaluate(self, n_episodes: int = None):
        """Comprehensive evaluation: default + cross-subset evaluations."""
        # Use config value if n_episodes not specified
        if n_episodes is None:
            n_episodes = getattr(self.config, 'n_eval_episodes', 5)
        
        print(f"Starting comprehensive evaluation with {n_episodes} episodes...")
        print(f"Number of eval configs: {self.config.num_eval_configs}")
        
        # 1. Evaluate student on default training environment (student keys only, no teacher masking)
        print("📚 Evaluating student on default training environment...")
        self._evaluate_student_default(n_episodes)
        
        # 2. Evaluate student on each teacher configuration (with proper masking)
        print("🎓 Evaluating student on teacher subsets...")
        for i in range(1, self.config.num_eval_configs + 1):
            env_name = f'env{i}'
            teacher_keys = getattr(self.config.eval_keys, env_name)
            self._evaluate_environment(env_name, teacher_keys, n_episodes)
        
        # 3. Evaluate each teacher on their specific environment configuration
        self._evaluate_teachers(n_episodes)
        
        print("Evaluation complete!")
    
    def _evaluate_student_default(self, n_episodes: int):
        """Evaluate student on training environment with student keys only."""
        print(f"  Evaluating student default (training environment)...")
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Convert obs to tensors and filter to student keys only
                obs_tensors = self._obs_to_tensors(obs)
                student_obs = {key: obs_tensors[key] for key in self.student_keys if key in obs_tensors}
                
                # Convert to numpy for SB3 model and add batch dimension
                obs_numpy = self._obs_to_numpy(student_obs)
                obs_batch = self._add_batch_dim(obs_numpy)
                
                # Get student action
                with torch.no_grad():
                    obs_tensor = self.student_model.obs_to_tensor(obs_batch)[0]
                    distribution = self.student_model.get_distribution(obs_tensor)
                    action_probs = F.softmax(distribution.distribution.logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
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
                
        # Log to wandb - this is the main student performance metric
        if self.config.track and wandb.run is not None:
            metrics_to_log = {
                "eval/mean_return": mean_return,
                "eval/std_return": std_return,
                "eval/mean_length": mean_length
            }
            # Use current step if available, otherwise let wandb use its internal counter
            step_to_use = getattr(self, 'current_step', None)
            print(f"    Logging to wandb: {list(metrics_to_log.keys())} at step {step_to_use}")
            if step_to_use is not None:
                wandb.log(metrics_to_log, step=step_to_use)
            else:
                wandb.log(metrics_to_log)
        else:
            print(f"    Wandb logging skipped: track={self.config.track}, wandb.run={wandb.run is not None}")
    
    def _evaluate_environment(self, env_name: str, teacher_keys, n_episodes: int):
        """Evaluate student policy on a specific teacher configuration."""
        print(f"  Evaluating {env_name}...")
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Convert observations to tensors for masking
                obs_tensors = self._obs_to_tensors(obs)
                
                # Get teacher probabilities for this environment configuration
                with torch.no_grad():
                    teacher_probs = self.get_teacher_probs(obs_tensors, env_name)
                    teacher_action = torch.argmax(teacher_probs).item()
                
                # Get student probabilities (with masking)
                with torch.no_grad():
                    student_probs = self.get_student_probs(obs_tensors, env_name)
                    student_action = torch.argmax(student_probs).item()
                
                # Compare teacher vs student actions for debugging
                # if teacher_action != student_action:
                #     print(f"    [{env_name}] Step {episode_length}: ACTION MISMATCH!")
                #     print(f"      Teacher action: {teacher_action}, probs: {teacher_probs.detach().cpu().numpy()}")
                #     print(f"      Student action: {student_action}, probs: {student_probs.detach().cpu().numpy()}")
                # elif episode == 0 and episode_length < 5:  # Show first few steps of first episode
                #     print(f"    [{env_name}] Step {episode_length}: actions match ({teacher_action})")
                #     print(f"      Teacher probs: {teacher_probs.detach().cpu().numpy()}")
                #     print(f"      Student probs: {student_probs.detach().cpu().numpy()}")
                
                # Use student action to step environment (since we're evaluating the student)
                action = student_action
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
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
        
        print(f"    {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
        
        # Log to wandb following ppo_dropout pattern
        if self.config.track and wandb.run is not None:
            metrics_to_log = {
                f"full_eval_return/{env_name}/mean_return": mean_return,
                f"full_eval/{env_name}/std_return": std_return,
                f"full_eval/{env_name}/mean_length": mean_length
            }
            # Use current step if available, otherwise let wandb use its internal counter
            step_to_use = getattr(self, 'current_step', None)
            print(f"    Logging to wandb: {list(metrics_to_log.keys())} at step {step_to_use}")
            if step_to_use is not None:
                wandb.log(metrics_to_log, step=step_to_use)
            else:
                wandb.log(metrics_to_log)
        else:
            print(f"    Wandb logging skipped: track={self.config.track}, wandb.run={wandb.run is not None}")

    def _evaluate_teachers(self, n_episodes: int):
        """Evaluate each teacher on their specific environment configuration."""
        print(f"  Evaluating teachers on their specific configurations...")
        
        for i in range(1, self.config.num_eval_configs + 1):
            env_name = f'env{i}'
            if hasattr(self.config.eval_keys, env_name):
                teacher_keys = getattr(self.config.eval_keys, env_name)                
                # Load teacher policy
                teacher_agent, _, _ = self.policy_loader.load_policy(env_name)
                
                # Get the exact keys the teacher expects
                expected_teacher_keys = list(teacher_agent.policy.observation_space.spaces.keys())
                
                episode_returns = []
                episode_lengths = []
                
                for episode in range(n_episodes):
                    obs, _ = self.env.reset()
                    episode_return = 0
                    episode_length = 0
                    
                    done = False
                    truncated = False
                    
                    while not (done or truncated):
                        # Convert observations to tensors
                        obs_tensors = self._obs_to_tensors(obs)
                        
                        # Filter observations to only include keys the teacher expects
                        teacher_obs = {}
                        for expected_key in expected_teacher_keys:
                            if expected_key in obs_tensors:
                                teacher_obs[expected_key] = obs_tensors[expected_key]
                            else:
                                print(f"ERROR: Teacher {env_name} expects key '{expected_key}' but it's not in observation!")
                                import sys
                                sys.exit(1)
                        
                        # Convert to numpy for SB3 and add batch dimension
                        obs_numpy = self._obs_to_numpy(teacher_obs)
                        obs_batch = self._add_batch_dim(obs_numpy)
                        
                        # Get teacher action
                        with torch.no_grad():
                            obs_tensor = teacher_agent.policy.obs_to_tensor(obs_batch)[0]
                            distribution = teacher_agent.policy.get_distribution(obs_tensor)
                            action_probs = F.softmax(distribution.distribution.logits, dim=-1)
                            action = torch.argmax(action_probs).item()
                        
                        # Step environment
                        obs, reward, done, truncated, info = self.env.step(action)
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
                
                print(f"      Teacher {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
                
                # Log to wandb
                if self.config.track and wandb.run is not None:
                    metrics_to_log = {
                        f"teacher/{env_name}/mean_return": mean_return,
                        f"teacher/{env_name}/std_return": std_return,
                        f"teacher/{env_name}/mean_length": mean_length
                    }
                    # Use current step if available
                    step_to_use = getattr(self, 'current_step', None)
                    if step_to_use is not None:
                        wandb.log(metrics_to_log, step=step_to_use)
                    else:
                        wandb.log(metrics_to_log)


def train_simple_imitation(config, expert_policy_dir: str, device: str = 'cpu', debug: bool = False):
    """Main training function for simple imitation learning."""
    # Initialize wandb
    if hasattr(config, 'track') and config.track:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"simple_imitation_{config.exp_name}_{config.seed}",
            config=vars(config)
        )
    
    # Create trainer and train
    trainer = SimpleImitationTrainer(config, expert_policy_dir, device, debug)
    trainer.train()
    
    if config.track and wandb.run is not None:
        wandb.finish() 