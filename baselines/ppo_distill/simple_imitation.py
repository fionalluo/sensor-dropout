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
import random
import trailenv
import sys
from typing import Dict, List, Tuple, Any
from stable_baselines3 import PPO
from baselines.shared.masking_utils import mask_observations_for_student
from subset_policies.load_subset_policy import SubsetPolicyLoader


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleImitationTrainer:
    """Simple imitation learning trainer using only PyTorch."""
    
    def __init__(self, config, expert_policy_dir: str, device: str = 'cpu', debug: bool = False):
        self.config = config
        self.expert_policy_dir = expert_policy_dir
        self.device = device
        self.debug = debug
        
        # Parse student keys
        self.student_keys = self._parse_student_keys(config)
        print(f"🔍 DEBUG: Parsed student keys: {self.student_keys}")
        
        # Parse all teacher keys  
        self.teacher_keys_by_config = self._get_all_teacher_keys(config)
        self.teacher_configs = list(self.teacher_keys_by_config.keys())
        self.current_teacher_idx = 0  # For cycling through teachers
        
        print(f"🔍 DEBUG: Teacher configs and keys:")
        for config_name, keys in self.teacher_keys_by_config.items():
            print(f"  {config_name}: {keys.__dict__ if hasattr(keys, '__dict__') else keys}")
        
        # Initialize teacher manager
        self.policy_loader = SubsetPolicyLoader(expert_policy_dir, device)
        
        # Create environment and model
        self.env = self._create_environment()
        self.student_model, self.action_space = self._create_student_model()
        
        # Initialize optimizer - use config directly
        print(f"🔧 Using learning rate: {self.config.distillation.learning_rate} (with cosine annealing)")
        
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=self.config.distillation.learning_rate,
            weight_decay=getattr(self.config.distillation, 'l2_regularization', 0.0)
        )
        
        # Add cosine annealing learning rate scheduler
        # T_max is the number of training iterations (total_timesteps / batch_size)
        max_iterations = self.config.total_timesteps // self.config.distillation.batch_size
        min_lr = self.config.distillation.learning_rate * 0.01  # Minimum LR is 1% of initial LR
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max_iterations,
            eta_min=min_lr
        )
        print(f"🔧 Cosine annealing: {max_iterations} iterations, min_lr={min_lr:.6f}")
        
        # Evaluate teachers once at initialization (they're static, so no need to re-evaluate)
        print("🔧 Evaluating teachers once at initialization...")
        self.teacher_metrics = self._evaluate_teachers_once(self.config.eval.n_eval_episodes)
        print("✅ Teacher evaluation complete!")
        
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
        sample_obs, _ = temp_env.reset(seed=config.seed)
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
        """Get all teacher keys from config (the regex)."""
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
        
        # Set environment seed for reproducibility
        env.reset(seed=self.config.seed)
        print(f"🔧 Environment seeded with seed: {self.config.seed}")
        
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
                import sys
                sys.exit(1)
            
            # Use the environment's existing space definition for this key
            student_obs_spaces[student_key] = env_obs_space.spaces[student_key]
        
        # Create student observation space using environment's definitions
        student_obs_space = gym.spaces.Dict(student_obs_spaces)
        print(f"🔍 DEBUG: Student observation space: {student_obs_space}")
        print(f"🔍 DEBUG: Student observation space keys: {list(student_obs_space.spaces.keys())}")
        for key, space in student_obs_space.spaces.items():
            print(f"  {key}: {space}")
        
        # Check action space type for compatibility
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            print(f"🔍 DEBUG: Discrete action space with {action_space.n} actions")
            self.is_discrete_action = True
        elif isinstance(action_space, gym.spaces.Box):
            print(f"🔍 DEBUG: Continuous action space with shape {action_space.shape}")
            self.is_discrete_action = False
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
        
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
    
    def get_teacher_distribution_params(self, full_obs: Dict, teacher_config: str = None) -> torch.Tensor:
        """Get teacher action distribution parameters for given observation."""
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
        
        # Get teacher distribution parameters
        with torch.no_grad():
            obs_tensor = teacher_agent.policy.obs_to_tensor(obs_batch)[0]
            distribution = teacher_agent.policy.get_distribution(obs_tensor)
            
            if self.is_discrete_action:
                # For discrete actions, return probabilities
                if hasattr(distribution.distribution, 'probs'):
                    teacher_params = distribution.distribution.probs.squeeze(0)
                else:
                    # Fallback: compute probabilities from logits
                    if hasattr(distribution.distribution, 'logits'):
                        teacher_params = F.softmax(distribution.distribution.logits.squeeze(0), dim=-1)
                    else:
                        # Fallback - uniform distribution
                        teacher_params = torch.ones(self.action_space.n, device=self.device) / self.action_space.n
            else:
                # For continuous actions, return mean and log_std
                if hasattr(distribution.distribution, 'loc') and hasattr(distribution.distribution, 'scale'):
                    # Normal distribution
                    mean = distribution.distribution.loc.squeeze(0)
                    log_std = torch.log(distribution.distribution.scale.squeeze(0))
                    teacher_params = torch.cat([mean, log_std], dim=-1)  # Concatenate mean and log_std
                else:
                    # Fallback for other continuous distributions
                    mean = distribution.mean.squeeze(0)
                    # Assume unit variance if scale not available
                    log_std = torch.zeros_like(mean)
                    teacher_params = torch.cat([mean, log_std], dim=-1)
        
        return teacher_params.to(self.device)
    
    def get_student_distribution_params(self, full_obs: Dict, teacher_config: str = None) -> torch.Tensor:
        """Get student action distribution parameters for given observation (after masking)."""
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
        
        # Get student distribution parameters
        obs_tensor = self.student_model.obs_to_tensor(obs_batch)[0]
        distribution = self.student_model.get_distribution(obs_tensor)
        
        if self.is_discrete_action:
            # For discrete actions, return probabilities
            if hasattr(distribution.distribution, 'probs'):
                student_params = distribution.distribution.probs.squeeze(0)
            else:
                # Fallback: compute probabilities from logits
                student_params = F.softmax(distribution.distribution.logits.squeeze(0), dim=-1)
        else:
            # For continuous actions, return mean and log_std
            if hasattr(distribution.distribution, 'loc') and hasattr(distribution.distribution, 'scale'):
                # Normal distribution
                mean = distribution.distribution.loc.squeeze(0)
                log_std = torch.log(distribution.distribution.scale.squeeze(0))
                student_params = torch.cat([mean, log_std], dim=-1)  # Concatenate mean and log_std
            else:
                # Fallback for other continuous distributions
                mean = distribution.mean.squeeze(0)
                # Assume unit variance if scale not available
                log_std = torch.zeros_like(mean)
                student_params = torch.cat([mean, log_std], dim=-1)
        
        return student_params
    
    def collect_and_train(self, num_samples: int = 2048) -> float:
        """Collect a fixed number of samples by rolling out episodes and train on the accumulated batch."""
        # Set training flag for debug prints
        self._in_training = True
        
        # Track teacher usage for verification
        teacher_usage = {teacher: 0 for teacher in self.teacher_configs}
        
        # Reset gradients at the start
        self.optimizer.zero_grad()
        
        # Initialize lists to collect distribution parameters
        student_params_list = []
        teacher_params_list = []
        
        # Initialize episode state if not already in progress
        if not hasattr(self, '_current_obs') or self._current_obs is None:
            # Start fresh episode with seed for reproducibility
            episode_seed = self.config.seed + getattr(self, '_episode_count', 0)
            obs, _ = self.env.reset(seed=episode_seed)
            self._current_obs = obs
            self._episode_length = 0
            self._episode_count = getattr(self, '_episode_count', 0)
            print(f"Starting new episode {self._episode_count} with seed {episode_seed}")
        else:
            # Resume from where we left off
            obs = self._current_obs
            print(f"Resuming episode {self._episode_count} from step {self._episode_length}")
        
        samples_collected_this_call = 0
        
        while samples_collected_this_call < num_samples:
            # Cycle through teachers for each sample to get diverse training data
            current_teacher = self.get_current_teacher_config()
            teacher_usage[current_teacher] += 1
            
            # Convert observation to tensors
            obs_tensors = self._obs_to_tensors(obs)
            
            # Get student distribution parameters
            student_params = self.get_student_distribution_params(obs_tensors, current_teacher)
            
            # Calculate teacher distribution parameters for this observation
            teacher_params = self.get_teacher_distribution_params(obs_tensors, current_teacher)
            
            # Collect parameters for batch processing
            student_params_list.append(student_params)
            teacher_params_list.append(teacher_params)
            
            samples_collected_this_call += 1
            
            # Cycle to next teacher for next sample
            self.cycle_to_next_teacher()
            
            # Get student action to continue episode
            with torch.no_grad():
                if self.is_discrete_action:
                    # For discrete actions, sample from probabilities
                    action = torch.multinomial(student_params, 1).item()
                else:
                    # For continuous actions, sample from normal distribution
                    action_dim = self.action_space.shape[0]
                    mean = student_params[:action_dim]
                    log_std = student_params[action_dim:]
                    std = torch.exp(log_std)
                    action = torch.normal(mean, std).cpu().numpy()
            
            # Step environment with student action
            obs, reward, done, truncated, info = self.env.step(action)
            self._episode_length += 1
            
            # Check if episode ended
            if done or truncated or self._episode_length >= 100:  # Max 100 steps per episode
                # Start new episode if we haven't collected enough samples yet
                if samples_collected_this_call < num_samples:
                    self._episode_count += 1
                    episode_seed = self.config.seed + self._episode_count
                    obs, _ = self.env.reset(seed=episode_seed)
                    self._episode_length = 0
                else:
                    # We've collected enough samples and episode ended - clear state
                    self._current_obs = None
            
            # Update current state
            self._current_obs = obs
        
        # Compute batch loss
        student_params_batch = torch.stack(student_params_list)  # [batch_size, param_dim]
        teacher_params_batch = torch.stack(teacher_params_list)
        
        if self.is_discrete_action:
            # For discrete actions, use KL divergence between probability distributions
            loss = F.kl_div(torch.log(student_params_batch + 1e-8), teacher_params_batch, reduction='batchmean')
        else:
            # For continuous actions, use MSE loss on distribution parameters
            # Split parameters into mean and log_std
            action_dim = self.action_space.shape[0]
            student_mean = student_params_batch[:, :action_dim]
            student_log_std = student_params_batch[:, action_dim:]
            teacher_mean = teacher_params_batch[:, :action_dim]
            teacher_log_std = teacher_params_batch[:, action_dim:]
            
            # MSE loss on means and log stds
            mean_loss = F.mse_loss(student_mean, teacher_mean)
            std_loss = F.mse_loss(student_log_std, teacher_log_std)
            loss = mean_loss + std_loss
        
        loss.backward()
        
        # Calculate total loss for logging (maintain same logging behavior)
        total_loss = loss.item() * num_samples
        
        # Apply gradient clipping if specified (on accumulated gradients)
        if hasattr(self.config.distillation, 'gradient_clip_norm') and self.config.distillation.gradient_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.distillation.gradient_clip_norm)
        
        # Single optimizer step on accumulated gradients
        self.optimizer.step()
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
        # Reset training flag
        self._in_training = False
        
        print(f"  ✅ Collected {samples_collected_this_call} samples, trained using teachers: {teacher_usage}")
        print(f"  📈 Current learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
        return total_loss / num_samples
    
    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("🎓 MULTI-TEACHER IMITATION LEARNING")
        print("Training student by cycling through all teacher configurations.")
        print(f"Available teachers: {self.teacher_configs}")
        print("Student will learn from diverse teacher policies to improve generalization.")
        print("=" * 60)
        print(f"Starting simple imitation learning for {self.config.total_timesteps} steps")
        
        step = 0
        iteration = 0
        print(f"🔧 Using batch size from config: {self.config.distillation.batch_size}")
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
            loss = self.collect_and_train(self.config.distillation.batch_size)
            print(f"  Training loss: {loss:.6f}")
            
            # Update step count (approximate based on data collected)
            prev_step = step
            step += self.config.distillation.batch_size # Use the actual number of samples collected
            self.current_step = step  # Update current step for evaluation logging
            
            # Log to wandb
            if self.config.track and wandb.run is not None:
                wandb.log({
                    'train/imitation_loss': loss,
                    'train/iteration': iteration,
                    'train/samples_collected': self.config.distillation.batch_size,
                    'train/learning_rate': self.scheduler.get_last_lr()[0],  # Get current LR from scheduler
                    'global_step': step
                }, step=step)
            
            # Check if we've crossed an evaluation boundary
            # Calculate how many eval_freq intervals we've passed since last evaluation
            prev_eval_count = last_eval_step // self.config.eval.eval_freq
            current_eval_count = step // self.config.eval.eval_freq
            
            if current_eval_count > prev_eval_count:
                # We've crossed at least one evaluation boundary
                next_eval_step = (prev_eval_count + 1) * self.config.eval.eval_freq
                print(f"\nEvaluation triggered: step {step} crossed boundary at {next_eval_step} (eval_freq={self.config.eval.eval_freq})")
                print(f"  Previous eval at step {last_eval_step}, current step {step}")
                self.evaluate()
                last_eval_step = step
        
        print("Simple imitation training completed!")
    
    def evaluate(self, n_episodes: int = None):
        """Comprehensive evaluation: default + cross-subset evaluations."""
        # Use config value if n_episodes not specified
        if n_episodes is None:
            n_episodes = self.config.eval.n_eval_episodes
        
        print(f"Starting comprehensive evaluation with {n_episodes} episodes...")
        print(f"Number of eval configs: {self.config.num_eval_configs}")
        
        # 1. Evaluate student on default training environment (student keys only, no teacher masking)
        self._evaluate_student_default(n_episodes)
        
        # 2. Evaluate student on each teacher configuration (with proper masking)
        env_metrics = {}  # Collect metrics from all environments
        for i in range(1, self.config.num_eval_configs + 1):
            env_name = f'env{i}'
            teacher_keys = getattr(self.config.eval_keys, env_name)
            metrics = self._evaluate_environment(env_name, teacher_keys, n_episodes)
            env_metrics[env_name] = metrics
        
        # 3. Compute and log mean metrics across all environments
        self._log_mean_metrics(env_metrics)
        
        # 4. Log pre-computed teacher metrics (teachers are static, so we evaluated them once at initialization)
        self._log_teacher_metrics()
        
        print("Evaluation complete!")

    def _evaluate_teachers_once(self, n_episodes: int):
        """Evaluate each teacher once and return their metrics (since teachers are static)."""
        print(f"  Evaluating teachers on their specific configurations...")
        teacher_metrics = {}
        
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
                    # Use deterministic seed for teacher evaluation reproducibility
                    eval_seed = self.config.seed + 10000 + i * 1000 + episode
                    obs, _ = self.env.reset(seed=eval_seed)
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
                            
                            if isinstance(self.env.action_space, gym.spaces.Discrete):
                                # For discrete actions, use argmax for deterministic evaluation
                                if hasattr(distribution.distribution, 'logits'):
                                    action_probs = F.softmax(distribution.distribution.logits, dim=-1)
                                    action = torch.argmax(action_probs).item()
                                else:
                                    action_probs = distribution.distribution.probs
                                    action = torch.argmax(action_probs).item()
                            else:
                                # For continuous actions, use mean for deterministic evaluation
                                if hasattr(distribution.distribution, 'loc'):
                                    action = distribution.distribution.loc.squeeze(0).cpu().numpy()
                                else:
                                    action = distribution.mean.squeeze(0).cpu().numpy()
                        
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
                
                # Store metrics instead of logging immediately
                teacher_metrics[env_name] = {
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'mean_length': mean_length
                }
        
        return teacher_metrics

    def _log_teacher_metrics(self):
        """Log pre-computed teacher metrics to wandb with current step."""
        if not hasattr(self, 'teacher_metrics') or not self.teacher_metrics:
            return
            
        if self.config.track and wandb.run is not None:
            # Log all teacher metrics with current step
            for env_name, metrics in self.teacher_metrics.items():
                metrics_to_log = {
                    f"teacher/{env_name}/mean_return": metrics['mean_return'],
                    f"teacher/{env_name}/std_return": metrics['std_return'],
                    f"teacher/{env_name}/mean_length": metrics['mean_length']
                }
                # Use current step if available
                step_to_use = getattr(self, 'current_step', None)
                if step_to_use is not None:
                    metrics_to_log['global_step'] = step_to_use
                    wandb.log(metrics_to_log, step=step_to_use)
                else:
                    wandb.log(metrics_to_log)

    def _evaluate_student_default(self, n_episodes: int):
        """Evaluate student on training environment with student keys only."""
        print(f"  Evaluating student default (training environment)...")
        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            # Use deterministic seed for student evaluation reproducibility  
            eval_seed = self.config.seed + 20000 + episode
            obs, _ = self.env.reset(seed=eval_seed)
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
                    
                    if self.is_discrete_action:
                        # For discrete actions, use argmax for deterministic evaluation
                        if hasattr(distribution.distribution, 'logits'):
                            action_probs = F.softmax(distribution.distribution.logits, dim=-1)
                            action = torch.argmax(action_probs).item()
                        else:
                            action_probs = distribution.distribution.probs
                            action = torch.argmax(action_probs).item()
                    else:
                        # For continuous actions, use mean for deterministic evaluation
                        if hasattr(distribution.distribution, 'loc'):
                            action = distribution.distribution.loc.squeeze(0).cpu().numpy()
                        else:
                            action = distribution.mean.squeeze(0).cpu().numpy()
                
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
            if step_to_use is not None:
                metrics_to_log['global_step'] = step_to_use
            print(f"    Logging to wandb: {list(metrics_to_log.keys())} at step {step_to_use}")
            if step_to_use is not None:
                wandb.log(metrics_to_log, step=step_to_use)
            else:
                wandb.log(metrics_to_log)
        else:
            print(f"    Wandb logging skipped: track={self.config.track}, wandb.run={wandb.run is not None}")
    
    def _evaluate_environment(self, env_name: str, teacher_keys, n_episodes: int):
        """Evaluate student policy on a specific teacher configuration.
        
        Returns:
            dict: Dictionary containing mean_return, std_return, mean_length
        """        
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            # Use deterministic seed for environment evaluation reproducibility
            eval_seed = self.config.seed + 30000 + hash(env_name) % 1000 + episode
            obs, _ = self.env.reset(seed=eval_seed)
            episode_return = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Convert observations to tensors for masking
                obs_tensors = self._obs_to_tensors(obs)
                
                # Get student action (with masking)
                with torch.no_grad():
                    student_params = self.get_student_distribution_params(obs_tensors, env_name)
                    if self.is_discrete_action:
                        # For discrete actions, use argmax for deterministic evaluation
                        student_action = torch.argmax(student_params).item()
                    else:
                        # For continuous actions, use mean for deterministic evaluation
                        action_dim = self.action_space.shape[0]
                        student_action = student_params[:action_dim].cpu().numpy()
                
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
        
        # Log individual environment metrics to wandb
        if self.config.track and wandb.run is not None:
            metrics_to_log = {
                f"full_eval_return/{env_name}/mean_return": mean_return,
                f"full_eval/{env_name}/std_return": std_return,
                f"full_eval/{env_name}/mean_length": mean_length
            }
            # Use current step if available, otherwise let wandb use its internal counter
            step_to_use = getattr(self, 'current_step', None)
            if step_to_use is not None:
                metrics_to_log['global_step'] = step_to_use
                wandb.log(metrics_to_log, step=step_to_use)
            else:
                wandb.log(metrics_to_log)
        else:
            print(f"    Wandb logging skipped: track={self.config.track}, wandb.run={wandb.run is not None}")
        
        # Return metrics for aggregation
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length
        }

    def _log_mean_metrics(self, env_metrics: dict):
        """Compute and log mean metrics across all evaluation environments."""
        if not env_metrics:
            return
        
        # Compute means across all environments
        mean_mean_return = sum(metrics['mean_return'] for metrics in env_metrics.values()) / len(env_metrics)
        mean_std_return = sum(metrics['std_return'] for metrics in env_metrics.values()) / len(env_metrics)
        mean_mean_length = sum(metrics['mean_length'] for metrics in env_metrics.values()) / len(env_metrics)
        
        print(f"    env_mean: mean_return={mean_mean_return:.2f}, std_return={mean_std_return:.2f}, mean_length={mean_mean_length:.1f}")
        
        # Log mean metrics to wandb
        if self.config.track and wandb.run is not None:
            mean_metrics_to_log = {
                f"full_eval_return/env_mean/mean_return": mean_mean_return,
                f"full_eval/env_mean/std_return": mean_std_return,
                f"full_eval/env_mean/mean_length": mean_mean_length
            }
            # Use current step if available
            step_to_use = getattr(self, 'current_step', None)
            if step_to_use is not None:
                mean_metrics_to_log['global_step'] = step_to_use
                wandb.log(mean_metrics_to_log, step=step_to_use)
            else:
                wandb.log(mean_metrics_to_log)


def train_simple_imitation(config, expert_policy_dir: str, device: str = 'cpu', debug: bool = False):
    """Main training function for simple imitation learning."""
    # Set random seed for reproducibility
    print(f"🔧 Setting random seed: {config.seed}")
    set_seed(config.seed)
    
    # Handle torch deterministic setting
    if hasattr(config, 'torch_deterministic') and config.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🔧 PyTorch deterministic mode enabled")
    
    # Debug: Check what exp_name actually is
    print(f"🔍 DEBUG: config.exp_name = '{config.exp_name}'")
    print(f"🔍 DEBUG: Expected exp_name = 'tigerdoorkey_simple_imitation'")
    
    # Initialize wandb
    if config.track:
        wandb_run_name = f"simple_imitation_{config.exp_name}_{config.seed}"
        print(f"🔍 DEBUG: wandb run name = '{wandb_run_name}'")
        
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=wandb_run_name,
            config=vars(config)
        )
    
    # Create trainer and train
    trainer = SimpleImitationTrainer(config, expert_policy_dir, device, debug)
    trainer.train()
    
    if config.track and wandb.run is not None:
        wandb.finish() 