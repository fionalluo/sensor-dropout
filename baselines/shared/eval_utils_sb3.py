#!/usr/bin/env python3
"""
Evaluation utilities for SB3-based training scripts.
Contains custom evaluation callbacks and related functions.
"""

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from baselines.shared.masking_utils import mask_observations_for_student


class CustomEvalCallback(BaseCallback):
    """Custom evaluation callback that evaluates across different observation subsets."""
    
    def __init__(
        self,
        eval_env,
        make_eval_env_func,
        expert_manager,
        student_keys,
        teacher_keys_by_config,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        debug=False
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.make_eval_env_func = make_eval_env_func
        self.expert_manager = expert_manager
        self.student_keys = student_keys
        self.teacher_keys_by_config = teacher_keys_by_config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.debug = debug
        self.last_eval = 0
        
        if self.debug:
            print(f"[EVAL CALLBACK] Student keys: {sorted(self.student_keys)}")
            print(f"[EVAL CALLBACK] Teacher configs: {list(self.teacher_keys_by_config.keys())}")
    
    def _on_step(self):
        """Called after each step."""
        # Check if we should run evaluation
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            print(f"Running custom evaluation at step {self.num_timesteps}...")
            
            # First, evaluate teacher policies (baselines)
            print("📚 Evaluating teacher policies...")
            self._evaluate_teacher_policies()
            
            # Then, evaluate student policy on different configurations
            print("🎓 Evaluating student policy...")
            for env_name, teacher_keys in self.teacher_keys_by_config.items():
                if self.debug:
                    print(f"[EVAL CALLBACK] {env_name} - Teacher keys: {sorted(teacher_keys)}")
                
                # Run evaluation for this environment
                self._evaluate_student_environment(env_name, teacher_keys)
            
            self.last_eval = self.num_timesteps
            
        return True
    
    def _evaluate_teacher_policies(self):
        """Evaluate each teacher policy on their respective configuration."""
        # Import here to avoid circular imports
        from baselines.ppo_distill_sb3.ppo_distill_sb3 import ObservationFilterWrapper
        
        for env_name, teacher_keys in self.teacher_keys_by_config.items():
            # Check if teacher policy exists
            if env_name not in self.expert_manager.expert_policies:
                print(f"Warning: Teacher policy {env_name} not found")
                continue
            
            # Get teacher policy
            teacher_agent = self.expert_manager.expert_policies[env_name]
            
            # Create environment with teacher's observation filtering
            # For teacher evaluation, we need to create a filtered environment using patterns
            # since the teacher was trained with filtered observations
            eval_env = self.make_eval_env_func()
            
            # Get the regex patterns for this teacher from expert manager
            if env_name in self.expert_manager.expert_eval_keys:
                eval_keys = self.expert_manager.expert_eval_keys[env_name]
                mlp_keys_pattern = eval_keys.get('mlp_keys', '.*')
                cnn_keys_pattern = eval_keys.get('cnn_keys', '.*')
                # Apply teacher's specific observation filtering (not student masking!)
                eval_env = ObservationFilterWrapper(eval_env, mlp_keys_pattern, cnn_keys_pattern)
            else:
                print(f"Warning: No eval_keys found for teacher {env_name}, using unfiltered environment")
            
            episode_returns = []
            episode_lengths = []
            
            for episode in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                episode_return = 0
                episode_length = 0
                
                done = False
                truncated = False
                
                while not (done or truncated):
                    # Teacher uses its own filtered observations (no additional masking)
                    # Convert to format expected by SB3
                    obs_batch = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
                    
                    # Get action from teacher policy
                    with torch.no_grad():
                        action, _ = teacher_agent.predict(obs_batch, deterministic=self.deterministic)
                    
                    # Extract scalar action if it's a numpy array
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.size == 1 else action[0]
                    
                    # Step environment
                    obs, reward, done, truncated, info = eval_env.step(action)
                    
                    episode_return += reward
                    episode_length += 1
                
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
            
            # Compute teacher metrics
            episode_returns = np.array(episode_returns)
            episode_lengths = np.array(episode_lengths)
            
            mean_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            mean_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)
            
            print(f"  Teacher {env_name}: mean_return={mean_return:.2f}, std_return={std_return:.2f}, mean_length={mean_length:.1f}")
            
            # Log teacher performance
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.record(f"teacher_eval_return/{env_name}/mean_return", mean_return)
                self.logger.record(f"teacher_eval/{env_name}/std_return", std_return)
                self.logger.record(f"teacher_eval/{env_name}/mean_length", mean_length)
                self.logger.record(f"teacher_eval/{env_name}/std_length", std_length)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    f"teacher_eval_return/{env_name}/mean_return": mean_return,
                    f"teacher_eval/{env_name}/std_return": std_return,
                    f"teacher_eval/{env_name}/mean_length": mean_length,
                    f"teacher_eval/{env_name}/std_length": std_length
                })
            
            eval_env.close()
    
    def _evaluate_student_environment(self, env_name, teacher_keys):
        """Evaluate the student agent on a specific environment configuration."""
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
