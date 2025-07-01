#!/usr/bin/env python3
"""
PPO Distill Training Loop
Combines standard PPO objectives with distillation loss from expert policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import time
from collections import deque
import torch.optim as optim
import wandb

from baselines.ppo_distill.agent import PPODistillAgent
from baselines.shared.eval_utils import run_initial_evaluation, run_periodic_evaluation, evaluate_agent


class PPODistillTrainer:
    """PPO Distill Trainer that handles distillation from expert policies."""
    
    def __init__(self, envs, config, expert_policy_dir: str, device: str = 'cpu'):
        """
        Initialize the PPO Distill trainer.
        
        Args:
            envs: Environment
            config: Configuration
            expert_policy_dir: Directory containing expert subset policies
            device: Device to use
        """
        self.envs = envs
        self.config = config
        self.device = device
        
        # Create the base PPO trainer
        from baselines.ppo.ppo import PPOTrainer
        self.base_trainer = PPOTrainer(envs, config, seed=getattr(config, 'seed', 42))
        
        # Create PPO Distill agent
        self.agent = PPODistillAgent(envs, config, expert_policy_dir, device)
        
        # Copy all attributes from the base trainer (but exclude our custom methods and agent)
        for attr_name in dir(self.base_trainer):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.base_trainer, attr_name)
                # Only copy non-callable attributes or callable attributes that are NOT our custom methods
                if not callable(attr_value) or attr_name not in ['update_policy', 'get_action_and_value', 'collect_rollout', 'train', 'agent', 'log_metrics']:
                    setattr(self, attr_name, attr_value)
        

        
        # Cycling parameters
        self.cycle_mode = getattr(config, 'cycle_mode', 'episode')
        
        # Episode tracking for cycling
        self.episodes_completed = 0
        self.last_episode_count = 0
        

        
        # Configuration tracking
        self.current_config_name = None
        self.episode_configs = []
        
        # --- Compute the union of all required observation keys (student + all experts) ---
        self.all_required_keys = set(self.agent.mlp_keys + self.agent.cnn_keys)
        # Add all expert keys from eval_keys in config
        if hasattr(config, 'eval_keys'):
            for env_name in getattr(config, 'eval_keys').__dict__:
                eval_keys = getattr(config.eval_keys, env_name)
                # Parse mlp_keys and cnn_keys patterns into actual keys if possible
                for pattern in [eval_keys.mlp_keys, eval_keys.cnn_keys]:
                    # Remove regex boundaries and split by |
                    pattern_clean = pattern.replace('\\b', '').replace('(', '').replace(')', '')
                    for key in pattern_clean.split('|'):
                        if key and key != '^$' and key != '.*':
                            self.all_required_keys.add(key)
        # --- Use self.all_required_keys everywhere obs is constructed or updated ---

        # Initialize training storage (copied from base trainer)
        self.obs = {}
        for key in self.all_required_keys:
            if key in self.envs.obs_space:
                if len(self.envs.obs_space[key].shape) == 3 and self.envs.obs_space[key].shape[-1] == 3:
                    self.obs[key] = torch.zeros(
                        (self.config.num_steps, self.config.num_envs) + self.envs.obs_space[key].shape,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    size = np.prod(self.envs.obs_space[key].shape)
                    self.obs[key] = torch.zeros(
                        (self.config.num_steps, self.config.num_envs, size),
                        dtype=torch.float32,
                        device=self.device
                    )
        self.actions = torch.zeros((self.config.num_steps, self.config.num_envs) + self.envs.act_space['action'].shape, dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros(self.config.num_steps, self.config.num_envs, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.config.num_steps, self.config.num_envs, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.config.num_steps, self.config.num_envs, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.config.num_steps, self.config.num_envs, dtype=torch.float32, device=self.device)
        
        # Initialize next_obs and next_done (needed for compute_advantages)
        self.next_obs = {}
        self.next_done = torch.zeros(self.config.num_envs, dtype=torch.float32, device=self.device)
        
        # Initialize training state
        self.global_step = 0
        self.episode_count = 0
        self.last_eval = 0
        self.start_time = time.time()
        
        # Initialize optimizer (copy from base trainer)
        # self.optimizer = self.base_trainer.optimizer
        # FIX: Use the actual student model parameters for the optimizer
        self.optimizer = torch.optim.Adam(self.agent.base_agent.parameters(), lr=self.config.learning_rate)
        
        # Initialize logging (copy from base trainer)
        self.writer = self.base_trainer.writer
        self.use_wandb = self.base_trainer.use_wandb
        
    def collect_rollout(self):
        """Collect a rollout with configuration cycling and expert action collection."""
        # Remove excessive debug prints
        # print("🚀 PPODistillTrainer.collect_rollout called!")
        # print("🚀 About to initialize observation storage...")
        
        # Get initial observations using the same logic as thesis
        action_shape = self.envs.act_space['action'].shape
        acts = {
            'action': np.zeros((self.config.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(self.config.num_envs, dtype=bool)
        }
        obs_dict = self.envs.step(acts)
        
        # Process initial observations
        next_obs = {}
        for key in self.all_required_keys:
            if key in obs_dict:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Store observations for all keys
            for key in self.all_required_keys:
                if key in next_obs:
                    self.obs[key][step] = next_obs[key]
            self.dones[step] = next_done
            
            # Get current configuration
            config_name, _ = self.agent.get_current_config()
            
            # Debug logging for configuration cycling (only when config changes)
            if hasattr(self, '_last_config_log') and self._last_config_log != config_name:
                # Remove excessive debug print
                # print(f"🔄 Training with configuration: {config_name}")
                self._last_config_log = config_name
            elif not hasattr(self, '_last_config_log'):
                self._last_config_log = config_name
                # Remove excessive debug print
                # print(f"🔄 Starting training with configuration: {config_name}")
            
            # Get action and value, plus expert actions
            with torch.no_grad():
                result = self.agent.get_action_and_value(
                    next_obs,
                    None  # action=None for inference
                )
                
                # Handle the return format with student_logits
                if len(result) == 7:
                    action, logprob, entropy, value, _, expert_actions, student_logits = result
                else:
                    # Fallback for older version
                    action, logprob, entropy, value, _, expert_actions = result
                    student_logits = None
                self.values[step] = value.flatten()
            
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # Execute action
            action_np = action.cpu().numpy()
            if hasattr(self.agent, 'is_discrete') and self.agent.is_discrete:
                action_np = action_np.reshape(self.config.num_envs, -1)
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = self.envs.step(acts)
            
            # Process observations
            for key in self.all_required_keys:
                if key in obs_dict:
                    next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
            self.rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(self.device).view(-1)
            
            # Cycle configuration if episode is done
            if self.cycle_mode == 'episode':
                episode_done = next_done.any().item()
                # Remove excessive debug print
                # print(f"🔍 Checking episode done: {episode_done}")
                if episode_done:
                    self.episodes_completed += 1
                    # Only print episode completion every 100 episodes to reduce log spam
                    if self.episodes_completed % 100 == 0:
                        print(f"📺 Episode {self.episodes_completed} completed, cycling from {self.agent.current_config_name}")
                    self.agent.cycle_config(episode_done=True)
                    # Remove redundant print since cycle_config already prints
                    # print(f"✅ Cycled to configuration: {self.agent.current_config_name} (episode {self.episodes_completed})")
        
        # Cycle configuration for batch mode
        if self.cycle_mode == 'batch':
            print(f"🔄 Batch mode cycling from {self.agent.current_config_name}")
            self.agent.cycle_config()
            print(f"✅ Cycled to configuration: {self.agent.current_config_name}")
        
        # Store final observations and done state
        # Remove excessive debug prints
        # print(f"🔍 Processing next_obs: {list(next_obs.keys())}")
        # print(f"🔍 agent.mlp_keys: {self.agent.mlp_keys}")
        # print(f"🔍 agent.cnn_keys: {self.agent.cnn_keys}")
        
        processed_next_obs = {}
        for key in self.all_required_keys:
            if key in next_obs:
                processed_next_obs[key] = next_obs[key]
                # Remove excessive debug prints
                # print(f"🔍 Added key: {key}")
            else:
                # Remove excessive debug prints
                # print(f"🔍 Missing key: {key}")
                pass
        
        # Remove excessive debug print
        # print(f"🔍 Final processed_next_obs keys: {list(processed_next_obs.keys())}")
        self.next_obs = processed_next_obs
        self.next_done = next_done

    def compute_advantages(self):
        """Compute advantages and returns for the collected rollout.""" 
        with torch.no_grad():
            # Call get_value with PPO parameters
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values
        
        return advantages, returns

    def update_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        """Update the policy using pure distillation loss only, with debug prints for gradients and parameter updates."""
        assert self.config.num_envs % self.config.num_minibatches == 0
        envsperbatch = self.config.num_envs // self.config.num_minibatches
        envinds = np.arange(self.config.num_envs)
        flatinds = np.arange(self.config.num_envs * self.config.num_steps).reshape(self.config.num_steps, self.config.num_envs)

        for epoch in range(self.config.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, self.config.num_envs, envsperbatch):
                end = start + envsperbatch
                mb_envinds = envinds[start:end]
                mb_inds = flatinds[:, mb_envinds].ravel()

                self.optimizer.zero_grad()
                # Call get_action_and_value with PPO arguments
                _, _, _, _, _, expert_actions, student_logits = self.agent.get_action_and_value(b_obs, action=None)
                distill_loss = self.agent.compute_distillation_loss(student_logits, expert_actions)
                distill_loss.backward()

                self.optimizer.step()

        # Log metrics for distillation training
        self.log_metrics({
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/distill_loss": distill_loss.item(),
            "losses/total_loss": distill_loss.item(),
        })

    def log_metrics(self, metrics):
        """Log metrics to tensorboard and wandb."""
        for key, value in metrics.items():
            # Skip non-numeric values for tensorboard logging
            if isinstance(value, (int, float)) or (hasattr(value, 'item') and callable(value.item)):
                if self.writer:
                    self.writer.add_scalar(key, value, self.global_step)
                if self.use_wandb:
                    wandb.log({key: value, "global_step": self.global_step})
            else:
                # For non-numeric values, only log to wandb as text
                if self.use_wandb:
                    wandb.log({key: str(value), "global_step": self.global_step})

    def train(self, num_iterations: int = None):
        """
        Train the PPO Distill agent.
        
        Args:
            num_iterations: Number of training iterations (optional, overrides config)
        """
        print("🚀 PPODistillTrainer.train() called!")
        print("Starting PPO Distill training...")
        print(f"Expert policy directory: {self.agent.expert_manager.policy_dir}")
        print(f"Cycle mode: {self.cycle_mode}")

        print(f"Number of expert configurations: {len(self.agent.expert_manager.expert_policies)}")
        print(f"Expert configurations: {list(self.agent.expert_manager.expert_policies.keys())}")
        print("-" * 60)
        
        # Override num_iterations if provided
        if num_iterations is not None:
            self.config.num_iterations = num_iterations
            # Also update total_timesteps to match
            self.config.total_timesteps = num_iterations * self.config.num_envs * self.config.num_steps
        
        # Initialize training
        self.start_time = time.time()
        
        # Calculate num_iterations from total_timesteps if not set
        if not hasattr(self.config, 'num_iterations'):
            self.config.num_iterations = self.config.total_timesteps // (self.config.num_envs * self.config.num_steps)
        
        # Run initial evaluation using shared utility
        from baselines.ppo_distill.train import make_envs_ppo_distill
        eval_envs, _ = run_initial_evaluation(
            self.agent, self.config, self.device, make_envs_ppo_distill, 
            writer=self.writer, use_wandb=self.use_wandb
        )
        
        for iteration in range(self.config.num_iterations):
            print(f"🔄 Starting iteration {iteration}")
            
            # Annealing the learning rate if instructed
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect experience using our custom collect_rollout with cycling
            # print(f"🔄 About to call collect_rollout for iteration {iteration}")
            # print(f"🔄 collect_rollout method: {self.collect_rollout}")
            # print(f"🔄 collect_rollout method location: {self.collect_rollout.__module__}")
            self.collect_rollout()
            # print(f"🔄 Finished collect_rollout for iteration {iteration}")
            
            # For pure distillation, we don't need advantages or returns
            # We only need the observations and actions for distillation training
            
            # Flatten the batch
            b_obs = {}
            for key in self.obs:
                b_obs[key] = self.obs[key].reshape((-1,) + self.obs[key].shape[2:])
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.act_space['action'].shape)
            
            # Create dummy tensors for compatibility (not used in distillation)
            b_advantages = torch.zeros_like(b_logprobs)
            b_returns = torch.zeros_like(b_logprobs)
            b_values = self.values.reshape(-1)
            
            # Update policy using our custom update_policy with distillation
            self.update_policy(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
            
            # Print distillation progress
            if iteration % getattr(self.config, 'log_interval', 10) == 0:
                print(f"🎯 Distillation: mimicking {self.agent.current_config_name}")
                print(f"📊 Expert configurations loaded: {list(self.agent.expert_manager.expert_policies.keys())}")
            
            # Periodic evaluation using shared utility
            self.last_eval, eval_envs = run_periodic_evaluation(
                self.agent, self.config, self.device, self.global_step, self.last_eval, eval_envs,
                make_envs_func=make_envs_ppo_distill, writer=self.writer, use_wandb=self.use_wandb
            )

            # --- Evaluate all expert (teacher) policies ---
            for expert_config_name, expert_agent in self.agent.expert_manager.expert_policies.items():
                expert_eval_metrics = evaluate_agent(
                    expert_agent,
                    eval_envs,
                    self.device,
                    self.config,
                    log_video=False,
                    make_envs_func=make_envs_ppo_distill,
                    writer=None,  # Do not log teacher metrics to tensorboard
                    use_wandb=False,  # Prevent evaluate_agent from logging to wandb under 'eval/'
                    global_step=self.global_step
                )
                # Manually log all teacher metrics under eval_teacher/
                if self.use_wandb:
                    import wandb
                    for metric_key in ["mean_return", "std_return", "mean_length", "std_length"]:
                        if metric_key in expert_eval_metrics:
                            wandb.log({
                                f"eval_teacher/{metric_key}_{expert_config_name}": expert_eval_metrics[metric_key]
                            }, step=self.global_step)
            
            # Print progress with cycling info
            if iteration % getattr(self.config, 'log_interval', 10) == 0:
                print(f"Iteration {iteration}/{self.config.num_iterations}, Global step: {self.global_step}")
                print(f"  Episodes completed: {self.episodes_completed}")
                print(f"  Current config: {self.agent.current_config_name}")
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"  SPS: {int(self.global_step / (time.time() - self.start_time))}")
                print("-" * 50)
            
            # Log training metrics with cycling info
            self.log_metrics({
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                "charts/SPS": int(self.global_step / (time.time() - self.start_time)),
            })
        
        # Print cycling summary at the end
        print("\n" + "="*60)
        print("🎯 PPO DISTILL TRAINING SUMMARY")
        print("="*60)
        print(f"Total episodes completed: {self.episodes_completed}")
        print(f"Configuration cycles: {self.agent.config_scheduler.episode_count}")
        print(f"Expert configurations used: {list(self.agent.expert_manager.expert_policies.keys())}")
        print(f"Final configuration: {self.agent.current_config_name}")
        print(f"Cycle mode: {self.cycle_mode}")
        print("="*60)
        
        # Clean up
        self.envs.close()
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return self.agent


def train_ppo_distill(envs, config, seed: int, expert_policy_dir: str, num_iterations: int = 1000):
    """
    Train a PPO Distill agent.
    
    Args:
        envs: Environment
        config: Configuration
        seed: Random seed
        expert_policy_dir: Directory containing expert subset policies
        num_iterations: Number of training iterations
        
    Returns:
        PPODistillAgent: Trained agent
    """
    print("🚀 train_ppo_distill function called!")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create trainer
    print("🚀 Creating PPODistillTrainer...")
    trainer = PPODistillTrainer(envs, config, expert_policy_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 PPODistillTrainer created successfully!")
    
    # Train
    print("🚀 About to call trainer.train()...")
    print(f"🚀 Trainer type: {type(trainer)}")
    print(f"🚀 Trainer train method: {trainer.train}")
    trainer.train()
    print("🚀 trainer.train() completed!")
    
    return trainer.agent 