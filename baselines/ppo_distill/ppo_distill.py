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

from baselines.ppo_distill.agent import PPODistillAgent


class PPODistillTrainer:
    """PPO Distill Trainer that handles distillation from expert policies."""
    
    def __init__(self, envs, config, expert_policy_dir: str, device: str = 'cpu', student_policy_type: str = "ppo_rnn"):
        """
        Initialize the PPO Distill trainer.
        
        Args:
            envs: Environment
            config: Configuration
            expert_policy_dir: Directory containing expert subset policies
            device: Device to use
            student_policy_type: Type of student agent ("ppo" or "ppo_rnn")
        """
        self.envs = envs
        self.config = config
        self.device = device
        self.student_policy_type = student_policy_type
        
        # Create the appropriate base trainer based on student_policy_type
        if student_policy_type == "ppo":
            from baselines.ppo.ppo import PPOTrainer
            self.base_trainer = PPOTrainer(envs, config, seed=getattr(config, 'seed', 42))
        elif student_policy_type == "ppo_rnn":
            from baselines.ppo_rnn.ppo_rnn import PPORnnTrainer
            self.base_trainer = PPORnnTrainer(envs, config, seed=getattr(config, 'seed', 42))
        else:
            raise ValueError(f"Unknown student policy type: {student_policy_type}")
        
        # Create PPO Distill agent
        self.agent = PPODistillAgent(envs, config, expert_policy_dir, device, student_policy_type)
        
        # Copy all attributes from the base trainer
        for attr_name in dir(self.base_trainer):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.base_trainer, attr_name)
                if not callable(attr_value) or attr_name in ['collect_rollout', 'update_policy', 'train']:
                    setattr(self, attr_name, attr_value)
        
        # Distillation parameters
        self.distill_coef = getattr(config, 'distill_coef', 0.1)
        self.expert_coef = getattr(config, 'expert_coef', 0.5)
        
        # Cycling parameters
        self.cycle_mode = getattr(config, 'cycle_mode', 'episode')
        
        # Episode tracking for cycling
        self.episodes_completed = 0
        self.last_episode_count = 0
        
        # Initialize LSTM states for PPO-RNN
        if student_policy_type == "ppo_rnn":
            self.lstm_hidden = torch.zeros(
                self.agent.lstm.num_layers, 
                self.config.num_envs, 
                self.agent.lstm.hidden_size, 
                device=self.device
            )
            self.lstm_cell = torch.zeros(
                self.agent.lstm.num_layers, 
                self.config.num_envs, 
                self.agent.lstm.hidden_size, 
                device=self.device
            )
            self.next_lstm_state = (self.lstm_hidden, self.lstm_cell)
        else:
            self.lstm_hidden = None
            self.lstm_cell = None
            self.next_lstm_state = None
        
        # Configuration tracking
        self.current_config_name = None
        self.episode_configs = []
        
    def collect_rollout(self):
        """Collect a rollout with configuration cycling and expert action collection."""
        print("🚀 PPODistillTrainer.collect_rollout called!")
        
        # Initialize observation storage if not already done
        if not self.obs:
            # Initialize observation storage for all keys
            for key in self.agent.mlp_keys + self.agent.cnn_keys:
                if key in self.envs.obs_space:
                    if len(self.envs.obs_space[key].shape) == 3 and self.envs.obs_space[key].shape[-1] == 3:  # Image observations
                        self.obs[key] = torch.zeros(
                            (self.config.num_steps, self.config.num_envs) + self.envs.obs_space[key].shape,
                            dtype=torch.float32,
                            device=self.device
                        )
                    else:  # Non-image observations
                        size = np.prod(self.envs.obs_space[key].shape)
                        self.obs[key] = torch.zeros(
                            (self.config.num_steps, self.config.num_envs, size),
                            dtype=torch.float32,
                            device=self.device
                        )
        
        # Get initial observations using the same logic as thesis
        action_shape = self.envs.act_space['action'].shape
        acts = {
            'action': np.zeros((self.config.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(self.config.num_envs, dtype=bool)
        }
        obs_dict = self.envs.step(acts)
        
        # Process initial observations
        next_obs = {}
        for key in self.agent.mlp_keys + self.agent.cnn_keys:
            if key in obs_dict:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Store observations for all keys
            for key in self.agent.mlp_keys + self.agent.cnn_keys:
                if key in next_obs:
                    self.obs[key][step] = next_obs[key]
            self.dones[step] = next_done
            
            # Store LSTM states (only for PPO-RNN)
            if self.student_policy_type == "ppo_rnn" and self.next_lstm_state is not None:
                self.lstm_hidden[step] = self.next_lstm_state[0].transpose(0, 1)
                self.lstm_cell[step] = self.next_lstm_state[1].transpose(0, 1)
            
            # Get current configuration
            config_name, _ = self.agent.get_current_config()
            
            # Debug logging for configuration cycling
            if hasattr(self, '_last_config_log') and self._last_config_log != config_name:
                print(f"🔄 Training with configuration: {config_name}")
                self._last_config_log = config_name
            elif not hasattr(self, '_last_config_log'):
                self._last_config_log = config_name
                print(f"🔄 Starting training with configuration: {config_name}")
            
            # Get action and value with LSTM state, plus expert actions
            with torch.no_grad():
                result = self.agent.get_action_and_value(
                    next_obs, self.next_lstm_state, next_done
                )
                # Handle the extra expert_actions return value
                if len(result) == 6:
                    action, logprob, entropy, value, self.next_lstm_state, expert_actions = result
                else:
                    action, logprob, entropy, value, self.next_lstm_state = result
                    expert_actions = {}
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
            for key in self.agent.mlp_keys + self.agent.cnn_keys:
                if key in obs_dict:
                    next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
            self.rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(self.device).view(-1)
            
            # Cycle configuration if episode is done
            if self.cycle_mode == 'episode':
                episode_done = next_done.any().item()
                print(f"🔍 Checking episode done: {episode_done}")
                if episode_done:
                    self.episodes_completed += 1
                    print(f"📺 Episode {self.episodes_completed} completed, cycling from {self.agent.current_config_name}")
                    self.agent.cycle_config(episode_done=True)
                    print(f"✅ Cycled to configuration: {self.agent.current_config_name} (episode {self.episodes_completed})")
        
        # Cycle configuration for batch mode
        if self.cycle_mode == 'batch':
            print(f"🔄 Batch mode cycling from {self.agent.current_config_name}")
            self.agent.cycle_config()
            print(f"✅ Cycled to configuration: {self.agent.current_config_name}")
        
        # Store final observations and done state
        self.next_obs = next_obs
        self.next_done = next_done

    def compute_advantages(self):
        """Compute advantages and returns for the collected rollout."""
        with torch.no_grad():
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
        """Update the policy using PPO with distillation loss."""
        # Optimizing the policy and value network
        assert self.config.num_envs % self.config.num_minibatches == 0
        envsperbatch = self.config.num_envs // self.config.num_minibatches
        envinds = np.arange(self.config.num_envs)
        flatinds = np.arange(self.config.num_envs * self.config.num_steps).reshape(self.config.num_steps, self.config.num_envs)
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, self.config.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()
                
                # Prepare mini-batch observations
                mb_obs = {}
                for key in b_obs:
                    mb_obs[key] = b_obs[key][mb_inds]
                
                # Get initial LSTM state for this mini-batch
                # Use the stored LSTM states from the beginning of the rollout
                if self.student_policy_type == "ppo_rnn":
                    initial_lstm_state = (
                        self.lstm_hidden[0, mbenvinds].transpose(0, 1).clone(),
                        self.lstm_cell[0, mbenvinds].transpose(0, 1).clone()
                    )
                else:
                    initial_lstm_state = None
                
                result = self.agent.get_action_and_value(
                    mb_obs,
                    initial_lstm_state,
                    b_actions.long()[mb_inds] if hasattr(self.agent, 'is_discrete') and self.agent.is_discrete else b_actions[mb_inds],
                    b_actions.long()[mb_inds] if hasattr(self.agent, 'is_discrete') and self.agent.is_discrete else b_actions[mb_inds],
                )
                # Handle the extra expert_actions return value
                if len(result) == 6:
                    _, newlogprob, entropy, newvalue, _, expert_actions = result
                else:
                    _, newlogprob, entropy, newvalue, _ = result
                    expert_actions = {}
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Distillation loss (simplified - we'll add this later)
                distill_loss = torch.tensor(0.0, device=self.device)
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef + self.distill_coef * distill_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.base_agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            if hasattr(self.config, 'target_kl') and self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break
        
        # Log metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        self.log_metrics({
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/distill_loss": distill_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
        })

    def train(self, num_iterations: int = None):
        """
        Train the PPO Distill agent.
        
        Args:
            num_iterations: Number of training iterations (optional, overrides config)
        """
        print("Starting PPO Distill training...")
        print(f"Expert policy directory: {self.agent.expert_manager.policy_dir}")
        print(f"Student policy type: {self.student_policy_type}")
        print(f"Cycle mode: {self.cycle_mode}")
        print(f"Distillation coefficient: {self.distill_coef}")
        print(f"Expert coefficient: {self.expert_coef}")
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
        from baselines.ppo_rnn.train import make_envs
        eval_envs, _ = run_initial_evaluation(
            self.agent, self.config, self.device, make_envs, 
            writer=self.writer, use_wandb=self.use_wandb
        )
        
        for iteration in range(self.config.num_iterations):
            # Annealing the learning rate if instructed
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect experience using our custom collect_rollout with cycling
            self.collect_rollout()
            
            # Compute advantages
            advantages, returns = self.compute_advantages()
            
            # Flatten the batch
            b_obs = {}
            for key in self.obs:
                b_obs[key] = self.obs[key].reshape((-1,) + self.obs[key].shape[2:])
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.act_space['action'].shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)
            
            # Update policy using our custom update_policy with distillation
            self.update_policy(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
            
            # Periodic evaluation using shared utility
            self.last_eval, eval_envs = run_periodic_evaluation(
                self.agent, self.config, self.device, self.global_step, self.last_eval, eval_envs,
                make_envs_func=make_envs, writer=self.writer, use_wandb=self.use_wandb
            )
            
            # Print progress with cycling info
            if iteration % getattr(self.config, 'log_interval', 10) == 0:
                print(f"Iteration {iteration}/{self.config.num_iterations}, Global step: {self.global_step}")
                print(f"  Episodes completed: {self.episodes_completed}")
                print(f"  Current config: {self.agent.current_config_name}")
                print(f"  Average return (last rollout): {b_returns.mean().item():.2f}")
                print(f"  Average advantage: {b_advantages.mean().item():.2f}")
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"  SPS: {int(self.global_step / (time.time() - self.start_time))}")
                print("-" * 50)
            
            # Log training metrics with cycling info
            self.log_metrics({
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                "charts/avg_return": b_returns.mean().item(),
                "charts/avg_advantage": b_advantages.mean().item(),
                "charts/SPS": int(self.global_step / (time.time() - self.start_time)),
                "cycling/episodes_completed": self.episodes_completed,
                "cycling/current_config": self.agent.current_config_name,
                "cycling/config_cycle_count": self.agent.config_scheduler.episode_count,
            })
        
        # Print cycling summary at the end
        print("\n" + "="*60)
        print("🎯 PPO DISTILL TRAINING SUMMARY")
        print("="*60)
        print(f"Total episodes completed: {self.episodes_completed}")
        print(f"Configuration cycles: {self.agent.config_scheduler.episode_count}")
        print(f"Expert configurations used: {list(self.agent.expert_manager.expert_policies.keys())}")
        print(f"Final configuration: {self.agent.current_config_name}")
        print(f"Student policy type: {self.student_policy_type}")
        print(f"Cycle mode: {self.cycle_mode}")
        print("="*60)
        
        # Clean up
        self.envs.close()
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return self.agent


def train_ppo_distill(envs, config, seed: int, expert_policy_dir: str, student_policy_type: str = "ppo_rnn", num_iterations: int = 1000):
    """
    Train a PPO Distill agent.
    
    Args:
        envs: Environment
        config: Configuration
        seed: Random seed
        expert_policy_dir: Directory containing expert subset policies
        student_policy_type: Type of student agent to distill into ("ppo" or "ppo_rnn")
        num_iterations: Number of training iterations
        
    Returns:
        PPODistillAgent: Trained agent
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create trainer with the specified student policy type
    trainer = PPODistillTrainer(envs, config, expert_policy_dir, device='cuda' if torch.cuda.is_available() else 'cpu', student_policy_type=student_policy_type)
    
    # Train
    trainer.train()
    
    return trainer.agent 