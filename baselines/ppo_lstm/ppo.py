import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from .agent import PPOLSTMAgent
from baselines.shared.eval_utils import (
    evaluate_agent, 
    evaluate_agent_with_observation_subsets,
    run_periodic_evaluation, 
    run_initial_evaluation
)

class PPOLSTMTrainer:
    def __init__(self, envs, config, seed):
        self.config = config
        self.envs = envs
        self.seed = seed
        
        # Initialize agent
        self.agent = PPOLSTMAgent(envs, config)
        self.device = self.agent.device
        self.agent = self.agent.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config.learning_rate,
            eps=config.eps
        )
        
        # Initialize storage - will be set up properly in collect_rollout
        self.obs = {}
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
        self.actions = torch.zeros(
            (config.num_steps, config.num_envs) + envs.act_space['action'].shape,
            dtype=torch.float32,
            device=self.device
        )
        self.logprobs = torch.zeros(
            (config.num_steps, config.num_envs),
            dtype=torch.float32,
            device=self.device
        )
        self.rewards = torch.zeros(
            (config.num_steps, config.num_envs),
            dtype=torch.float32,
            device=self.device
        )
        self.dones = torch.zeros(
            (config.num_steps, config.num_envs),
            dtype=torch.float32,
            device=self.device
        )
        self.values = torch.zeros(
            (config.num_steps, config.num_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        # Initialize LSTM states storage
        self.lstm_states = torch.zeros(
            (config.num_steps, config.num_envs, 2, config.lstm.hidden_size),  # 2 for hidden and cell states
            dtype=torch.float32,
            device=self.device
        )
        
        # Initialize logging
        self.writer = SummaryWriter(f"runs/ppo_lstm_{config.task}_{seed}")
        
        # Initialize wandb
        if hasattr(config, 'use_wandb') and config.use_wandb:
            wandb.init(
                project=getattr(config, 'wandb_project', 'sensor-dropout'),
                name=f"ppo_lstm_{config.task}_{seed}",
                config=vars(config),
                monitor_gym=False,
                save_code=True,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Training stats
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # Evaluation tracking
        self.last_eval = 0

    def log_metrics(self, metrics_dict, step=None):
        """Log metrics to both TensorBoard and wandb."""
        if step is None:
            step = self.global_step
            
        # Log to TensorBoard
        for key, value in metrics_dict.items():
            self.writer.add_scalar(key, value, step)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)

    def collect_rollout(self):
        """Collect a rollout of experience with LSTM states."""
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
        
        # Initialize LSTM states
        next_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, self.config.num_envs, self.agent.lstm.hidden_size).to(self.device),
            torch.zeros(self.agent.lstm.num_layers, self.config.num_envs, self.agent.lstm.hidden_size).to(self.device),
        )
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Store observations for all keys
            for key in self.agent.mlp_keys + self.agent.cnn_keys:
                if key in next_obs:
                    self.obs[key][step] = next_obs[key]
            self.dones[step] = next_done
            
            # Store LSTM states
            self.lstm_states[step, :, 0] = next_lstm_state[0].squeeze(0)  # Hidden state
            self.lstm_states[step, :, 1] = next_lstm_state[1].squeeze(0)  # Cell state
            
            # Get action and value with LSTM
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = self.agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done
                )
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
            
            # Track episode info
            for i, d in enumerate(obs_dict['is_last']):
                if d:
                    self.episode_count += 1
                    if 'episode' in obs_dict and i < len(obs_dict.get('episode', [])):
                        episode_reward = obs_dict['episode'][i]['r']
                        episode_length = obs_dict['episode'][i]['l']
                        
                        # Log episode metrics
                        self.log_metrics({
                            "charts/episodic_return": episode_reward,
                            "charts/episodic_length": episode_length
                        }, step=self.episode_count)
                        
                        # Print episode info
                        print(f"Episode {self.episode_count}: Return={episode_reward:.2f}, Length={episode_length}")
        
        # Compute advantages and returns
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # Flatten the batch - handle multiple observation keys properly
        b_obs = {}
        for key in self.agent.mlp_keys + self.agent.cnn_keys:
            if key in next_obs:
                b_obs[key] = self.obs[key].reshape(-1, *self.obs[key].shape[2:])
        
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.act_space['action'].shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_lstm_states = self.lstm_states.reshape(-1, 2, self.config.lstm.hidden_size)
        
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_lstm_states

    def reconstruct_lstm_states(self, mb_inds, b_lstm_states, b_dones):
        """Reconstruct LSTM states for a minibatch by running through the sequence."""
        # Get the indices in the original rollout order
        rollout_indices = mb_inds // self.config.num_envs  # Which step in the rollout
        env_indices = mb_inds % self.config.num_envs       # Which environment
        
        # Sort by rollout step to maintain sequence order
        sorted_indices = np.argsort(rollout_indices)
        mb_inds_sorted = mb_inds[sorted_indices]
        rollout_indices_sorted = rollout_indices[sorted_indices]
        env_indices_sorted = env_indices[sorted_indices]
        
        # Initialize LSTM states for this minibatch
        mb_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, len(mb_inds), self.config.lstm.hidden_size, device=self.device),
            torch.zeros(self.agent.lstm.num_layers, len(mb_inds), self.config.lstm.hidden_size, device=self.device)
        )
        
        # For each unique rollout step, reconstruct the LSTM states
        unique_steps = np.unique(rollout_indices_sorted)
        for step in unique_steps:
            # Find all indices for this step
            step_mask = rollout_indices_sorted == step
            step_mb_inds = mb_inds_sorted[step_mask]
            step_env_indices = env_indices_sorted[step_mask]
            
            # Get the stored LSTM states for this step
            step_lstm_states = b_lstm_states[step_mb_inds]
            
            # Update the LSTM states for this step
            for i, (mb_idx, env_idx) in enumerate(zip(step_mb_inds, step_env_indices)):
                mb_lstm_state[0][:, i] = step_lstm_states[i, 0]  # Hidden state
                mb_lstm_state[1][:, i] = step_lstm_states[i, 1]  # Cell state
        
        return mb_lstm_state

    def update_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_lstm_states):
        """Update the policy using PPO with LSTM."""
        # Calculate batch size like the thesis
        batch_size = int(self.config.num_envs * self.config.num_steps)
        minibatch_size = int(batch_size // self.config.num_minibatches)
        
        # Optimize the policy and value function
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        # Save initial LSTM state for reconstruction
        initial_lstm_state = (
            b_lstm_states[0, 0].unsqueeze(0).unsqueeze(0).expand(
                self.agent.lstm.num_layers, self.config.num_envs, self.config.lstm.hidden_size
            ),
            b_lstm_states[0, 1].unsqueeze(0).unsqueeze(0).expand(
                self.agent.lstm.num_layers, self.config.num_envs, self.config.lstm.hidden_size
            )
        )
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                # Create minibatch observations dictionary
                mb_obs = {}
                for key in b_obs.keys():
                    mb_obs[key] = b_obs[key][mb_inds]
                
                # Reconstruct LSTM states for this minibatch
                mb_lstm_state = self.reconstruct_lstm_states(mb_inds, b_lstm_states, self.dones.reshape(-1))
                
                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                    mb_obs, mb_lstm_state, torch.zeros(mb_inds.shape[0], device=self.device), b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
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
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break
        
        # Log training info
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log all metrics
        self.log_metrics({
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(self.global_step / (time.time() - self.start_time))
        })

    def train(self):
        """Main training loop."""
        self.start_time = time.time()
        
        # Run initial evaluation using shared utility
        from baselines.ppo_lstm.train import make_envs
        eval_envs, _ = run_initial_evaluation(
            self.agent, self.config, self.device, make_envs, 
            writer=self.writer, use_wandb=self.use_wandb
        )
        
        for iteration in range(self.config.num_iterations):
            # Collect rollout
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_lstm_states = self.collect_rollout()
            
            # Update policy
            self.update_policy(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_lstm_states)
            
            # Periodic evaluation using shared utility
            self.last_eval, eval_envs = run_periodic_evaluation(
                self.agent, self.config, self.device, self.global_step, self.last_eval, eval_envs,
                make_envs_func=make_envs, writer=self.writer, use_wandb=self.use_wandb
            )
            
            # Log iteration info
            if iteration % self.config.log_interval == 0:
                print(f"Iteration {iteration}/{self.config.num_iterations}")
                print(f"  Global step: {self.global_step}")
                print(f"  Episodes completed: {self.episode_count}")
                print(f"  Average return (last rollout): {b_returns.mean().item():.2f}")
                print(f"  Average advantage: {b_advantages.mean().item():.2f}")
                print("-" * 50)
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return self.agent

def train_ppo_lstm(envs, config, seed, num_iterations=None):
    """Main function to train PPO LSTM agent."""
    trainer = PPOLSTMTrainer(envs, config, seed)
    
    # Override num_iterations if provided
    if num_iterations is not None:
        trainer.config.num_iterations = num_iterations
    
    return trainer.train() 