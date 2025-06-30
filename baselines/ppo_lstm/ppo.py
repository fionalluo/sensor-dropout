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
        super().__init__()
        self.config = config
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        
        # Initialize agent
        self.agent = PPOLSTMAgent(envs, config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=config.eps)
        
        # Initialize logging
        self.writer = SummaryWriter(f"runs/ppo_lstm_{config.task}_{seed}")
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            if not hasattr(config, 'wandb_project') or config.wandb_project is None:
                raise ValueError("wandb_project must be set in config when use_wandb is True")
            
            wandb.init(
                project=config.wandb_project,
                name=f"ppo_lstm_{config.task}_{seed}",
                config=vars(config),
                monitor_gym=True,
                save_code=True,
            )
        
        # Initialize environment
        self.envs = envs
        
        # Initialize storage
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
        
        # LSTM state storage - store both hidden and cell states
        self.lstm_hidden = torch.zeros(
            (config.num_steps, config.num_envs, self.agent.lstm.num_layers, self.agent.lstm.hidden_size),
            dtype=torch.float32,
            device=self.device
        )
        self.lstm_cell = torch.zeros(
            (config.num_steps, config.num_envs, self.agent.lstm.num_layers, self.agent.lstm.hidden_size),
            dtype=torch.float32,
            device=self.device
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.last_eval = 0
        self.start_time = None
        
        # Store initial LSTM state for training
        self.initial_lstm_state = None
        self.next_lstm_state = None

    def reset_lstm_states(self, done_flags):
        """Reset LSTM states for environments that are done.
        
        Args:
            done_flags: Tensor of done flags for each environment
        """
        if self.next_lstm_state is not None:
            # Reset LSTM states for done environments
            for i, done in enumerate(done_flags):
                if done:
                    # Reset both hidden and cell states to zeros
                    self.next_lstm_state = (
                        self.next_lstm_state[0].clone(),
                        self.next_lstm_state[1].clone()
                    )
                    self.next_lstm_state[0][:, i, :] = 0
                    self.next_lstm_state[1][:, i, :] = 0

    def log_metrics(self, metrics):
        """Log metrics to tensorboard and wandb."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
        
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

    def collect_rollout(self):
        """Collect a rollout of experience."""
        # Get initial observations
        obs_dict = self.envs.step({
            'action': np.zeros((self.config.num_envs,) + self.envs.act_space['action'].shape, dtype=np.float32),
            'reset': np.ones(self.config.num_envs, dtype=bool)
        })
        
        next_obs = {}
        for key in self.agent.mlp_keys + self.agent.cnn_keys:
            if key in obs_dict:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
        
        # Initialize LSTM states
        initial_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, self.config.num_envs, self.agent.lstm.hidden_size).to(self.device),
            torch.zeros(self.agent.lstm.num_layers, self.config.num_envs, self.agent.lstm.hidden_size).to(self.device),
        )
        self.next_lstm_state = initial_lstm_state
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Store observations for all keys
            for key in self.agent.mlp_keys + self.agent.cnn_keys:
                if key in next_obs:
                    self.obs[key][step] = next_obs[key]
            self.dones[step] = next_done
            
            # Store LSTM states
            self.lstm_hidden[step] = self.next_lstm_state[0].transpose(0, 1)
            self.lstm_cell[step] = self.next_lstm_state[1].transpose(0, 1)
            
            # Get action and value with LSTM
            with torch.no_grad():
                action, logprob, _, value, self.next_lstm_state = self.agent.get_action_and_value(
                    next_obs, self.next_lstm_state, next_done
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
            
            # Process rewards and dones
            self.rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(self.device)
            next_done = torch.tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
            
            # Process next observations
            next_obs = {}
            for key in self.agent.mlp_keys + self.agent.cnn_keys:
                if key in obs_dict:
                    next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
            
            # CRITICAL FIX: Reset LSTM states for environments that are done
            self.reset_lstm_states(next_done)
            
            # Track episode completions
            for i, d in enumerate(next_done):
                if d:
                    self.episode_count += 1
        
        # Store final observations and done flags for bootstrapping
        self.next_obs = next_obs
        self.next_done = next_done
        
        # Store initial LSTM state for training
        self.initial_lstm_state = initial_lstm_state
        
        # Log rollout statistics
        avg_reward = self.rewards.mean().item()
        avg_value = self.values.mean().item()
        self.log_metrics({
            "rollout/avg_reward": avg_reward,
            "rollout/avg_value": avg_value,
            "rollout/episode_count": self.episode_count,
        })
        
        # Compute advantages and returns
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs, self.next_lstm_state, next_done).reshape(1, -1)
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

        # Reshape for training
        b_obs = {}
        for key in self.obs:
            b_obs[key] = self.obs[key].reshape((-1,) + self.obs[key].shape[2:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.act_space['action'].shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def update_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        """Update the policy using PPO with proper LSTM state handling."""
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
                # Use the stored initial LSTM states from the beginning of the rollout
                initial_lstm_state = (
                    self.initial_lstm_state[0][:, mbenvinds].clone(),
                    self.initial_lstm_state[1][:, mbenvinds].clone()
                )
                
                # Reconstruct LSTM states sequentially for this mini-batch
                # This is crucial for proper LSTM training - we need to process the sequence in order
                mb_obs_reshaped = {}
                for key in mb_obs:
                    mb_obs_reshaped[key] = mb_obs[key].reshape(self.config.num_steps, envsperbatch, -1)
                
                # Process each timestep sequentially to reconstruct LSTM states
                current_lstm_state = initial_lstm_state
                reconstructed_hidden = []
                
                for step in range(self.config.num_steps):
                    step_obs = {}
                    for key in mb_obs_reshaped:
                        step_obs[key] = mb_obs_reshaped[key][step]
                    
                    # Get done flags for this step
                    step_dones = self.dones[step, mbenvinds]
                    
                    # Get hidden states for this step
                    with torch.no_grad():
                        step_hidden, current_lstm_state = self.agent.get_states(
                            step_obs, current_lstm_state, step_dones
                        )
                    reconstructed_hidden.append(step_hidden)
                
                # Now compute action and value using the reconstructed hidden states
                # The reconstructed_hidden contains hidden states for each timestep
                # We need to use the hidden states corresponding to the actual actions taken
                reconstructed_hidden = torch.stack(reconstructed_hidden, dim=0)  # [num_steps, envsperbatch, hidden_size]
                
                # Compute action and value using the reconstructed hidden states
                if self.agent.is_discrete:
                    logits = self.agent.actor(reconstructed_hidden.view(-1, reconstructed_hidden.size(-1)))
                    probs = Categorical(logits=logits)
                    action_indices = b_actions.long()[mb_inds].argmax(dim=1)
                    newlogprob = probs.log_prob(action_indices)
                    entropy = probs.entropy()
                else:
                    action_mean = self.agent.actor_mean(reconstructed_hidden.view(-1, reconstructed_hidden.size(-1)))
                    action_logstd = self.agent.actor_logstd.expand_as(action_mean)
                    action_std = torch.exp(action_logstd)
                    probs = Normal(action_mean, action_std)
                    newlogprob = probs.log_prob(b_actions[mb_inds]).sum(1)
                    entropy = probs.entropy().sum(1)
                
                newvalue = self.agent.critic(reconstructed_hidden.view(-1, reconstructed_hidden.size(-1))).flatten()
                
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
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
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
            # Annealing the learning rate if instructed
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect rollout
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = self.collect_rollout()
            
            # Update policy
            self.update_policy(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
            
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