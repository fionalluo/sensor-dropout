# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # Get observation space and keys
        obs_space = envs.obs_space
        
        # Get the full_keys from config
        import re
        mlp_keys_pattern = r'.*'  # Default pattern to match all keys
        cnn_keys_pattern = r'.*'  # Default pattern to match all keys
        if hasattr(envs, 'config') and 'full_keys' in envs.config:
            mlp_keys_pattern = envs.config['full_keys']['mlp_keys']
            cnn_keys_pattern = envs.config['full_keys']['cnn_keys']
        
        # Filter keys based on the regex patterns and observation shapes
        self.mlp_keys = []
        self.cnn_keys = []
        for k in obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                self.cnn_keys.append(k)
            else:  # Non-image observations
                self.mlp_keys.append(k)
        
        # MLP and CNN keys that match the regex patterns in the config
        self.mlp_keys = [k for k in self.mlp_keys if re.match(mlp_keys_pattern, k)]
        self.cnn_keys = [k for k in self.cnn_keys if re.match(cnn_keys_pattern, k)]
        
        # Calculate total input size for MLP
        self.total_mlp_size = 0
        self.mlp_key_sizes = {}  # Store the size of each MLP key
        for key in self.mlp_keys:
            if isinstance(obs_space[key].shape, tuple):
                size = np.prod(obs_space[key].shape)
            else:
                size = 1
            self.mlp_key_sizes[key] = size
            self.total_mlp_size += size
            print(f"  {key}: {obs_space[key].shape} -> size: {size}")
                
        print(f"Using MLP keys: {self.mlp_keys}")
        print(f"Using CNN keys: {self.cnn_keys}")
        print(f"Total MLP input size: {self.total_mlp_size}")
        print(f"Observation space shapes:")
        for key in self.mlp_keys + self.cnn_keys:
            print(f"  {key}: {obs_space[key].shape}")

        # CNN network for image observations
        if self.cnn_keys:
            # Get the shape of the first CNN observation
            cnn_shape = obs_space[self.cnn_keys[0]].shape
            print(f"CNN input shape: {cnn_shape}")
            
            # Calculate the size after convolutions
            h, w = cnn_shape[0], cnn_shape[1]
            h = ((h - 8) // 4 + 1 - 4) // 2 + 1
            w = ((w - 8) // 4 + 1 - 4) // 2 + 1
            h = (h - 3) // 1 + 1
            w = (w - 3) // 1 + 1
            cnn_output_size = 64 * h * w
            print(f"CNN output size: {cnn_output_size}")
            
            self.cnn = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(cnn_output_size, 512)),
                nn.ReLU(),
            )
            cnn_output_size = 512 * len(self.cnn_keys)  # One CNN output per image
        else:
            cnn_output_size = 0

        # MLP network for non-image observations
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(self.total_mlp_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        mlp_output_size = 64

        # Combine features from both networks
        combined_size = cnn_output_size + mlp_output_size
        
        # Actor and critic networks
        self.critic = nn.Sequential(
            layer_init(nn.Linear(combined_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(combined_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.act_space['action'].shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.act_space['action'].shape)))

    def get_value(self, x):
        # Handle both dictionary and tensor inputs
        if isinstance(x, dict):
            # Process CNN keys if they exist
            cnn_features = []
            if self.cnn_keys:
                for key in self.cnn_keys:
                    # Get batch size from the first observation
                    batch_size = x[key].shape[0]
                    # Reshape image to (batch_size, channels, height, width)
                    img = x[key].permute(0, 3, 1, 2) / 255.0
                    features = self.cnn(img)
                    cnn_features.append(features)
                cnn_features = torch.cat(cnn_features, dim=1)
            else:
                cnn_features = None
                # Get batch size from the first MLP observation
                batch_size = x[self.mlp_keys[0]].shape[0] if self.mlp_keys else 1
            
            # Process MLP keys
            mlp_features = torch.zeros(batch_size, self.total_mlp_size).to(x[self.mlp_keys[0]].device)
            current_idx = 0
            for key in self.mlp_keys:
                size = self.mlp_key_sizes[key]
                if isinstance(x[key].shape, tuple):
                    mlp_features[:, current_idx:current_idx+size] = x[key].view(batch_size, -1)
                else:
                    mlp_features[:, current_idx] = x[key]
                current_idx += size
            mlp_features = self.mlp(mlp_features)
        else:
            # x is already a tensor of shape (batch_size, total_mlp_size)
            batch_size = x.shape[0]
            # Ensure x is properly reshaped to (batch_size, total_mlp_size)
            if x.dim() > 2:
                x = x.view(batch_size, -1)
            mlp_features = self.mlp(x)
            cnn_features = None
        
        # Combine features
        if cnn_features is not None:
            combined_features = torch.cat([cnn_features, mlp_features], dim=1)
        else:
            combined_features = mlp_features

        return self.critic(combined_features)

    def get_action_and_value(self, x, action=None):
        # Handle both dictionary and tensor inputs
        if isinstance(x, dict):
            # Process CNN keys if they exist
            cnn_features = []
            if self.cnn_keys:
                for key in self.cnn_keys:
                    # Get batch size from the first observation
                    batch_size = x[key].shape[0]
                    # Reshape image to (batch_size, channels, height, width)
                    img = x[key].permute(0, 3, 1, 2) / 255.0
                    features = self.cnn(img)
                    cnn_features.append(features)
                cnn_features = torch.cat(cnn_features, dim=1)
            else:
                cnn_features = None
                # Get batch size from the first MLP observation
                batch_size = x[self.mlp_keys[0]].shape[0] if self.mlp_keys else 1
            
            # Process MLP keys
            mlp_features = torch.zeros(batch_size, self.total_mlp_size).to(x[self.mlp_keys[0]].device)
            current_idx = 0
            for key in self.mlp_keys:
                size = self.mlp_key_sizes[key]
                if isinstance(x[key].shape, tuple):
                    mlp_features[:, current_idx:current_idx+size] = x[key].view(batch_size, -1)
                else:
                    mlp_features[:, current_idx] = x[key]
                current_idx += size
            mlp_features = self.mlp(mlp_features)
        else:
            # x is already a tensor of shape (batch_size, total_mlp_size)
            batch_size = x.shape[0]
            # Ensure x is properly reshaped to (batch_size, total_mlp_size)
            if x.dim() > 2:
                x = x.view(batch_size, -1)
            mlp_features = self.mlp(x)
            cnn_features = None
        
        # Combine features
        if cnn_features is not None:
            combined_features = torch.cat([cnn_features, mlp_features], dim=1)
        else:
            combined_features = mlp_features

        # Get action and value
        action_mean = self.actor_mean(combined_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(combined_features)


def main(envs, config, seed: int = 0):
    # Calculate batch sizes and iterations
    batch_size = int(config['num_envs'] * config['num_steps'])
    minibatch_size = int(batch_size // config['num_minibatches'])
    num_iterations = config['total_timesteps'] // batch_size
    
    # Setup experiment name and tracking
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{config['task']}__{exp_name}__{seed}__{int(time.time())}"
    
    if config['track']:
        import wandb
        wandb.init(
            project=config['wandb_project_name'],
            entity=config['wandb_entity'],
            sync_tensorboard=True,
            config=config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config['torch_deterministic']

    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((config['num_steps'], config['num_envs'], agent.total_mlp_size)).to(device)
    actions = torch.zeros((config['num_steps'], config['num_envs']) + envs.act_space['action'].shape).to(device)
    logprobs = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    rewards = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    dones = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    values = torch.zeros((config['num_steps'], config['num_envs'])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    # Initialize actions with zeros and reset flags
    action_shape = envs.act_space['action'].shape
    print(f"Action space shape: {action_shape}")
    
    # Get num_envs from config
    num_envs = config['num_envs']
    print(f"Number of environments from config: {num_envs}")
    
    # Initialize actions with the correct shapes from the environment
    acts = {
        'action': np.zeros((num_envs,) + action_shape, dtype=np.float32),  # Shape (num_envs, action_dim)
        'reset': np.ones(num_envs, dtype=bool)  # Shape (num_envs,)
    }
    
    # Verify shapes
    print(f"Initial acts shape: {acts['action'].shape}, {acts['reset'].shape}")
    
    # Get initial observations
    obs_dict = envs.step(acts)
    
    # Convert observations to correct dtype and store in dictionary
    next_obs = {}
    for key in agent.mlp_keys + agent.cnn_keys:
        # Keep the batch dimension (num_envs) as the first dimension
        next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if config['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * config['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config['num_steps']):
            global_step += config['num_envs']
            
            # Store MLP observations for each environment
            mlp_obs = []
            for key in agent.mlp_keys:
                # Reshape to (batch_size, feature_dim) while preserving the batch dimension
                # The input shape is (num_envs, ...), we want to keep num_envs as first dim
                mlp_obs.append(next_obs[key].reshape(config['num_envs'], -1))
            
            # Concatenate along the feature dimension (dim=1)
            # This will give us shape (num_envs, total_mlp_size)
            obs[step] = torch.cat(mlp_obs, dim=1)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # Get actions for each environment
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Prepare actions for environment step
            action_np = action.cpu().numpy()            
            # Create acts dictionary with the right shapes
            acts = {
                'action': action_np,  # Should already be (4,4)
                'reset': next_done.cpu().numpy()  # Shape (4,)
            }

            # TRY NOT TO MODIFY: execute the game and log data.
            obs_dict = envs.step(acts)
            
            # Convert observations to correct dtype and store in dictionary
            for key in agent.mlp_keys + agent.cnn_keys:
                # Keep the batch dimension (num_envs) as the first dimension
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
            rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(device).view(-1)

            if "episode" in obs_dict:
                print(f"global_step={global_step}, episodic_return={obs_dict['episode']['r']}")
                writer.add_scalar("charts/episodic_return", obs_dict["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", obs_dict["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config['num_steps'])):
                if t == config['num_steps'] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape(-1, agent.total_mlp_size)  # Reshape to (batch_size, total_input_size)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, np.prod(envs.act_space['action'].shape))  # Reshape to (batch_size, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config['clip_coef'],
                        config['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config['ent_coef'] * entropy_loss + v_loss * config['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
                optimizer.step()

            if config['target_kl'] is not None and approx_kl > config['target_kl']:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if config['save_model']:
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            config['task'],
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=config['gamma'],
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if config['upload_model']:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{config['task']}-{exp_name}-seed{seed}"
            repo_id = f"{config['hf_entity']}/{repo_name}" if config['hf_entity'] else repo_name
            push_to_hub(config, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
