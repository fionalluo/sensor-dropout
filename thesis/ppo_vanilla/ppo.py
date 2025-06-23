import os
import sys
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import re

# Add thesis directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.nets import ImageEncoderResnet

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config
        
        # Get observation space and keys
        obs_space = envs.obs_space
        
        # Get encoder config
        encoder_config = config.encoder
        
        # Filter keys based on the regex patterns
        self.mlp_keys = []
        self.cnn_keys = []
        for k in obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(config.full_keys.cnn_keys, k):
                    self.cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(config.full_keys.mlp_keys, k):
                    self.mlp_keys.append(k)
        
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

        print(f"Using MLP keys: {self.mlp_keys}")
        print(f"Using CNN keys: {self.cnn_keys}")
        print(f"Total MLP input size: {self.total_mlp_size}")

        # Initialize activation function
        if encoder_config.act == 'silu':
            self.act = nn.SiLU()
        elif encoder_config.act == 'relu':
            self.act = nn.ReLU()
        elif encoder_config.act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {encoder_config.act}")

        # Calculate CNN output dimension
        # For ResNet encoder:
        # Final depth = depth * 2 ** (stages - 1)
        # Output dim = Final depth × minres × minres
        if self.cnn_keys:
            # Calculate number of stages based on minres
            # Assuming input size is 64x64 (from config)
            input_size = 64  # From config.env.atari.size
            stages = int(np.log2(input_size) - np.log2(encoder_config.minres))
            final_depth = encoder_config.cnn_depth * (2 ** (stages - 1))
            cnn_output_dim = final_depth * encoder_config.minres * encoder_config.minres
            
            # CNN encoder for image observations
            self.cnn_encoder = nn.Sequential(
                ImageEncoderResnet(
                    depth=encoder_config.cnn_depth,
                    blocks=encoder_config.cnn_blocks,
                    resize=encoder_config.resize,
                    minres=encoder_config.minres,
                    output_dim=cnn_output_dim
                ),
                nn.LayerNorm(cnn_output_dim) if encoder_config.norm == 'layer' else nn.Identity()
            )
        else:
            self.cnn_encoder = None
            cnn_output_dim = 0

        # MLP encoder for non-image observations
        if self.mlp_keys:
            layers = []
            input_dim = self.total_mlp_size
            
            # Add MLP layers
            for _ in range(encoder_config.mlp_layers):
                layers.extend([
                    layer_init(nn.Linear(input_dim, encoder_config.mlp_units)),
                    self.act,
                    nn.LayerNorm(encoder_config.mlp_units) if encoder_config.norm == 'layer' else nn.Identity()
                ])
                input_dim = encoder_config.mlp_units
            
            self.mlp_encoder = nn.Sequential(*layers)
        else:
            self.mlp_encoder = None
        
        # Calculate total input dimension for latent projector
        total_input_dim = (cnn_output_dim if self.cnn_encoder is not None else 0) + (encoder_config.mlp_units if self.mlp_encoder is not None else 0)
        
        # Project concatenated features to latent space
        self.latent_projector = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, encoder_config.output_dim)),
            self.act,
            nn.LayerNorm(encoder_config.output_dim) if encoder_config.norm == 'layer' else nn.Identity(),
            layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim)),
            self.act,
            nn.LayerNorm(encoder_config.output_dim) if encoder_config.norm == 'layer' else nn.Identity()
        )
        
        # Determine if action space is discrete or continuous
        self.is_discrete = envs.act_space['action'].discrete
        
        # Actor and critic networks operating on latent space
        self.critic = nn.Sequential(
            layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim // 2)),
            self.act,
            layer_init(nn.Linear(encoder_config.output_dim // 2, 1), std=1.0),
        )
        
        if self.is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim // 2)),
                self.act,
                layer_init(nn.Linear(encoder_config.output_dim // 2, envs.act_space['action'].shape[0]), std=0.01),
            )
        else:
            action_size = np.prod(envs.act_space['action'].shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim // 2)),
                self.act,
                layer_init(nn.Linear(encoder_config.output_dim // 2, action_size), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

    def encode_observations(self, x):
        if isinstance(x, dict):
            # Process CNN observations
            cnn_features = None
            if self.cnn_keys and self.cnn_encoder is not None:
                # Stack all CNN observations along channels
                cnn_inputs = []
                for key in self.cnn_keys:
                    if key not in x:  # Skip if key doesn't exist
                        continue
                    batch_size = x[key].shape[0]
                    img = x[key].permute(0, 3, 1, 2) / 255.0  # Convert to [B, C, H, W] and normalize
                    cnn_inputs.append(img)
                
                if cnn_inputs:  # Only process if we have any CNN features
                    # Stack along channel dimension
                    cnn_input = torch.cat(cnn_inputs, dim=1)
                    cnn_features = self.cnn_encoder(cnn_input)
            
            # Process MLP observations
            mlp_features = None
            if self.mlp_keys and self.mlp_encoder is not None:
                mlp_features = []
                batch_size = None
                for key in self.mlp_keys:
                    if key not in x:  # Skip if key doesn't exist
                        continue
                    if batch_size is None:
                        batch_size = x[key].shape[0]
                    size = self.mlp_key_sizes[key]
                    if isinstance(x[key].shape, tuple):
                        mlp_features.append(x[key].view(batch_size, -1))
                    else:
                        mlp_features.append(x[key].view(batch_size, 1))
                
                if mlp_features:  # Only process if we have any MLP features
                    mlp_features = torch.cat(mlp_features, dim=1)
                    mlp_features = self.mlp_encoder(mlp_features)
                        
            # Handle the case where neither exists
            if cnn_features is None and mlp_features is None:
                raise ValueError("No valid observations found in input dictionary")
            
            # Concatenate features if both exist, otherwise use whichever exists
            if cnn_features is not None and mlp_features is not None:
                features = torch.cat([cnn_features, mlp_features], dim=1)
            elif cnn_features is not None:
                features = cnn_features
            else:  # mlp_features is not None
                features = mlp_features
        else:
            # Handle tensor input (assumed to be MLP features)
            if self.mlp_encoder is None:
                raise ValueError("MLP encoder not initialized but received tensor input")
            batch_size = x.shape[0]
            if x.dim() > 2:
                x = x.view(batch_size, -1)
            features = self.mlp_encoder(x)
                
        # Project to final latent space
        latent = self.latent_projector(features)
        return latent

    def get_value(self, x):
        latent = self.encode_observations(x)
        return self.critic(latent)

    def get_action_and_value(self, x, action=None):
        latent = self.encode_observations(x)
        
        if self.is_discrete:
            logits = self.actor(latent)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
                # Convert to one-hot for environment
                one_hot_action = torch.zeros_like(logits)
                one_hot_action.scatter_(1, action.unsqueeze(1), 1.0)
                return one_hot_action, probs.log_prob(action), probs.entropy(), self.critic(latent)
            else:
                # Convert one-hot action back to indices for log_prob calculation
                action_indices = action.argmax(dim=1)
                return action, probs.log_prob(action_indices), probs.entropy(), self.critic(latent)
        else:
            action_mean = self.actor_mean(latent)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(latent)

def process_video_frames(frames, key):
    """Process frames for video logging following exact format requirements."""
    print(f"Processing {key} with shape {frames.shape}")
    print(f"Max value: {np.max(frames)}")
    
    if len(frames.shape) == 3:  # Single image [H, W, C]
        print(f"Single image: {key}, {frames.shape}")
        print(f"Last dim: {frames.shape[-1]}")
        # Check if the last dimension is 3 (RGB image) and the maximum value is greater than 1
        if frames.shape[-1] == 3 and np.max(frames) > 1:
            return frames  # Directly pass the image without modification
        else:
            print(f"Converting image: {key}, {frames.shape}")
            frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
            frames = np.transpose(frames, [2, 0, 1])
            return frames
    elif len(frames.shape) == 4:  # Video [T, H, W, C]
        # Sanity check that the channels dimension is last
        assert frames.shape[3] in [1, 3, 4], f"Invalid shape: {frames.shape}"
        is_depth = frames.shape[3] == 1
        frames = np.transpose(frames, [0, 3, 1, 2])
        # If the video is a float, convert it to uint8
        if np.issubdtype(frames.dtype, np.floating):
            if is_depth:
                frames = frames - frames.min()
                # Scale by 2 mean distances of near rays
                frames = frames / (2 * frames[frames <= 1].mean())
                # Scale to [0, 255]
                frames = np.clip(frames, 0, 1)
                # repeat channel dimension 3 times
                frames = np.repeat(frames, 3, axis=1)
            frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
        return frames
    else:
        raise ValueError(f"Unexpected shape for {key}: {frames.shape}")

def main(envs, config, seed: int = 0):
    batch_size = int(config.num_envs * config.num_steps)
    minibatch_size = int(batch_size // config.num_minibatches)
    num_iterations = config.total_timesteps // batch_size
    
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{config.task}__{exp_name}__{seed}__{int(time.time())}"
    
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=False,  # Disable automatic gym monitoring
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    agent = Agent(envs, config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Initialize observation storage
    obs = {}
    for key in agent.mlp_keys + agent.cnn_keys:
        if key in agent.mlp_keys:
            obs[key] = torch.zeros((config.num_steps, config.num_envs, agent.mlp_key_sizes[key])).to(device)
        else:  # CNN keys
            obs[key] = torch.zeros((config.num_steps, config.num_envs) + envs.obs_space[key].shape).to(device)
    
    # Initialize action storage with correct shape for one-hot if discrete
    if agent.is_discrete:
        action_shape = (config.num_steps, config.num_envs, envs.act_space['action'].shape[0])
    else:
        action_shape = (config.num_steps, config.num_envs) + envs.act_space['action'].shape
    actions = torch.zeros(action_shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    
    # Initialize episode tracking buffers
    episode_returns = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs)
    
    # Initialize video logging buffers
    video_frames = {key: [] for key in config.log_keys_video}
    last_video_log = 0
    video_log_interval = 10000  # Log a video every 10k steps
    
    action_shape = envs.act_space['action'].shape
    num_envs = config.num_envs
    
    acts = {
        'action': np.zeros((num_envs,) + action_shape, dtype=np.float32),
        'reset': np.ones(num_envs, dtype=bool)
    }
    
    obs_dict = envs.step(acts)
    next_obs = {}
    for key in agent.mlp_keys + agent.cnn_keys:
        next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)

    for iteration in range(1, num_iterations + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.num_steps):
            global_step += config.num_envs
            
            # Store observations
            for key in agent.mlp_keys + agent.cnn_keys:
                obs[key][step] = next_obs[key]
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Convert action to numpy for environment
            action_np = action.cpu().numpy()
            if agent.is_discrete:
                # For discrete actions, we already have one-hot encoding
                action_np = action_np.reshape(num_envs, -1)
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = envs.step(acts)
            # Store raw observations for video logging before any processing
            for key in config.log_keys_video:
                if key in obs_dict:
                    video_frames[key].append(obs_dict[key][0].copy())  # Store raw observation
            
            # Process observations for the agent
            for key in agent.mlp_keys + agent.cnn_keys:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
            rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(device).view(-1)

            # Update episode returns and lengths
            for env_idx in range(config.num_envs):
                episode_returns[env_idx] += obs_dict['reward'][env_idx]
                episode_lengths[env_idx] += 1
                
                if obs_dict['is_last'][env_idx]:
                    print(f"global_step={global_step}, episode_return={episode_returns[env_idx]}, episode_length={episode_lengths[env_idx]}")
                    writer.add_scalar("charts/episode_return", episode_returns[env_idx], global_step)
                    writer.add_scalar("charts/episode_length", episode_lengths[env_idx], global_step)
                    
                    if config.track:
                        wandb.log({
                            "charts/episode_return": episode_returns[env_idx],
                            "charts/episode_length": episode_lengths[env_idx],
                            "metrics/global_step": global_step,
                        })

                    # Log video if enough steps have passed
                    if global_step - last_video_log >= video_log_interval:
                        for key in config.log_keys_video:
                            if video_frames[key]:
                                # Convert frames to video
                                frames = np.stack(video_frames[key])
                                # Process frames using the exact logic
                                processed_frames = process_video_frames(frames, key)
                                
                                if config.track:
                                    wandb.log({
                                        f"videos/{key}": wandb.Video(
                                            processed_frames,
                                            fps=10,
                                            format="gif"
                                        )
                                    }, step=global_step)
                                # Remove TensorBoard video logging
                                # writer.add_video(f"videos/{key}", processed_frames[None], global_step, fps=10)
                        
                        # Reset video frames and update last log
                        video_frames = {key: [] for key in config.log_keys_video}
                        last_video_log = global_step

                    # Reset for the next episode
                    episode_returns[env_idx] = 0
                    episode_lengths[env_idx] = 0

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Prepare minibatches
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Create minibatch observations dictionary
                mb_obs = {}
                for key in agent.mlp_keys + agent.cnn_keys:
                    mb_obs[key] = obs[key].reshape(-1, *obs[key].shape[2:])[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, actions.reshape(-1, *actions.shape[2:])[mb_inds])
                logratio = newlogprob - logprobs.reshape(-1)[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = advantages.reshape(-1)[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - returns.reshape(-1)[mb_inds]) ** 2
                    v_clipped = values.reshape(-1)[mb_inds] + torch.clamp(
                        newvalue - values.reshape(-1)[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns.reshape(-1)[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns.reshape(-1)[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            # Handle target_kl comparison with string 'None' support
            target_kl = config.target_kl
            if isinstance(target_kl, str) and target_kl.lower() == 'none':
                target_kl = None

            if target_kl is not None and approx_kl.item() > float(target_kl):
                break

        y_pred, y_true = values.reshape(-1).cpu().numpy(), returns.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: Log to TensorBoard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Additional metrics for WandB
        if config.track:
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "metrics/mean_reward": rewards.mean().item(),
                "metrics/max_reward": rewards.max().item(),
                "metrics/min_reward": rewards.min().item(),
                "metrics/mean_value": values.mean().item(),
                "metrics/max_value": values.max().item(),
                "metrics/min_value": values.min().item(),
                "metrics/mean_advantage": advantages.mean().item(),
                "metrics/max_advantage": advantages.max().item(),
                "metrics/min_advantage": advantages.min().item(),
                "metrics/mean_return": returns.mean().item(),
                "metrics/max_return": returns.max().item(),
                "metrics/min_return": returns.min().item(),
                "metrics/mean_entropy": entropy.mean().item(),
                "metrics/max_entropy": entropy.max().item(),
                "metrics/min_entropy": entropy.min().item(),
                "metrics/mean_ratio": ratio.mean().item(),
                "metrics/max_ratio": ratio.max().item(),
                "metrics/min_ratio": ratio.min().item(),
                "metrics/mean_logprob": logprobs.mean().item(),
                "metrics/max_logprob": logprobs.max().item(),
                "metrics/min_logprob": logprobs.min().item(),
                "metrics/mean_grad_norm": torch.norm(torch.cat([p.grad.view(-1) for p in agent.parameters() if p.grad is not None])).item() if any(p.grad is not None for p in agent.parameters()) else 0.0,
                "metrics/mean_param_norm": torch.norm(torch.cat([p.view(-1) for p in agent.parameters()])).item(),
                "metrics/num_updates": iteration,
                "metrics/global_step": global_step,
                "metrics/epoch": epoch,
            })

        # Print recent rewards (last 10 steps)
        recent_steps = min(10, step + 1)  # Handle case where we haven't collected 10 steps yet
        recent_rewards = rewards[step-recent_steps+1:step+1].mean().item()
        # print(f"Global step: {global_step}, Recent rewards: {recent_rewards:.2f}")

    if config.save_model:
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
