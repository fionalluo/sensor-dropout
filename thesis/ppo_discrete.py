# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import embodied


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
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
        
        # Filter keys based on the regex patterns
        self.mlp_keys = [k for k in obs_space.keys() 
                        if k not in ['reward', 'is_first', 'is_last', 'is_terminal']
                        and re.match(mlp_keys_pattern, k)]
        self.cnn_keys = [k for k in obs_space.keys() 
                        if k not in ['reward', 'is_first', 'is_last', 'is_terminal']
                        and re.match(cnn_keys_pattern, k)]
        
        # Calculate total input size for MLP
        self.total_mlp_size = 0
        for key in self.mlp_keys:
            if isinstance(obs_space[key].shape, tuple):
                self.total_mlp_size += np.prod(obs_space[key].shape)
            else:
                self.total_mlp_size += 1
                
        print(f"Using MLP keys: {self.mlp_keys}")
        print(f"Using CNN keys: {self.cnn_keys}")
        print(f"Total MLP input size: {self.total_mlp_size}")

        # CNN network for image observations
        if self.cnn_keys:
            self.cnn = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            cnn_output_size = 512
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
        self.actor = nn.Sequential(
            layer_init(nn.Linear(combined_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.act_space['action'].shape[0]), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # Process CNN keys if they exist
        cnn_features = torch.zeros(x.shape[0], 512).to(x.device) if self.cnn_keys else None
        if self.cnn_keys:
            # Assuming CNN keys are images, process them through CNN
            for key in self.cnn_keys:
                cnn_features = self.cnn(x[key] / 255.0)
        
        # Process MLP keys
        mlp_features = torch.zeros(x.shape[0], self.total_mlp_size).to(x.device)
        for i, key in enumerate(self.mlp_keys):
            if isinstance(x[key].shape, tuple):
                mlp_features[:, i:i+np.prod(x[key].shape)] = x[key].view(x.shape[0], -1)
            else:
                mlp_features[:, i] = x[key]
        mlp_features = self.mlp(mlp_features)
        
        # Combine features
        if cnn_features is not None:
            combined_features = torch.cat([cnn_features, mlp_features], dim=1)
        else:
            combined_features = mlp_features

        # Get action and value
        logits = self.actor(combined_features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(combined_features)


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
    actions = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    logprobs = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    rewards = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    dones = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    values = torch.zeros((config['num_steps'], config['num_envs'])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    # Initialize actions with zeros and reset flags
    acts = {
        k: np.zeros((config['num_envs'],) + v.shape, v.dtype)
        for k, v in envs.act_space.items()}
    acts['reset'] = np.ones(config['num_envs'], bool)  # Reset all environments initially
    
    # Get initial observations
    obs_dict = envs.step(acts)
    # Convert observations to correct dtype and concatenate
    next_obs = torch.cat([
        torch.Tensor(obs_dict[k].astype(np.float32)).view(config['num_envs'], -1) 
        for k in agent.mlp_keys
    ], dim=1).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if config['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * config['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config['num_steps']):
            global_step += config['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Prepare actions for environment step
            action_np = action.cpu().numpy()
            # Convert to one-hot format
            one_hot_action = np.zeros((config['num_envs'], envs.act_space['action'].shape[0]), dtype=np.float32)
            one_hot_action[np.arange(config['num_envs']), action_np] = 1.0
            
            acts = {
                'action': one_hot_action,
                'reset': next_done.cpu().numpy()  # Reset environments that are done
            }
            
            # TRY NOT TO MODIFY: execute the game and log data.
            obs_dict = envs.step(acts)
            # Convert observations to correct dtype and concatenate
            next_obs = torch.cat([
                torch.Tensor(obs_dict[k].astype(np.float32)).view(config['num_envs'], -1) 
                for k in agent.mlp_keys
            ], dim=1).to(device)
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
        b_obs = obs.reshape(-1, next_obs.shape[1])  # Reshape to (batch_size, total_input_size)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

    envs.close()
    writer.close()
