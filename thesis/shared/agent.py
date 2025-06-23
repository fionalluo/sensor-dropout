import torch
import torch.nn as nn
import numpy as np
import re
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from .nets import ImageEncoderResnet

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BaseAgent(nn.Module):
    def __init__(self, envs, config, dual_encoder=None):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        self.dual_encoder = dual_encoder
        
        # Get observation space and keys
        obs_space = envs.obs_space
        
        # Filter keys based on the regex patterns
        self.student_mlp_keys = []
        self.student_cnn_keys = []
        self.teacher_mlp_keys = []
        self.teacher_cnn_keys = []
        for k in obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(config.full_keys.cnn_keys, k):
                    self.teacher_cnn_keys.append(k)
                if re.match(config.keys.cnn_keys, k):
                    self.student_cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(config.full_keys.mlp_keys, k):
                    self.teacher_mlp_keys.append(k)
                if re.match(config.keys.mlp_keys, k):
                    self.student_mlp_keys.append(k)
        
        # Initialize activation function
        if config.actor_critic.act == 'silu':
            self.act = nn.SiLU()
        elif config.actor_critic.act == 'relu':
            self.act = nn.ReLU()
        elif config.actor_critic.act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {encoder_config.act}")
        
        # Determine if action space is discrete or continuous
        self.is_discrete = envs.act_space['action'].discrete
        
        # Actor and critic networks operating on latent space
        self.critic = nn.Sequential(
            layer_init(nn.Linear(config.encoder.output_dim, config.encoder.output_dim // 2)),
            self.act,
            layer_init(nn.Linear(config.encoder.output_dim // 2, 1), std=1.0),
        )
        
        if self.is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(config.encoder.output_dim, config.encoder.output_dim // 2)),
                self.act,
                layer_init(nn.Linear(config.encoder.output_dim // 2, envs.act_space['action'].shape[0]), std=0.01),
            )
        else:
            action_size = np.prod(envs.act_space['action'].shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(config.encoder.output_dim, config.encoder.output_dim // 2)),
                self.act,
                layer_init(nn.Linear(config.encoder.output_dim // 2, action_size), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

    def encode_observations(self, x):
        """This method should be overridden by subclasses to use the appropriate encoder."""
        raise NotImplementedError("Subclasses must implement encode_observations")

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