import torch
import torch.nn as nn
import numpy as np
import re
from ..shared.agent import BaseAgent, layer_init
from ..shared.nets import ImageEncoderResnet, LightweightImageEncoder
from torch.distributions import Categorical, Normal

class PPOAgent(BaseAgent):
    def __init__(self, envs, config):
        super().__init__(envs, config)
        
        # Use full keys (all observations) for baselines
        self.mlp_keys = []
        self.cnn_keys = []
        self.lightweight_cnn_keys = []  # Keys for small images
        self.heavyweight_cnn_keys = []  # Keys for large images
        
        for k in envs.obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(envs.obs_space[k].shape) == 3 and envs.obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(config.full_keys.cnn_keys, k):
                    self.cnn_keys.append(k)
                    # Check if image is very small (≤ 7x7 in first two dimensions)
                    if envs.obs_space[k].shape[0] <= 7 and envs.obs_space[k].shape[1] <= 7:
                        self.lightweight_cnn_keys.append(k)
                    else:
                        self.heavyweight_cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(config.full_keys.mlp_keys, k):
                    self.mlp_keys.append(k)
        
        # Calculate total input size for MLP
        self.total_mlp_size = 0
        self.mlp_key_sizes = {}  # Store the size of each MLP key
        for key in self.mlp_keys:
            if isinstance(envs.obs_space[key].shape, tuple):
                size = np.prod(envs.obs_space[key].shape)
            else:
                size = 1
            self.mlp_key_sizes[key] = size
            self.total_mlp_size += size
        
        # Initialize activation function
        if config.encoder.act == 'silu':
            self.act = nn.SiLU()
        elif config.encoder.act == 'relu':
            self.act = nn.ReLU()
        elif config.encoder.act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {config.encoder.act}")
        
        # Calculate CNN output dimensions
        self.cnn_output_dim = 0
        
        # Heavyweight CNN encoder for large images
        if self.heavyweight_cnn_keys:
            # Calculate number of stages based on minres
            input_size = 64  # From config.env.atari.size
            stages = int(np.log2(input_size) - np.log2(config.encoder.minres))
            final_depth = config.encoder.cnn_depth * (2 ** (stages - 1))
            heavyweight_output_dim = final_depth * config.encoder.minres * config.encoder.minres
            
            # Heavyweight CNN encoder for large image observations
            self.heavyweight_cnn_encoder = nn.Sequential(
                ImageEncoderResnet(
                    depth=config.encoder.cnn_depth,
                    blocks=config.encoder.cnn_blocks,
                    resize=config.encoder.resize,
                    minres=config.encoder.minres,
                    output_dim=heavyweight_output_dim
                ),
                nn.LayerNorm(heavyweight_output_dim) if config.encoder.norm == 'layer' else nn.Identity()
            )
            self.cnn_output_dim += heavyweight_output_dim
        else:
            self.heavyweight_cnn_encoder = None
        
        # Lightweight CNN encoder for small images
        if self.lightweight_cnn_keys:
            # Calculate total channels for all lightweight images
            total_lightweight_channels = 0
            for key in self.lightweight_cnn_keys:
                total_lightweight_channels += envs.obs_space[key].shape[-1]  # channels dimension
            
            lightweight_output_dim = 64  # Reduced output dimension for lightweight encoder
            
            # Lightweight CNN encoder for small image observations
            self.lightweight_cnn_encoder = nn.Sequential(
                LightweightImageEncoder(
                    in_channels=total_lightweight_channels,
                    output_dim=lightweight_output_dim
                ),
                nn.LayerNorm(lightweight_output_dim) if config.encoder.norm == 'layer' else nn.Identity()
            )
            self.cnn_output_dim += lightweight_output_dim
        else:
            self.lightweight_cnn_encoder = None
        
        # MLP encoder for non-image observations
        if self.mlp_keys:
            layers = []
            input_dim = self.total_mlp_size
            
            # Add MLP layers
            for _ in range(config.encoder.mlp_layers):
                layers.extend([
                    layer_init(nn.Linear(input_dim, config.encoder.mlp_units)),
                    self.act,
                    nn.LayerNorm(config.encoder.mlp_units) if config.encoder.norm == 'layer' else nn.Identity()
                ])
                input_dim = config.encoder.mlp_units
            
            self.mlp_encoder = nn.Sequential(*layers)
        else:
            self.mlp_encoder = None
        
        # Calculate total input dimension for latent projector
        total_input_dim = self.cnn_output_dim + (config.encoder.mlp_units if self.mlp_encoder is not None else 0)
        
        # Project concatenated features to latent space
        self.latent_projector = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, config.encoder.output_dim)),
            self.act,
            nn.LayerNorm(config.encoder.output_dim) if config.encoder.norm == 'layer' else nn.Identity(),
            layer_init(nn.Linear(config.encoder.output_dim, config.encoder.output_dim)),
            self.act,
            nn.LayerNorm(config.encoder.output_dim) if config.encoder.norm == 'layer' else nn.Identity()
        )

    def encode_observations(self, x):
        """Encode observations into a latent representation.
        
        Args:
            x: Dictionary of observations or tensor input
            
        Returns:
            Latent representation
        """
        if isinstance(x, dict):
            # Filter observations based on whether this is student or teacher encoder
            filtered_x = {}
            # Copy only the relevant keys
            for key in self.mlp_keys:
                if key in x:
                    filtered_x[key] = x[key]
            for key in self.cnn_keys:
                if key in x:
                    filtered_x[key] = x[key]
            
            # Update x to use filtered observations
            x = filtered_x
            
            # Process CNN observations
            cnn_features = []
            
            # Process heavyweight CNN observations (large images)
            if self.heavyweight_cnn_keys and self.heavyweight_cnn_encoder is not None:
                heavyweight_inputs = []
                for key in self.heavyweight_cnn_keys:
                    if key not in x:  # Skip if key doesn't exist
                        continue
                    batch_size = x[key].shape[0]
                    img = x[key].permute(0, 3, 1, 2) / 255.0  # Convert to [B, C, H, W] and normalize
                    heavyweight_inputs.append(img)
                
                if heavyweight_inputs:  # Only process if we have any heavyweight CNN features
                    # Stack along channel dimension
                    heavyweight_input = torch.cat(heavyweight_inputs, dim=1)
                    heavyweight_features = self.heavyweight_cnn_encoder(heavyweight_input)
                    cnn_features.append(heavyweight_features)
            
            # Process lightweight CNN observations (small images)
            if self.lightweight_cnn_keys and self.lightweight_cnn_encoder is not None:
                lightweight_inputs = []
                for key in self.lightweight_cnn_keys:
                    if key not in x:  # Skip if key doesn't exist
                        continue
                    batch_size = x[key].shape[0]
                    img = x[key].permute(0, 3, 1, 2) / 255.0  # Convert to [B, C, H, W] and normalize
                    lightweight_inputs.append(img)
                
                if lightweight_inputs:  # Only process if we have any lightweight CNN features
                    # Stack along channel dimension
                    lightweight_input = torch.cat(lightweight_inputs, dim=1)
                    lightweight_features = self.lightweight_cnn_encoder(lightweight_input)
                    cnn_features.append(lightweight_features)
            
            # Combine all CNN features
            if cnn_features:
                cnn_features = torch.cat(cnn_features, dim=1)
            else:
                cnn_features = None
            
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
                    # Ensure the observation is flattened
                    if x[key].dim() > 2:
                        mlp_features.append(x[key].view(batch_size, -1))
                    else:
                        mlp_features.append(x[key])
                
                if mlp_features:  # Only process if we have any MLP features
                    mlp_features = torch.cat(mlp_features, dim=1)
                    # Ensure the input size matches what the MLP encoder expects
                    if mlp_features.shape[1] != self.total_mlp_size:
                        raise ValueError(f"MLP input size mismatch. Expected {self.total_mlp_size}, got {mlp_features.shape[1]}")
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
        
        # Project to latent space
        latent = self.latent_projector(features)
        return latent

    def get_action_and_value(self, x, action=None):
        """Get action and value from policy for PPO (no imitation losses)."""
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