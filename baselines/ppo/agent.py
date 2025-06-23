import torch
import torch.nn as nn
import numpy as np
import re
from ..shared.agent import BaseAgent, layer_init
from ..shared.nets import ImageEncoderResnet

class PPOAgent(BaseAgent):
    def __init__(self, envs, config):
        super().__init__(envs, config)
        
        # Use full keys (all observations) for baselines
        self.mlp_keys = []
        self.cnn_keys = []
        for k in envs.obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(envs.obs_space[k].shape) == 3 and envs.obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(config.full_keys.cnn_keys, k):
                    self.cnn_keys.append(k)
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
        
        # Calculate CNN output dimension
        if self.cnn_keys:
            # Calculate number of stages based on minres
            input_size = 64  # From config.env.atari.size
            stages = int(np.log2(input_size) - np.log2(config.encoder.minres))
            final_depth = config.encoder.cnn_depth * (2 ** (stages - 1))
            self.cnn_output_dim = final_depth * config.encoder.minres * config.encoder.minres
            
            # CNN encoder for image observations
            self.cnn_encoder = nn.Sequential(
                ImageEncoderResnet(
                    depth=config.encoder.cnn_depth,
                    blocks=config.encoder.cnn_blocks,
                    resize=config.encoder.resize,
                    minres=config.encoder.minres,
                    output_dim=self.cnn_output_dim
                ),
                nn.LayerNorm(self.cnn_output_dim) if config.encoder.norm == 'layer' else nn.Identity()
            )
        else:
            self.cnn_encoder = None
            self.cnn_output_dim = 0
        
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
        total_input_dim = (self.cnn_output_dim if self.cnn_encoder is not None else 0) + (config.encoder.mlp_units if self.mlp_encoder is not None else 0)
        
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