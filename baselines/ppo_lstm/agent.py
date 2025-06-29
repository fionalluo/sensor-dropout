import torch
import torch.nn as nn
import numpy as np
import re
from ..shared.agent import BaseAgent, layer_init
from ..shared.nets import ImageEncoderResnet, LightweightImageEncoder
from torch.distributions import Categorical, Normal

class PPOLSTMAgent(BaseAgent):
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
        
        # MLP encoder for non-image observations - SIMPLIFIED
        if self.mlp_keys:
            self.mlp_encoder = nn.Sequential(
                layer_init(nn.Linear(self.total_mlp_size, config.encoder.mlp_units)),
                self.act,
                layer_init(nn.Linear(config.encoder.mlp_units, config.encoder.mlp_units)),
                self.act,
            )
        else:
            self.mlp_encoder = None
        
        # Calculate total input dimension for encoder output
        total_input_dim = self.cnn_output_dim + (config.encoder.mlp_units if self.mlp_encoder is not None else 0)
        
        # SIMPLIFIED: Direct projection to encoder output dimension
        self.encoder_output = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, config.encoder.output_dim)),
            self.act,
        )

        # LSTM layer
        self.lstm = nn.LSTM(config.encoder.output_dim, config.lstm.hidden_size)
        
        # Initialize LSTM weights with orthogonal initialization and zero bias
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # SIMPLIFIED: Actor and critic networks (Version 1 style - single hidden layer)
        lstm_output_dim = config.lstm.hidden_size
        hidden_dim = lstm_output_dim // 2  # Simple halving like Version 1
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_output_dim, hidden_dim)),
            self.act,
            layer_init(nn.Linear(hidden_dim, envs.act_space['action'].shape[0]), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_output_dim, hidden_dim)),
            self.act,
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        if not self.is_discrete:
            action_size = np.prod(envs.act_space['action'].shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(lstm_output_dim, hidden_dim)),
                self.act,
                layer_init(nn.Linear(hidden_dim, action_size), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

        # SIMPLIFIED: Direct projection for non-LSTM mode
        self.encoder_to_lstm_projection = nn.Sequential(
            layer_init(nn.Linear(config.encoder.output_dim, lstm_output_dim)),
            self.act,
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
            
            # Process heavyweight CNN observations
            if self.heavyweight_cnn_encoder is not None:
                for key in self.heavyweight_cnn_keys:
                    if key in x:
                        cnn_features.append(self.heavyweight_cnn_encoder(x[key]))
            
            # Process lightweight CNN observations
            if self.lightweight_cnn_encoder is not None:
                lightweight_inputs = []
                for key in self.lightweight_cnn_keys:
                    if key in x:
                        lightweight_inputs.append(x[key])
                
                if lightweight_inputs:
                    # Concatenate lightweight images along channel dimension
                    lightweight_concat = torch.cat(lightweight_inputs, dim=-1)
                    cnn_features.append(self.lightweight_cnn_encoder(lightweight_concat))
            
            # Process MLP observations
            mlp_features = []
            if self.mlp_encoder is not None:
                for key in self.mlp_keys:
                    if key in x:
                        # Flatten the observation
                        if len(x[key].shape) > 2:
                            batch_size = x[key].shape[0]
                            flattened = x[key].view(batch_size, -1)
                        else:
                            flattened = x[key]
                        mlp_features.append(flattened)
                
                if mlp_features:
                    # Concatenate all MLP features
                    mlp_concat = torch.cat(mlp_features, dim=-1)
                    mlp_features = [self.mlp_encoder(mlp_concat)]
            
            # Concatenate all features
            all_features = cnn_features + mlp_features
            if all_features:
                features = torch.cat(all_features, dim=-1)
            else:
                # Handle case where no features are available
                batch_size = next(iter(x.values())).shape[0] if x else 1
                features = torch.zeros(batch_size, 0, device=self.device)
        
        else:
            # Handle tensor input (assumed to be MLP features)
            if self.mlp_encoder is None:
                raise ValueError("MLP encoder not initialized but received tensor input")
            batch_size = x.shape[0]
            if x.dim() > 2:
                x = x.view(batch_size, -1)
            features = self.mlp_encoder(x)
        
        # Project to encoder output dimension
        latent = self.encoder_output(features)
        return latent 

    def get_states(self, x, lstm_state, done):
        """Get LSTM states from observations and current LSTM state."""
        hidden = self.encode_observations(x)
        
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state=None, done=None):
        """Get value from observations and LSTM state."""
        if lstm_state is None or done is None:
            # Fallback to non-LSTM version but use our own critic network
            latent = self.encode_observations(x)
            # Project encoder output to LSTM output dimension
            projected_latent = self.encoder_to_lstm_projection(latent)
            return self.critic(projected_latent)
        
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
        """Get action and value from policy for PPO LSTM."""
        if lstm_state is None or done is None:
            # Fallback to non-LSTM version but use our own actor/critic networks
            latent = self.encode_observations(x)
            # Project encoder output to LSTM output dimension
            projected_latent = self.encoder_to_lstm_projection(latent)
            
            if self.is_discrete:
                logits = self.actor(projected_latent)
                probs = Categorical(logits=logits)
                if action is None:
                    action = probs.sample()
                    # Convert to one-hot for environment
                    one_hot_action = torch.zeros_like(logits)
                    one_hot_action.scatter_(1, action.unsqueeze(1), 1.0)
                    return one_hot_action, probs.log_prob(action), probs.entropy(), self.critic(projected_latent)
                else:
                    # Convert one-hot action back to indices for log_prob calculation
                    action_indices = action.argmax(dim=1)
                    return action, probs.log_prob(action_indices), probs.entropy(), self.critic(projected_latent)
            else:
                action_mean = self.actor_mean(projected_latent)
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = Normal(action_mean, action_std)
                if action is None:
                    action = probs.sample()
                return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(projected_latent)
        
        # LSTM version
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        
        if self.is_discrete:
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
                # Convert to one-hot for environment
                one_hot_action = torch.zeros_like(logits)
                one_hot_action.scatter_(1, action.unsqueeze(1), 1.0)
                return one_hot_action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
            else:
                # Convert one-hot action back to indices for log_prob calculation
                action_indices = action.argmax(dim=1)
                return action, probs.log_prob(action_indices), probs.entropy(), self.critic(hidden), lstm_state
        else:
            action_mean = self.actor_mean(hidden)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state

    def get_initial_lstm_state(self, batch_size=None):
        """Get initial LSTM state (zeros) for a given batch size.
        
        Args:
            batch_size: Number of environments/batch size. If None, returns a default size of 1.
            
        Returns:
            tuple: (hidden_state, cell_state) initialized to zeros
        """
        if batch_size is None:
            batch_size = 1  # Default for evaluation
        
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device)
        ) 