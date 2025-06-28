import torch
import torch.nn as nn
import numpy as np
import re
from ..shared.agent import BaseAgent, layer_init
from ..shared.nets import ImageEncoderResnet, LightweightImageEncoder
from torch.distributions import Categorical, Normal

class PPORnnAgent(BaseAgent):
    def __init__(self, envs, config, training_env=None):
        super().__init__(envs, config)
        
        # Determine which keys to use for training
        if hasattr(config, 'eval_keys'):
            if training_env is not None:
                print(f"DEBUG: hasattr(config.eval_keys, '{training_env}') = {hasattr(config.eval_keys, training_env)}")
                # Try dictionary access as well
                if hasattr(config.eval_keys, '__getitem__'):
                    print(f"DEBUG: training_env in config.eval_keys = {training_env in config.eval_keys}")
        
        # Priority order for key selection:
        # 1. If training_env is specified, use eval_keys[training_env]
        # 2. If config.keys has been modified (subset_policies mode), use config.keys
        # 3. Only use full_keys as a last resort
        
        training_keys = None
        
        # Priority 1: Check if training_env is specified
        if training_env is not None and hasattr(config, 'eval_keys'):
            # Try attribute access first
            if hasattr(config.eval_keys, training_env):
                env_keys = getattr(config.eval_keys, training_env)
                training_keys = type('Keys', (), {
                    'mlp_keys': env_keys.mlp_keys,
                    'cnn_keys': env_keys.cnn_keys
                })()
                print(f"Training on {training_env} with keys: {training_keys}")
            # Try dictionary access
            elif hasattr(config.eval_keys, '__getitem__') and training_env in config.eval_keys:
                env_keys = config.eval_keys[training_env]
                training_keys = type('Keys', (), {
                    'mlp_keys': env_keys.mlp_keys,
                    'cnn_keys': env_keys.cnn_keys
                })()
                print(f"Training on {training_env} with keys: {training_keys}")
        
        # Priority 2: Check if config.keys has been modified (subset_policies mode)
        if training_keys is None and hasattr(config, 'keys'):
            # Check if keys has the expected attributes
            if hasattr(config.keys, 'mlp_keys') and hasattr(config.keys, 'cnn_keys'):
                training_keys = config.keys
                print(f"Using config.keys (subset_policies mode)")
        
        # Priority 3: Fall back to full_keys (default behavior)
        if training_keys is None:
            training_keys = config.full_keys
            print(f"Training on full_keys: {training_keys}")
        
        print(f"Training with MLP pattern: {getattr(training_keys, 'mlp_keys', 'N/A')}")
        print(f"Training with CNN pattern: {getattr(training_keys, 'cnn_keys', 'N/A')}")
        
        # Use specified keys for training
        self.mlp_keys = []
        self.cnn_keys = []
        self.lightweight_cnn_keys = []  # Keys for small images
        self.heavyweight_cnn_keys = []  # Keys for large images
        
        for k in envs.obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(envs.obs_space[k].shape) == 3 and envs.obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(training_keys.cnn_keys, k):
                    self.cnn_keys.append(k)
                    # Check if image is very small (≤ 7x7 in first two dimensions)
                    if envs.obs_space[k].shape[0] <= 7 and envs.obs_space[k].shape[1] <= 7:
                        self.lightweight_cnn_keys.append(k)
                    else:
                        self.heavyweight_cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(training_keys.mlp_keys, k):
                    self.mlp_keys.append(k)
        
        print(f"Using MLP keys: {self.mlp_keys}")
        print(f"Using CNN keys: {self.cnn_keys}")
        
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
            # Use a default input size of 64 for large images (can be overridden in config)
            input_size = getattr(config.encoder, 'input_size', 64)
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
        
        # LSTM layer with better initialization
        self.lstm = nn.LSTM(config.encoder.output_dim, config.rnn.hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        # Improved actor and critic networks
        hidden_size = config.rnn.hidden_size
        actor_hidden_size = hidden_size // 2
        critic_hidden_size = hidden_size // 2
        
        # Actor network
        if self.is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(hidden_size, actor_hidden_size)),
                self.act,
                nn.LayerNorm(actor_hidden_size) if config.encoder.norm == 'layer' else nn.Identity(),
                layer_init(nn.Linear(actor_hidden_size, envs.act_space['action'].shape[0]), std=0.01),
            )
        else:
            action_size = np.prod(envs.act_space['action'].shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(hidden_size, actor_hidden_size)),
                self.act,
                nn.LayerNorm(actor_hidden_size) if config.encoder.norm == 'layer' else nn.Identity(),
                layer_init(nn.Linear(actor_hidden_size, action_size), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, critic_hidden_size)),
            self.act,
            nn.LayerNorm(critic_hidden_size) if config.encoder.norm == 'layer' else nn.Identity(),
            layer_init(nn.Linear(critic_hidden_size, 1), std=1.0),
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

    def get_states(self, x, lstm_state, done):
        """Get LSTM states from observations.
        
        Args:
            x: Observations (dict or tensor)
            lstm_state: Current LSTM state (h, c)
            done: Done flags
            
        Returns:
            tuple: (hidden_states, new_lstm_state)
        """
        hidden = self.encode_observations(x)
        
        # LSTM logic with proper batch handling
        batch_size = lstm_state[0].shape[1]
        
        # Handle single timestep vs sequence
        if hidden.dim() == 2:
            # Single timestep - reshape for LSTM
            hidden = hidden.unsqueeze(0)  # Add sequence dimension
            done = done.unsqueeze(0) if done.dim() == 1 else done
            single_step = True
        else:
            # Multiple timesteps - reshape for LSTM
            hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
            done = done.reshape((-1, batch_size))
            single_step = False
        
        new_hidden = []
        
        # Process each timestep
        for h, d in zip(hidden, done):
            # Apply done mask to LSTM states
            # This ensures LSTM states are reset when episodes end
            masked_h = (1.0 - d).view(1, -1, 1) * lstm_state[0]
            masked_c = (1.0 - d).view(1, -1, 1) * lstm_state[1]
            
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (masked_h, masked_c),
            )
            new_hidden += [h]
        
        if single_step:
            new_hidden = torch.cat(new_hidden, dim=0).squeeze(0)
        else:
            new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state=None, done=None):
        """Get value from observations and LSTM state."""
        # Handle case where LSTM states are not provided (e.g., during evaluation)
        if lstm_state is None:
            # Create default LSTM states for evaluation
            batch_size = x[list(x.keys())[0]].shape[0] if isinstance(x, dict) else x.shape[0]
            lstm_state = (
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device)
            )
        
        if done is None:
            # Create default done flags for evaluation
            batch_size = x[list(x.keys())[0]].shape[0] if isinstance(x, dict) else x.shape[0]
            done = torch.zeros(batch_size, device=self.device)
        
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
        """Get action and value from policy for PPO RNN.
        
        Args:
            x: Observations
            lstm_state: LSTM state (h, c)
            done: Done flags
            action: Optional action for evaluation
            
        Returns:
            tuple: (action, log_prob, entropy, value, new_lstm_state)
        """
        # Handle case where LSTM states are not provided (e.g., during evaluation)
        if lstm_state is None:
            # Create default LSTM states for evaluation
            batch_size = x[list(x.keys())[0]].shape[0] if isinstance(x, dict) else x.shape[0]
            lstm_state = (
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device)
            )
        
        if done is None:
            # Create default done flags for evaluation
            batch_size = x[list(x.keys())[0]].shape[0] if isinstance(x, dict) else x.shape[0]
            done = torch.zeros(batch_size, device=self.device)
        
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