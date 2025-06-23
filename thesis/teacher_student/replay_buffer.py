import numpy as np
import torch
from collections import deque
import re

class ReplayBuffer:
    def __init__(self, capacity, obs_space, full_keys, partial_keys):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.full_keys = full_keys
        self.partial_keys = partial_keys
        
        # Initialize storage
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
        
        # Get observation space and keys
        self.obs_space = obs_space
        
        # Filter keys based on the regex patterns
        self.full_mlp_keys = []
        self.full_cnn_keys = []
        self.partial_mlp_keys = []
        self.partial_cnn_keys = []
        
        for k in obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(full_keys.cnn_keys, k):
                    self.full_cnn_keys.append(k)
                if re.match(partial_keys.cnn_keys, k):
                    self.partial_cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(full_keys.mlp_keys, k):
                    self.full_mlp_keys.append(k)
                if re.match(partial_keys.mlp_keys, k):
                    self.partial_mlp_keys.append(k)
        
        # Calculate total input size for MLP
        self.full_mlp_size = 0
        self.partial_mlp_size = 0
        self.mlp_key_sizes = {}  # Store the size of each MLP key
        
        for key in set(self.full_mlp_keys + self.partial_mlp_keys):
            if isinstance(obs_space[key].shape, tuple):
                size = np.prod(obs_space[key].shape)
            else:
                size = 1
            self.mlp_key_sizes[key] = size
            if key in self.full_mlp_keys:
                self.full_mlp_size += size
            if key in self.partial_mlp_keys:
                self.partial_mlp_size += size
        
        # Calculate CNN output dimension
        if self.full_cnn_keys or self.partial_cnn_keys:
            # Calculate number of stages based on minres
            input_size = 64  # From config.env.atari.size
            stages = int(np.log2(input_size) - np.log2(encoder_config.minres))
            final_depth = encoder_config.cnn_depth * (2 ** (stages - 1))
            self.cnn_output_dim = final_depth * encoder_config.minres * encoder_config.minres
        else:
            self.cnn_output_dim = 0
        
    def add(self, transition):
        """Add a transition to the buffer.
        
        Args:
            transition: dict containing:
                - obs: dict of observations
                - action: action taken
                - reward: reward received
                - next_obs: dict of next observations
                - done: whether episode ended
        """
        self.buffer.append(transition)
    
    def _filter_observations(self, obs_dict, is_full=True):
        """Filter observations based on whether we want full or partial observations.
        
        Args:
            obs_dict: Dictionary of observations
            is_full: If True, filter using full_keys pattern, else use partial_keys pattern
            
        Returns:
            Filtered dictionary of observations
        """
        filtered_obs = {}
        for key, value in obs_dict.items():
            if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
                
            is_image = len(self.obs_space[key].shape) == 3 and self.obs_space[key].shape[-1] == 3
            
            if is_full:
                if is_image:
                    should_include = key in self.full_cnn_keys
                else:
                    should_include = key in self.full_mlp_keys
            else:
                if is_image:
                    should_include = key in self.partial_cnn_keys
                else:
                    should_include = key in self.partial_mlp_keys
                    
            if should_include:
                filtered_obs[key] = value
                
        return filtered_obs
    
    def sample(self, batch_size):
        """Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            dict containing:
                - full_obs: dict of full observations
                - partial_obs: dict of partial observations
                - action: action taken
                - reward: reward received
                - next_full_obs: dict of next full observations
                - next_partial_obs: dict of next partial observations
                - done: whether episode ended
        """
        if len(self) < batch_size:
            raise ValueError(f"Not enough transitions in buffer. Have {len(self)}, need {batch_size}")
        
        # Sample indices
        indices = np.random.choice(len(self), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        
        # Create batch
        batch = {
            'full_obs': {},
            'partial_obs': {},
            'next_full_obs': {},
            'next_partial_obs': {},
        }
        
        # Process each transition
        for i, transition in enumerate(transitions):
            # Filter observations for full and partial
            full_obs = self._filter_observations(transition['obs'], is_full=True)
            partial_obs = self._filter_observations(transition['obs'], is_full=False)
            next_full_obs = self._filter_observations(transition['next_obs'], is_full=True)
            next_partial_obs = self._filter_observations(transition['next_obs'], is_full=False)
            
            # Add to batch
            for key, value in full_obs.items():
                if key not in batch['full_obs']:
                    batch['full_obs'][key] = []
                batch['full_obs'][key].append(value)
                
            for key, value in partial_obs.items():
                if key not in batch['partial_obs']:
                    batch['partial_obs'][key] = []
                batch['partial_obs'][key].append(value)
                
            for key, value in next_full_obs.items():
                if key not in batch['next_full_obs']:
                    batch['next_full_obs'][key] = []
                batch['next_full_obs'][key].append(value)
                
            for key, value in next_partial_obs.items():
                if key not in batch['next_partial_obs']:
                    batch['next_partial_obs'][key] = []
                batch['next_partial_obs'][key].append(value)
        
        # Convert lists to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for key in batch['full_obs']:
            batch['full_obs'][key] = torch.tensor(np.stack(batch['full_obs'][key]), dtype=torch.float32).to(device)
            
        for key in batch['partial_obs']:
            batch['partial_obs'][key] = torch.tensor(np.stack(batch['partial_obs'][key]), dtype=torch.float32).to(device)
            
        for key in batch['next_full_obs']:
            batch['next_full_obs'][key] = torch.tensor(np.stack(batch['next_full_obs'][key]), dtype=torch.float32).to(device)
            
        for key in batch['next_partial_obs']:
            batch['next_partial_obs'][key] = torch.tensor(np.stack(batch['next_partial_obs'][key]), dtype=torch.float32).to(device)
        
        # Process actions, rewards, and dones
        batch['action'] = torch.tensor(np.stack([t['action'] for t in transitions]), dtype=torch.float32).to(device)
        batch['reward'] = torch.tensor(np.stack([t['reward'] for t in transitions]), dtype=torch.float32).to(device)
        batch['done'] = torch.tensor(np.stack([t['done'] for t in transitions]), dtype=torch.float32).to(device)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
