import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from ..shared.agent import BaseAgent
import numpy as np
import re
from thesis.embodied import Space
from thesis.teacher_student.encoder import DualEncoder, layer_init

class StudentPolicy(BaseAgent):
    def __init__(self, envs, config, dual_encoder):
        super().__init__(envs, config, dual_encoder)
        
        # Get all observation keys
        all_keys = list(envs.obs_space.keys())
        
        obs_space = envs.obs_space
        
        # Match keys against regex patterns
        self.mlp_keys = self.student_mlp_keys
        self.cnn_keys = self.student_cnn_keys
        self.all_mlp_keys = set(self.mlp_keys + self.teacher_mlp_keys)
        self.all_cnn_keys = set(self.cnn_keys + self.teacher_cnn_keys)
        
        # Check if action space is discrete
        self.is_discrete = envs.act_space['action'].discrete
    
    def encode_observations(self, x):
        """Encode observations using student encoder."""
        return self.dual_encoder.encode_student_observations(x)
    
    def get_value(self, x):
        latent = self.encode_observations(x)
        return self.critic(latent)
    
    def get_action_and_value(self, x, action=None):
        """Get action and value from student policy, with additional imitation loss.
        
        Args:
            x: dict of observations
            action: optional action for computing log probability
            
        Returns:
            tuple of (action, log_prob, entropy, value, imitation_losses)
        """
        # Get base action and value from parent class
        action, log_prob, entropy, value = super().get_action_and_value(x, action)
        
        # Calculate imitation loss if enabled and lambda > 0
        imitation_losses = {}
        if self.config.encoder.student_to_teacher_imitation and self.config.encoder.student_to_teacher_lambda > 0:
            # Get teacher's latent representation with stop gradient
            with torch.no_grad():
                teacher_latent = self.dual_encoder.encode_teacher_observations(x)
            # Get student's latent representation
            student_latent = self.encode_observations(x)
            # Compute imitation loss
            imitation_losses['student_to_teacher'] = self.dual_encoder.compute_student_to_teacher_loss(teacher_latent, student_latent)
        
        return action, log_prob, entropy, value, imitation_losses

    def collect_transitions(self, envs, num_steps):
        """Collect transitions from the environment using the student policy.
        
        Args:
            envs: Vectorized environment
            num_steps: Number of steps to collect
            
        Returns:
            list of transitions, each containing:
                - obs: dict of observations (both partial and full)
                - action: action taken (raw index for discrete actions)
                - reward: reward received
                - next_obs: dict of next observations (both partial and full)
                - done: whether episode ended
        """
        transitions = []
        
        # Get all keys needed for both student and teacher
        all_keys = self.all_mlp_keys | self.all_cnn_keys
        # print("all_keys", all_keys)
        
        # Initialize observation storage
        obs = {}
        for key in self.all_mlp_keys:
            size = np.prod(envs.obs_space[key].shape)
            obs[key] = torch.zeros((num_steps, self.config.num_envs, size)).to(self.device)
        for key in self.all_cnn_keys:  # CNN keys
            obs[key] = torch.zeros((num_steps, self.config.num_envs) + envs.obs_space[key].shape).to(self.device)
        
        # Initialize action storage - storing raw indices for discrete actions
        if self.is_discrete:
            action_shape = (num_steps, self.config.num_envs)
        else:
            action_shape = (num_steps, self.config.num_envs) + envs.act_space['action'].shape
        actions = torch.zeros(action_shape).to(self.device)
        
        # Initialize reward and done storage
        rewards = torch.zeros((num_steps, self.config.num_envs)).to(self.device)
        dones = torch.zeros((num_steps, self.config.num_envs)).to(self.device)
        
        # Initialize actions with zeros and reset flags
        action_shape = envs.act_space['action'].shape
        acts = {
            'action': np.zeros((self.config.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(self.config.num_envs, dtype=bool)  # Reset all environments initially
        }
        
        # Get initial observations using step with reset flags
        obs_dict = envs.step(acts)
        next_obs = {}
        for key in all_keys:
            next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
        
        # Collect transitions
        for step in range(num_steps):
            # Store observations
            for key in all_keys:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            
            # Get action from policy using only student's observation keys
            with torch.no_grad():
                # Flatten batch and num_envs dimensions for encoder
                flattened_obs = {}
                for key in all_keys:
                    # Reshape to (batch * num_envs, ...)
                    flattened_obs[key] = obs[key][step].reshape(-1, *obs[key][step].shape[1:])
                
                action, _, _, _, _ = self.get_action_and_value(flattened_obs)
                # For discrete actions, convert from one-hot to index
                if self.is_discrete:
                    action = torch.argmax(action, dim=-1)
                actions[step] = action
            
            # Step environment
            action_np = action.cpu().numpy()
            if self.is_discrete:
                # Convert to one-hot for environment step
                action_np = np.eye(envs.act_space['action'].shape[0])[action_np]
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = envs.step(acts)
            
            # Process observations
            for key in all_keys:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
            rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(self.device)
            
            # Store transition
            for env_idx in range(self.config.num_envs):
                transition = {
                    'obs': {key: obs[key][step, env_idx].cpu().numpy() for key in all_keys},
                    'action': actions[step, env_idx].cpu().numpy(),
                    'reward': rewards[step, env_idx].cpu().numpy(),
                    'next_obs': {key: next_obs[key][env_idx].cpu().numpy() for key in all_keys},
                    'done': next_done[env_idx].cpu().numpy()
                }
                transitions.append(transition)
        
        return transitions
