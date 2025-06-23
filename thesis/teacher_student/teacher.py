import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from ..shared.agent import BaseAgent
import numpy as np
import re
from thesis.embodied import Space
from thesis.teacher_student.encoder import DualEncoder, layer_init
import torch.nn.functional as F

class TeacherPolicy(BaseAgent):
    def __init__(self, envs, config, dual_encoder):
        super().__init__(envs, config, dual_encoder)
        
        # Get observation space and keys
        obs_space = envs.obs_space
        
        # Filter keys based on the regex patterns
        self.mlp_keys = self.teacher_mlp_keys
        self.cnn_keys = self.teacher_cnn_keys
        self.all_mlp_keys = set(self.mlp_keys + self.student_mlp_keys)
        self.all_cnn_keys = set(self.cnn_keys + self.student_cnn_keys)
        
        # Check if action space is discrete
        self.is_discrete = envs.act_space['action'].discrete
    
    def encode_observations(self, x):
        """Encode observations using teacher encoder."""
        return self.dual_encoder.encode_teacher_observations(x)
    
    def get_value(self, x):
        latent = self.encode_observations(x)
        return self.critic(latent)
    
    def get_action_and_value(self, x, action=None):
        """Get action and value from policy.
        
        Args:
            x: Dictionary of observations
            action: Optional action for computing log probability
            
        Returns:
            action: Action to take
            logprob: Log probability of action
            entropy: Entropy of action distribution
            value: Value estimate
            imitation_losses: Dictionary of imitation losses
        """
        # Get base action and value
        action, logprob, entropy, value = super().get_action_and_value(x, action)
        
        # Compute imitation losses if enabled
        imitation_losses = {}
        if self.config.encoder.teacher_to_student_imitation and self.config.encoder.teacher_to_student_lambda > 0:
            # Get teacher's latent representation
            teacher_latent = self.encode_observations(x)
            # Get student's latent representation with stop gradient
            with torch.no_grad():
                student_latent = self.dual_encoder.encode_student_observations(x)
            imitation_losses['teacher_to_student'] = self.dual_encoder.compute_teacher_to_student_loss(teacher_latent, student_latent)
        
        return action, logprob, entropy, value, imitation_losses
