import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F

class BehavioralCloning:
    def __init__(self, student_policy, teacher_policy, config):
        self.student = student_policy
        self.teacher = teacher_policy
        self.config = config
        self.device = student_policy.device  # Get device from student policy
        self.batch_size = config.bc.batch_size
        self.optimizer = optim.Adam(
            student_policy.parameters(),
            lr=config.bc.learning_rate,
            eps=1e-5
        )
        
    def train_step(self, batch):
        """Perform a single BC training step.
        
        Args:
            batch: dict containing:
                - partial_obs: dict of partial observations from student trajectories
                - full_obs: dict of full observations from student trajectories
                
        Returns:
            dict containing training metrics
        """
        # Get student's action predictions using partial observations
        student_actions, student_log_probs, _, _ = self.student.get_action_and_value(batch['partial_obs'])
        
        # Get teacher's actions using full observations
        with torch.no_grad():
            # Filter observations to only include teacher's keys
            teacher_obs = {}
            for key in self.teacher.mlp_keys + self.teacher.cnn_keys:
                if key in batch['full_obs']:
                    # Ensure the observation has the correct shape
                    obs = batch['full_obs'][key]
                    if len(obs.shape) == 2:  # [batch_size, flattened_size]
                        teacher_obs[key] = obs
                    else:  # [batch_size, *original_shape]
                        # For MLP observations, flatten all dimensions except batch
                        if key in self.teacher.mlp_keys:
                            teacher_obs[key] = obs.reshape(obs.shape[0], -1)
                        # For CNN observations, keep the image shape
                        else:
                            teacher_obs[key] = obs
            
            teacher_actions, _, _, _ = self.teacher.get_action_and_value(teacher_obs)
        
        # Compute loss
        if self.student.is_discrete:
            # For discrete actions, use cross entropy loss
            # Convert teacher's one-hot actions to class indices
            teacher_action_indices = teacher_actions.argmax(dim=1)
            # Get student's logits
            student_logits = self.student.actor(self.student.encode_observations(batch['partial_obs']))
            # Compute cross entropy loss
            loss = nn.CrossEntropyLoss()(student_logits, teacher_action_indices)
        else:
            # For continuous actions, use MSE loss
            student_actions = self.student.actor_mean(self.student.encode_observations(batch['partial_obs']))
            loss = ((student_actions - teacher_actions) ** 2).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.bc.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.config.bc.max_grad_norm)
        self.optimizer.step()
        
        # Convert tensors to scalars for metrics
        metrics = {
            'bc_loss': loss.item()
        }
        
        if not self.student.is_discrete:
            metrics['action_diff'] = (student_actions - teacher_actions).abs().mean().item()
        else:
            # For discrete actions, compute accuracy
            student_action_indices = student_actions.argmax(dim=1)
            accuracy = (student_action_indices == teacher_action_indices).float().mean().item()
            metrics['action_accuracy'] = accuracy
        
        return metrics
    
    def train(self, transitions, num_epochs=1):
        """Train the student policy using behavioral cloning.
        
        Args:
            transitions: List of transitions from teacher policy
            num_epochs: Number of epochs to train for
            
        Returns:
            dict of training metrics
        """
        # Convert transitions to tensors
        obs = {}
        for key in transitions[0]['obs'].keys():
            obs[key] = torch.tensor(np.stack([t['obs'][key] for t in transitions]), device=self.device)
        
        # Get teacher actions
        with torch.no_grad():
            teacher_actions, _, _, _, _ = self.teacher.get_action_and_value(obs)
        
        # Train student
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(transitions))
            obs_shuffled = {k: v[indices] for k, v in obs.items()}
            teacher_actions_shuffled = teacher_actions[indices]
            
            # Train in batches
            for i in range(0, len(transitions), self.batch_size):
                batch_obs = {k: v[i:i+self.batch_size] for k, v in obs_shuffled.items()}
                batch_teacher_actions = teacher_actions_shuffled[i:i+self.batch_size]
                
                # Get student's action predictions
                student_actions, student_log_probs, _, _, _ = self.student.get_action_and_value(batch_obs)
                
                # Compute loss
                if self.student.is_discrete:
                    # For discrete actions, use cross entropy loss
                    teacher_action_indices = batch_teacher_actions.argmax(dim=1)
                    student_logits = self.student.actor(self.student.encode_observations(batch_obs))
                    loss = nn.CrossEntropyLoss()(student_logits, teacher_action_indices)
                else:
                    # For continuous actions, use MSE loss
                    # Ensure student actions have gradients enabled
                    student_actions = self.student.actor_mean(self.student.encode_observations(batch_obs))
                    loss = F.mse_loss(student_actions, batch_teacher_actions)
                
                # Add imitation loss if enabled
                if self.config.encoder.student_to_teacher_imitation and self.config.encoder.student_to_teacher_lambda > 0:
                    # Get teacher's latent representation with stop gradient
                    with torch.no_grad():
                        teacher_latent = self.student.dual_encoder.encode_teacher_observations(batch_obs)
                    # Get student's latent representation
                    student_latent = self.student.encode_observations(batch_obs)
                    # Compute imitation loss
                    imitation_loss = self.student.dual_encoder.compute_student_to_teacher_loss(teacher_latent, student_latent)
                    loss = loss + self.config.encoder.student_to_teacher_lambda * imitation_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.bc.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.config.bc.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        metrics = {
            'bc_loss': total_loss / num_batches,
            'num_transitions': len(transitions),
            'num_epochs': num_epochs
        }
        
        return metrics
