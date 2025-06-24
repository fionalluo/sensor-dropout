"""
Shared evaluation utilities for all baselines.
This module provides evaluation functions that can be used across different baseline implementations.
"""

import numpy as np
import torch


def evaluate_policy(agent, envs, device, config, log_video=False):
    """Evaluate a policy for a specified number of episodes.
    
    Args:
        agent: Agent to evaluate (must have get_action_and_value method)
        envs: Vectorized environment
        device: Device to run evaluation on
        config: Configuration object with eval settings
        log_video: Whether to log video frames
        
    Returns:
        dict of evaluation metrics
    """
    # Initialize metrics
    episode_returns = []
    episode_lengths = []
    num_episodes = config.eval.num_eval_episodes
    
    # Initialize video logging if enabled
    video_frames = {key: [] for key in envs.obs_space.keys() if key in config.log_keys_video} if log_video else {}
    
    # Initialize observation storage
    obs = {}
    # Get observation keys from agent
    if hasattr(agent, 'mlp_keys') and hasattr(agent, 'cnn_keys'):
        all_keys = agent.mlp_keys + agent.cnn_keys
    elif hasattr(agent, 'dual_encoder'):
        # For teacher-student agents
        all_keys = set(
            agent.dual_encoder.student_encoder.mlp_keys + 
            agent.dual_encoder.student_encoder.cnn_keys +
            agent.dual_encoder.teacher_encoder.mlp_keys +
            agent.dual_encoder.teacher_encoder.cnn_keys
        )
    else:
        # Fallback: use all available keys
        all_keys = list(envs.obs_space.keys())
    
    for key in all_keys:
        if key in envs.obs_space:
            if len(envs.obs_space[key].shape) == 3 and envs.obs_space[key].shape[-1] == 3:  # Image observations
                obs[key] = torch.zeros((envs.num_envs,) + envs.obs_space[key].shape).to(device)
            else:  # Non-image observations
                size = np.prod(envs.obs_space[key].shape)
                obs[key] = torch.zeros((envs.num_envs, size)).to(device)
    
    # Initialize actions with zeros and reset flags
    action_shape = envs.act_space['action'].shape
    acts = {
        'action': np.zeros((envs.num_envs,) + action_shape, dtype=np.float32),
        'reset': np.ones(envs.num_envs, dtype=bool)
    }
    
    # Get initial observations
    obs_dict = envs.step(acts)
    next_obs = {}
    for key in all_keys:
        if key in obs_dict:
            next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
    
    # Track episode returns and lengths for each environment
    env_returns = np.zeros(envs.num_envs)
    env_lengths = np.zeros(envs.num_envs)
    
    # Run evaluation until we have enough episodes
    while len(episode_returns) < num_episodes:
        # Get action from policy
        with torch.no_grad():
            if hasattr(agent, 'get_action_and_value'):
                # For PPO-style agents
                action, _, _, _ = agent.get_action_and_value(next_obs)
            else:
                # For other agent types
                action = agent.get_action(next_obs)
        
        # Step environment
        action_np = action.cpu().numpy()
        if hasattr(agent, 'is_discrete') and agent.is_discrete:
            action_np = action_np.reshape(envs.num_envs, -1)
        
        acts = {
            'action': action_np,
            'reset': next_done.cpu().numpy()
        }
        
        obs_dict = envs.step(acts)
        
        # Store video frames if logging
        if log_video:
            for key in video_frames.keys():
                if key in obs_dict:
                    video_frames[key].append(obs_dict[key][0].copy())
        
        # Process observations
        for key in all_keys:
            if key in obs_dict:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
        
        # Update episode tracking
        for env_idx in range(envs.num_envs):
            env_returns[env_idx] += obs_dict['reward'][env_idx]
            env_lengths[env_idx] += 1
            
            if obs_dict['is_last'][env_idx]:
                # Store episode metrics if we haven't collected enough episodes yet
                if len(episode_returns) < num_episodes:
                    episode_returns.append(env_returns[env_idx])
                    episode_lengths.append(env_lengths[env_idx])
                
                # Reset environment tracking
                env_returns[env_idx] = 0
                env_lengths[env_idx] = 0
    
    # Convert lists to numpy arrays for statistics
    episode_returns = np.array(episode_returns)
    episode_lengths = np.array(episode_lengths)
    
    metrics = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    if log_video:
        metrics['video_frames'] = video_frames
    
    return metrics


def process_video_frames(frames, key):
    """Process frames for video logging following exact format requirements."""
    if len(frames.shape) == 3:  # Single image [H, W, C]
        # Check if the last dimension is 3 (RGB image) and the maximum value is greater than 1
        if frames.shape[-1] == 3 and np.max(frames) > 1:
            return frames  # Directly pass the image without modification
        else:
            frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
            frames = np.transpose(frames, [2, 0, 1])
            return frames
    elif len(frames.shape) == 4:  # Video [T, H, W, C]
        # Sanity check that the channels dimension is last
        assert frames.shape[3] in [1, 3, 4], f"Invalid shape: {frames.shape}"
        is_depth = frames.shape[3] == 1
        frames = np.transpose(frames, [0, 3, 1, 2])
        # If the video is a float, convert it to uint8
        if np.issubdtype(frames.dtype, np.floating):
            if is_depth:
                frames = frames - frames.min()
                # Scale by 2 mean distances of near rays
                frames = frames / (2 * frames[frames <= 1].mean())
                # Scale to [0, 255]
                frames = np.clip(frames, 0, 1)
                # repeat channel dimension 3 times
                frames = np.repeat(frames, 3, axis=1)
            frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
        return frames
    else:
        raise ValueError(f"Unexpected shape for {key}: {frames.shape}")


def log_evaluation_metrics(metrics_dict, step, use_wandb=False, writer=None):
    """Log evaluation metrics to wandb and/or tensorboard.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Current step for logging
        use_wandb: Whether to log to wandb
        writer: TensorBoard writer (optional)
    """
    # Log to TensorBoard
    if writer is not None:
        for key, value in metrics_dict.items():
            if key != 'video_frames':  # Don't log video frames to tensorboard
                writer.add_scalar(key, value, step)
    
    # Log to wandb
    if use_wandb:
        import wandb
        wandb_metrics = {key: value for key, value in metrics_dict.items() if key != 'video_frames'}
        wandb.log(wandb_metrics, step=step)


def log_evaluation_videos(video_frames, step, use_wandb=False, prefix=""):
    """Log evaluation videos to wandb.
    
    Args:
        video_frames: Dictionary of video frames
        step: Current step for logging
        use_wandb: Whether to log to wandb
        prefix: Prefix for video keys (e.g., "teacher_", "student_")
    """
    if not use_wandb or not video_frames:
        return
    
    import wandb
    
    for key, frames in video_frames.items():
        if frames:
            frames = np.stack(frames)
            processed_frames = process_video_frames(frames, key)
            
            video_key = f"videos/{prefix}{key}" if prefix else f"videos/eval_{key}"
            wandb.log({
                video_key: wandb.Video(
                    processed_frames,
                    fps=10,
                    format="gif"
                )
            }, step=step) 