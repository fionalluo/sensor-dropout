"""
Shared evaluation utilities for all baselines.
This module provides evaluation functions that can be used across different baseline implementations.
"""

import numpy as np
import torch
import re


def get_eval_keys(config, eval_mode="full"):
    """Get evaluation keys for a specific mode from the flat config structure.
    
    Args:
        config: Configuration object with flat eval_keys structure
        eval_mode: Evaluation mode (e.g., "full", "no_door", "no_key", "no_door_no_key")
        
    Returns:
        dict: Dictionary with mlp_keys and cnn_keys for the specified mode
    """
    mlp_keys = getattr(config, f'eval_{eval_mode}_mlp_keys', '.*')
    cnn_keys = getattr(config, f'eval_{eval_mode}_cnn_keys', '.*')
    
    return {
        'mlp_keys': mlp_keys,
        'cnn_keys': cnn_keys
    }


def get_available_eval_modes(config):
    """Get list of available evaluation modes from the config.
    
    Args:
        config: Configuration object
        
    Returns:
        list: List of available evaluation mode names
    """
    modes = []
    for attr in dir(config):
        if attr.startswith('eval_') and attr.endswith('_mlp_keys'):
            mode = attr[5:-10]  # Remove 'eval_' prefix and '_mlp_keys' suffix
            modes.append(mode)
    return modes


def filter_observations_by_keys(obs_dict, mlp_keys_pattern, cnn_keys_pattern):
    """Filter observations based on regex patterns and substitute unprivileged keys.
    
    Args:
        obs_dict: Dictionary of observations from environment
        mlp_keys_pattern: Regex pattern for MLP keys to keep
        cnn_keys_pattern: Regex pattern for CNN keys to keep
        
    Returns:
        dict: Filtered observations with unprivileged substitutions
    """
    filtered_obs = {}
    
    # Helper function to check if a key matches a pattern
    def matches_pattern(key, pattern):
        if pattern == '.*':
            return True
        elif pattern == '^$':
            return False
        else:
            return re.search(pattern, key) is not None
    
    # First pass: collect all keys that match the patterns (these are the "available" keys)
    available_keys = set()
    
    # Process MLP keys
    for key, value in obs_dict.items():
        if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
            continue
        if matches_pattern(key, mlp_keys_pattern):
            available_keys.add(key)
    
    # Process CNN keys
    for key, value in obs_dict.items():
        if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
            continue
        if matches_pattern(key, cnn_keys_pattern):
            available_keys.add(key)
    
    # Second pass: for each key in obs_dict, if it's in available_keys, add it to filtered_obs
    for key, value in obs_dict.items():
        if key in available_keys:
            filtered_obs[key] = value
    
    return filtered_obs


def substitute_unprivileged_for_agent(agent_keys, filtered_obs_dict, obs_dict):
    """Substitute unprivileged keys for privileged keys that the agent needs.
    
    Args:
        agent_keys: List of keys the agent expects
        filtered_obs_dict: Dictionary of available filtered observations
        obs_dict: Original observation dictionary
        
    Returns:
        dict: Final observations with substitutions for the agent
    """
    final_obs = {}
    
    for key in agent_keys:
        if key in filtered_obs_dict:
            # Key is directly available
            final_obs[key] = filtered_obs_dict[key]
        else:
            # Key is not available, look for unprivileged version with prefix matching
            unprivileged_key = find_unprivileged_key(key, filtered_obs_dict)
            if unprivileged_key:
                # Use unprivileged version as substitute
                final_obs[key] = filtered_obs_dict[unprivileged_key]
            else:
                # Neither privileged nor unprivileged available, will be zeroed later
                final_obs[key] = None
    
    return final_obs


def find_unprivileged_key(privileged_key, available_keys):
    """Find the unprivileged key that matches a privileged key using prefix matching.
    
    Args:
        privileged_key: The privileged key to find a substitute for
        available_keys: Dictionary of available keys
        
    Returns:
        str or None: The matching unprivileged key, or None if not found
    """
    # Look for key that starts with 'privileged_key_unprivileged'
    prefix = f"{privileged_key}_unprivileged"
    
    for key in available_keys.keys():
        if key.startswith(prefix):
            return key
    
    return None


def evaluate_policy(agent, envs, device, config, log_video=False):
    """Evaluate a policy across multiple episodes.
    
    Args:
        agent: Agent to evaluate
        envs: Vectorized environment
        device: Device to run evaluation on
        config: Configuration object
        log_video: Whether to log video frames
        
    Returns:
        dict: Evaluation metrics
    """
    num_episodes = getattr(config.eval, 'num_episodes', 10)
    
    # Initialize episode tracking
    episode_returns = []
    episode_lengths = []
    
    # Initialize video frames if logging
    video_frames = {}
    if log_video:
        for key in envs.obs_space:
            if len(envs.obs_space[key].shape) == 3 and envs.obs_space[key].shape[-1] == 3:
                video_frames[key] = []
    
    # Get all observation keys
    all_keys = []
    for key in envs.obs_space:
        if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
            all_keys.append(key)
    
    # Initialize observations
    obs = {}
    for key in all_keys:
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
    
    # Initialize LSTM state for RNN agents
    lstm_state = None
    if hasattr(agent, 'get_initial_lstm_state'):
        try:
            # Try calling with num_envs argument first (for PPO-RNN agents)
            initial_state = agent.get_initial_lstm_state(envs.num_envs)
            lstm_state = (
                initial_state[0].expand(-1, envs.num_envs, -1),
                initial_state[1].expand(-1, envs.num_envs, -1)
            )
        except TypeError:
            # If that fails, try calling without arguments (for PPO agents that return None)
            initial_state = agent.get_initial_lstm_state()
            if initial_state is not None:
                lstm_state = (
                    initial_state[0].expand(-1, envs.num_envs, -1),
                    initial_state[1].expand(-1, envs.num_envs, -1)
                )
    
    # Run evaluation until we have enough episodes
    while len(episode_returns) < num_episodes:
        # Get action from policy
        with torch.no_grad():
            if hasattr(agent, 'get_action_and_value'):
                # For PPO-style agents
                if hasattr(agent, 'get_initial_lstm_state') and lstm_state is not None:
                    # PPO RNN agent: needs lstm_state and done
                    result = agent.get_action_and_value(next_obs, lstm_state, next_done)
                    if len(result) == 5:
                        # PPO RNN agent: (action, log_prob, entropy, value, new_lstm_state)
                        action, _, _, _, lstm_state = result
                    elif len(result) == 6:
                        # PPO Distill agent: (action, log_prob, entropy, value, new_lstm_state, expert_actions)
                        action, _, _, _, lstm_state, _ = result
                    else:
                        # Fallback: just take the first value as action
                        action = result[0]
                else:
                    # Regular PPO agent
                    result = agent.get_action_and_value(next_obs)
                    if len(result) == 4:
                        # Regular PPO agent: (action, log_prob, entropy, value)
                        action, _, _, _ = result
                    else:
                        # Fallback: just take the first value as action
                        action = result[0]
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
        
        # Store video frames for the first episode only (to log 1 video per evaluation)
        if log_video and len(episode_returns) == 0:  # Only for the first episode
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
    
    if log_video and video_frames:
        metrics['video_frames'] = video_frames
    
    return metrics


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


def evaluate_agent_with_observation_subsets(agent, envs, device, config, make_envs_func=None, 
                                          writer=None, use_wandb=False, global_step=0, debug=False):
    """Evaluate agent across multiple observation subsets defined in eval_keys.
    
    Args:
        agent: Agent to evaluate
        envs: Vectorized environment
        device: Device to run evaluation on
        config: Configuration object with eval_keys structure
        make_envs_func: Function to create evaluation environments
        writer: TensorBoard writer (optional)
        use_wandb: Whether to use wandb logging
        global_step: Current global step for logging
        debug: Whether to print detailed debugging information
        
    Returns:
        dict: Combined evaluation metrics
    """
    if not hasattr(config, 'eval') or not hasattr(config.eval, 'num_eval_configs'):
        print("Warning: No eval.num_eval_configs found in config, skipping subset evaluation")
        return {}
    
    num_eval_configs = config.eval.num_eval_configs
    env_metrics = {}
    
    if debug:
        print(f"Running evaluation across {num_eval_configs} observation subsets...")
    
    all_episode_returns = []
    all_episode_lengths = []
    
    for subset_idx in range(1, num_eval_configs + 1):
        env_name = f"env{subset_idx}"
        
        # Get eval keys for this environment
        if hasattr(config, 'eval_keys') and hasattr(config.eval_keys, env_name):
            eval_keys = getattr(config.eval_keys, env_name)
            mlp_keys_pattern = eval_keys.mlp_keys
            cnn_keys_pattern = eval_keys.cnn_keys
        else:
            if debug:
                print(f"Warning: No eval_keys.{env_name} found, using default patterns")
            mlp_keys_pattern = '.*'
            cnn_keys_pattern = '.*'
        
        if debug:
            print(f"\n=== {env_name} Configuration ===")
            print(f"MLP keys pattern: {mlp_keys_pattern}")
            print(f"CNN keys pattern: {cnn_keys_pattern}")
        
        # Create evaluation environments
        if make_envs_func is None:
            raise ValueError("make_envs_func must be provided")
        eval_envs = make_envs_func(config, num_envs=config.eval.eval_envs)
        eval_envs.num_envs = config.eval.eval_envs
        
        # If this is env1, compare with training configuration
        if env_name == "env1" and debug:
            print(f"\n--- Comparing env1 with training configuration ---")
            print(f"Training MLP keys: {agent.mlp_keys}")
            print(f"Training CNN keys: {agent.cnn_keys}")
            print(f"Training full_keys MLP pattern: {getattr(config.full_keys, 'mlp_keys', 'N/A')}")
            print(f"Training full_keys CNN pattern: {getattr(config.full_keys, 'cnn_keys', 'N/A')}")
            
            # Check if patterns are equivalent
            training_mlp_matches = []
            training_cnn_matches = []
            
            for key in eval_envs.obs_space.keys():
                if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    if re.search(getattr(config.full_keys, 'mlp_keys', '.*'), key):
                        training_mlp_matches.append(key)
                    if re.search(getattr(config.full_keys, 'cnn_keys', '.*'), key):
                        training_cnn_matches.append(key)
            
            print(f"Training would match MLP keys: {training_mlp_matches}")
            print(f"Training would match CNN keys: {training_cnn_matches}")
        
        # Print available observation keys in environment
        if debug:
            print(f"Available observation keys in environment:")
            for key, space in eval_envs.obs_space.items():
                if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    print(f"  {key}: {space.shape}")
            
            # Print agent's expected keys
            print(f"Agent expects these keys:")
            print(f"  MLP keys: {agent.mlp_keys}")
            print(f"  CNN keys: {agent.cnn_keys}")
        
        # Run evaluation with observation filtering
        env_episode_returns = []
        env_episode_lengths = []
        num_episodes = config.eval.num_eval_episodes
        
        # Initialize video logging for this environment
        video_frames = {key: [] for key in getattr(config, 'log_keys_video', [])}
        
        # Initialize observation storage
        obs = {}
        all_keys = agent.mlp_keys + agent.cnn_keys
        for key in all_keys:
            if key in eval_envs.obs_space:
                if len(eval_envs.obs_space[key].shape) == 3 and eval_envs.obs_space[key].shape[-1] == 3:
                    obs[key] = torch.zeros((eval_envs.num_envs,) + eval_envs.obs_space[key].shape).to(device)
                else:
                    size = np.prod(eval_envs.obs_space[key].shape)
                    obs[key] = torch.zeros((eval_envs.num_envs, size)).to(device)
        
        # Initialize actions
        action_shape = eval_envs.act_space['action'].shape
        acts = {
            'action': np.zeros((eval_envs.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(eval_envs.num_envs, dtype=bool)
        }
        
        # Get initial observations
        obs_dict = eval_envs.step(acts)
        next_obs = {}
        
        # Filter and substitute observations using prefix-based approach
        filtered_obs_dict = filter_observations_by_keys(obs_dict, mlp_keys_pattern, cnn_keys_pattern)
        final_obs_dict = substitute_unprivileged_for_agent(all_keys, filtered_obs_dict, obs_dict)
        
        if debug:
            print(f"After filtering, available keys:")
            for key in filtered_obs_dict.keys():
                if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    print(f"  {key}: {filtered_obs_dict[key].shape}")
        
        if debug:
            print(f"Keys that will be zeroed out (agent expects but not in filtered):")
            for key in all_keys:
                if final_obs_dict[key] is None:
                    print(f"  {key}")
            
            # Debug: Show what the agent will actually receive
            print(f"Final observation keys that agent will receive:")
            for key in all_keys:
                if final_obs_dict[key] is not None:
                    unprivileged_key = find_unprivileged_key(key, filtered_obs_dict)
                    if unprivileged_key and key not in filtered_obs_dict:
                        print(f"  {key}: {final_obs_dict[key].shape} (substituted from {unprivileged_key})")
                    else:
                        print(f"  {key}: {final_obs_dict[key].shape} (direct)")
                else:
                    print(f"  {key}: zeroed out")
        
        for key in all_keys:
            if final_obs_dict[key] is not None:
                next_obs[key] = torch.Tensor(final_obs_dict[key].astype(np.float32)).to(device)
            else:
                # Zero out missing observations
                if key in obs:
                    next_obs[key] = torch.zeros_like(obs[key])
        
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
        
        # Track episode returns and lengths
        env_returns = np.zeros(eval_envs.num_envs)
        env_lengths = np.zeros(eval_envs.num_envs)
        
        # Initialize LSTM state for RNN agents
        lstm_state = None
        if hasattr(agent, 'get_initial_lstm_state'):
            try:
                # Try calling with num_envs argument first (for PPO-RNN agents)
                initial_state = agent.get_initial_lstm_state(eval_envs.num_envs)
                lstm_state = (
                    initial_state[0].expand(-1, eval_envs.num_envs, -1),
                    initial_state[1].expand(-1, eval_envs.num_envs, -1)
                )
            except TypeError:
                # If that fails, try calling without arguments (for PPO agents that return None)
                initial_state = agent.get_initial_lstm_state()
                if initial_state is not None:
                    lstm_state = (
                        initial_state[0].expand(-1, eval_envs.num_envs, -1),
                        initial_state[1].expand(-1, eval_envs.num_envs, -1)
                    )
        
        # Run evaluation until we have enough episodes
        while len(env_episode_returns) < num_episodes:
            # Get action from policy
            with torch.no_grad():
                if hasattr(agent, 'get_action_and_value'):
                    if hasattr(agent, 'get_initial_lstm_state') and lstm_state is not None:
                        # PPO RNN agent: needs lstm_state and done
                        result = agent.get_action_and_value(next_obs, lstm_state, next_done)
                        if len(result) == 5:
                            # PPO RNN agent: (action, log_prob, entropy, value, new_lstm_state)
                            action, _, _, _, lstm_state = result
                        elif len(result) == 6:
                            # PPO Distill agent: (action, log_prob, entropy, value, new_lstm_state, expert_actions)
                            action, _, _, _, lstm_state, _ = result
                        else:
                            # Fallback: just take the first value as action
                            action = result[0]
                    else:
                        # Regular PPO agent
                        result = agent.get_action_and_value(next_obs)
                        if len(result) == 4:
                            # Regular PPO agent: (action, log_prob, entropy, value)
                            action, _, _, _ = result
                        else:
                            # Fallback: just take the first value as action
                            action = result[0]
                else:
                    action = agent.get_action(next_obs)
            
            # Step environment
            action_np = action.cpu().numpy()
            if hasattr(agent, 'is_discrete') and agent.is_discrete:
                action_np = action_np.reshape(eval_envs.num_envs, -1)
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = eval_envs.step(acts)
            
            # Filter and substitute observations using prefix-based approach
            filtered_obs_dict = filter_observations_by_keys(obs_dict, mlp_keys_pattern, cnn_keys_pattern)
            final_obs_dict = substitute_unprivileged_for_agent(all_keys, filtered_obs_dict, obs_dict)
            
            # Store video frames for the first episode only (to log 1 video per evaluation)
            # Use filtered observations to show what the agent actually sees
            if len(env_episode_returns) == 0:  # Only for the first episode
                for key in video_frames.keys():
                    if key in final_obs_dict and final_obs_dict[key] is not None:
                        # Use the filtered/substituted observation
                        video_frames[key].append(final_obs_dict[key][0].copy())
                    elif key in obs_dict:
                        # Fallback to original if not in filtered
                        video_frames[key].append(obs_dict[key][0].copy())
            
            # Process observations
            for key in all_keys:
                if final_obs_dict[key] is not None:
                    next_obs[key] = torch.Tensor(final_obs_dict[key].astype(np.float32)).to(device)
                else:
                    # Zero out missing observations
                    if key in obs:
                        next_obs[key] = torch.zeros_like(obs[key])
            
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
            
            # Update episode tracking
            for env_idx in range(eval_envs.num_envs):
                env_returns[env_idx] += obs_dict['reward'][env_idx]
                env_lengths[env_idx] += 1
                
                if obs_dict['is_last'][env_idx]:
                    if len(env_episode_returns) < num_episodes:
                        env_episode_returns.append(env_returns[env_idx])
                        env_episode_lengths.append(env_lengths[env_idx])
                    
                    env_returns[env_idx] = 0
                    env_lengths[env_idx] = 0
        
        # Calculate metrics for this environment
        env_episode_returns = np.array(env_episode_returns)
        env_episode_lengths = np.array(env_episode_lengths)
        
        # Collect for overall stats
        all_episode_returns.extend(env_episode_returns.tolist())
        all_episode_lengths.extend(env_episode_lengths.tolist())
        
        env_metrics[env_name] = {
            'mean_return': np.mean(env_episode_returns),
            'std_return': np.std(env_episode_returns),
            'mean_length': np.mean(env_episode_lengths),
            'std_length': np.std(env_episode_lengths)
        }
        
        print(f"  {env_name}: mean_return={env_metrics[env_name]['mean_return']:.2f}, "
              f"std_return={env_metrics[env_name]['std_return']:.2f}")
        
        # Log metrics with full_eval prefix for subset evaluations
        if writer is not None:
            writer.add_scalar(f"full_eval/{env_name}/mean_return", env_metrics[env_name]['mean_return'], global_step)
            writer.add_scalar(f"full_eval/{env_name}/std_return", env_metrics[env_name]['std_return'], global_step)
            writer.add_scalar(f"full_eval/{env_name}/mean_length", env_metrics[env_name]['mean_length'], global_step)
            writer.add_scalar(f"full_eval/{env_name}/std_length", env_metrics[env_name]['std_length'], global_step)
        
        if use_wandb:
            import wandb
            wandb.log({
                f"full_eval/{env_name}/mean_return": env_metrics[env_name]['mean_return'],
                f"full_eval/{env_name}/std_return": env_metrics[env_name]['std_return'],
                f"full_eval/{env_name}/mean_length": env_metrics[env_name]['mean_length'],
                f"full_eval/{env_name}/std_length": env_metrics[env_name]['std_length'],
            }, step=global_step)
        
        # Log video if available - use full_eval prefix for subset videos
        if video_frames and any(video_frames.values()) and writer is not None:
            for key, frames in video_frames.items():
                if frames:
                    # Convert frames to video format
                    video_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
                    video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                    writer.add_video(f"full_eval/{env_name}/{key}", video_tensor, global_step, fps=4)
        
        eval_envs.close()
    
    # Print overall mean and std return across all evaluation environments
    if all_episode_returns:
        overall_mean = np.mean(all_episode_returns)
        overall_std = np.std(all_episode_returns)
        print(f"  OVERALL: mean_return={overall_mean:.2f}, std_return={overall_std:.2f}")
    
    # Return only the individual environment metrics for subset evaluations
    # No overall metrics should be calculated for subset evaluations
    return env_metrics


def evaluate_agent(agent, envs, device, config, log_video=False, make_envs_func=None, writer=None, use_wandb=False, global_step=0):
    """Run evaluation and log metrics.
    
    Args:
        agent: Agent to evaluate
        envs: Vectorized environment
        device: Device to run evaluation on
        config: Configuration object
        log_video: Whether to log video frames
        make_envs_func: Function to create evaluation environments
        writer: TensorBoard writer (optional)
        use_wandb: Whether to use wandb logging
        global_step: Current global step for logging
        
    Returns:
        dict of evaluation metrics
    """
    if envs is None:
        # Create evaluation environments
        if make_envs_func is None:
            raise ValueError("make_envs_func must be provided if envs is None")
        envs = make_envs_func(config, num_envs=config.eval.eval_envs)
        envs.num_envs = config.eval.eval_envs
    
    # Run evaluation using shared function
    eval_metrics = evaluate_policy(agent, envs, device, config, log_video=log_video)
    
    # Log evaluation metrics using shared function
    eval_metrics_dict = {
        "eval/mean_return": eval_metrics['mean_return'],
        "eval/std_return": eval_metrics['std_return'],
        "eval/mean_length": eval_metrics['mean_length'],
        "eval/std_length": eval_metrics['std_length'],
    }
    log_evaluation_metrics(eval_metrics_dict, global_step, use_wandb, writer)
    
    # Log video if available using shared function
    if log_video and 'video_frames' in eval_metrics and eval_metrics['video_frames']:
        log_evaluation_videos(eval_metrics['video_frames'], global_step, use_wandb, prefix="eval_")
    
    return eval_metrics


def run_periodic_evaluation(agent, config, device, global_step, last_eval, eval_envs, 
                          make_envs_func=None, writer=None, use_wandb=False, debug=False):
    """Run periodic evaluation if enough steps have passed.
    
    Args:
        agent: Agent to evaluate
        config: Configuration object
        device: Device to run evaluation on
        global_step: Current global step
        last_eval: Last evaluation step
        eval_envs: Evaluation environments
        make_envs_func: Function to create evaluation environments
        writer: TensorBoard writer (optional)
        use_wandb: Whether to use wandb logging
        debug: Whether to print detailed debugging information
        
    Returns:
        tuple: (new_last_eval, eval_envs)
    """
    if global_step - last_eval >= config.eval.eval_interval * config.num_envs:
        # Determine if we should log video based on video_log_interval
        eval_count = global_step // (config.eval.eval_interval * config.num_envs)
        log_video = (eval_count % config.eval.video_log_interval == 0)
        
        print(f"Running evaluation at step {global_step}...")
        
        # Run original evaluation (full environment)
        eval_metrics = evaluate_agent(
            agent, eval_envs, device, config, 
            log_video=log_video, 
            make_envs_func=make_envs_func,
            writer=writer,
            use_wandb=use_wandb,
            global_step=global_step
        )
        
        # Run subset evaluation (observation subsets)
        subset_metrics = evaluate_agent_with_observation_subsets(
            agent, eval_envs, device, config,
            make_envs_func=make_envs_func,
            writer=writer,
            use_wandb=use_wandb,
            global_step=global_step,
            debug=debug
        )
        
        last_eval = global_step
    
    return last_eval, eval_envs


def run_initial_evaluation(agent, config, device, make_envs_func, writer=None, use_wandb=False, debug=False):
    """Run initial evaluation at the start of training.
    
    Args:
        agent: Agent to evaluate
        config: Configuration object
        device: Device to run evaluation on
        make_envs_func: Function to create evaluation environments
        writer: TensorBoard writer (optional)
        use_wandb: Whether to use wandb logging
        debug: Whether to print detailed debugging information
        
    Returns:
        tuple: (eval_envs, eval_metrics)
    """
    # Create evaluation environments once
    eval_envs = make_envs_func(config, num_envs=config.eval.eval_envs)
    eval_envs.num_envs = config.eval.eval_envs
    
    # Run initial evaluation (full environment)
    print("Running initial evaluation...")
    eval_metrics = evaluate_agent(
        agent, eval_envs, device, config, 
        log_video=True, 
        make_envs_func=make_envs_func,
        writer=writer,
        use_wandb=use_wandb,
        global_step=0
    )
    
    # Run initial subset evaluation (observation subsets)
    print("Running initial subset evaluation...")
    subset_metrics = evaluate_agent_with_observation_subsets(
        agent, eval_envs, device, config,
        make_envs_func=make_envs_func,
        writer=writer,
        use_wandb=use_wandb,
        global_step=0,
        debug=debug
    )
    
    return eval_envs, eval_metrics 