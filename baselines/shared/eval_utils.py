"""
Shared evaluation utilities for all baselines.
This module provides evaluation functions that can be used across different baseline implementations.
"""

import numpy as np
import torch
import re
from baselines.shared.masking_utils import mask_observations_for_student


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
                if hasattr(agent, 'get_initial_lstm_state') and lstm_state is not None:
                    # PPO RNN agent: needs lstm_state and done
                    result = agent.get_action_and_value(next_obs, lstm_state, next_done)
                    if len(result) == 5:
                        # PPO RNN agent: (action, log_prob, entropy, value, new_lstm_state)
                        action, _, _, _, lstm_state = result
                    elif len(result) == 6:
                        # PPO Distill agent (old): (action, log_prob, entropy, value, new_lstm_state, expert_actions)
                        action, _, _, _, lstm_state, _ = result
                    elif len(result) == 7:
                        # PPO Distill agent (new): (action, log_prob, entropy, value, new_lstm_state, expert_actions, student_logits)
                        action, _, _, _, lstm_state, _, _ = result
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


def evaluate_agent_with_observation_subsets(agent, envs, device, config, make_envs_func=None, writer=None, use_wandb=False, global_step=0, debug=False):
    """
    Evaluate agent across multiple observation subsets defined in eval_keys.
    For each envN, mask observations to the keys specified in config.eval_keys.envN.
    The student always receives all training keys (agent.mlp_keys + agent.cnn_keys), zeroing out missing ones.
    Fail fast if any config is missing. No fallbacks, no hardcoded logic.
    """
    if not hasattr(config, 'eval_keys') or not hasattr(config, 'eval') or not hasattr(config.eval, 'num_eval_configs'):
        raise RuntimeError("Missing eval_keys or num_eval_configs in config!")
    num_eval_configs = config.eval.num_eval_configs
    env_metrics = {}

    # The full set of student keys (in order) from training
    if not hasattr(agent, 'mlp_keys') or not hasattr(agent, 'cnn_keys'):
        raise RuntimeError("Agent is missing mlp_keys or cnn_keys!")
    student_keys = agent.mlp_keys + agent.cnn_keys

    for subset_idx in range(1, num_eval_configs + 1):
        env_name = f"env{subset_idx}"
        if not hasattr(config.eval_keys, env_name):
            raise RuntimeError(f"Missing eval_keys for {env_name} in config!")
        eval_keys = getattr(config.eval_keys, env_name)
        mlp_keys_pattern = getattr(eval_keys, 'mlp_keys', None)
        cnn_keys_pattern = getattr(eval_keys, 'cnn_keys', None)
        if mlp_keys_pattern is None or cnn_keys_pattern is None:
            raise RuntimeError(f"Missing mlp_keys or cnn_keys for {env_name} in config!")

        # Create evaluation environments
        if make_envs_func is None:
            raise RuntimeError("make_envs_func must be provided!")
        eval_envs = make_envs_func(config, num_envs=config.eval.eval_envs)
        eval_envs.num_envs = config.eval.eval_envs

        num_episodes = config.eval.num_eval_episodes
        env_episode_returns = []
        env_episode_lengths = []
        video_frames = {key: [] for key in getattr(config, 'log_keys_video', [])}

        # Initialize actions
        action_shape = eval_envs.act_space['action'].shape
        acts = {
            'action': np.zeros((eval_envs.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(eval_envs.num_envs, dtype=bool)
        }
        obs_dict = eval_envs.step(acts)
        
        # Now parse teacher_keys from the current envN's patterns using actual available keys
        def parse_keys(pattern, available_keys):
            """Filter available keys based on regex pattern."""
            if pattern == '.*':
                return available_keys
            elif pattern == '^$':
                return []
            else:
                import re
                matched_keys = [k for k in available_keys if re.search(pattern, k)]
                print(f"[REGEX DEBUG] Pattern: '{pattern}'")
                print(f"[REGEX DEBUG] Available keys: {available_keys}")
                print(f"[REGEX DEBUG] Matched keys: {matched_keys}")
                return matched_keys
        
        # Get available keys from the first observation
        available_keys = [k for k in obs_dict.keys() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']]
        print(f"[REGEX DEBUG] All available keys: {available_keys}")
        
        teacher_keys = parse_keys(mlp_keys_pattern, available_keys) + parse_keys(cnn_keys_pattern, available_keys)
        
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)

        # Track episode returns and lengths
        env_returns = np.zeros(eval_envs.num_envs)
        env_lengths = np.zeros(eval_envs.num_envs)
        lstm_state = None
        if hasattr(agent, 'get_initial_lstm_state'):
            try:
                initial_state = agent.get_initial_lstm_state(eval_envs.num_envs)
                lstm_state = (
                    initial_state[0].expand(-1, eval_envs.num_envs, -1),
                    initial_state[1].expand(-1, eval_envs.num_envs, -1)
                )
            except TypeError:
                initial_state = agent.get_initial_lstm_state()
                if initial_state is not None:
                    lstm_state = (
                        initial_state[0].expand(-1, eval_envs.num_envs, -1),
                        initial_state[1].expand(-1, eval_envs.num_envs, -1)
                    )

        first_episode = True
        while len(env_episode_returns) < num_episodes:
            # Convert obs_dict to tensors
            obs_tensors = {}
            for key, value in obs_dict.items():
                if key not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    obs_tensors[key] = torch.tensor(value, device=device, dtype=torch.float32)
                else:
                    obs_tensors[key] = value
            # Mask to the full set of student_keys, using teacher_keys from the current envN
            masked_obs = mask_observations_for_student(obs_tensors, student_keys, teacher_keys, device=device, debug=debug)

            # Debug logging for the first episode of each envN
            if first_episode:
                print(f"[DEBUG][{env_name}] Teacher keys: {teacher_keys}")
                print(f"[DEBUG][{env_name}] Student keys: {student_keys}")
                print(f"[DEBUG][{env_name}] Masked obs keys: {list(masked_obs.keys())}")
                for k in masked_obs:
                    v = masked_obs[k]
                    if isinstance(v, torch.Tensor):
                        print(f"  key: {k}, shape: {tuple(v.shape)}, mean: {v.float().mean().item():.4f}, std: {v.float().std().item():.4f}, min: {v.float().min().item():.4f}, max: {v.float().max().item():.4f}")
                    else:
                        print(f"  key: {k}, value: {v}")
                first_episode = False

            # Get action from policy
            with torch.no_grad():
                if hasattr(agent, 'get_action_and_value'):
                    if hasattr(agent, 'get_initial_lstm_state') and lstm_state is not None:
                        # Check if agent supports evaluation_mode parameter
                        import inspect
                        sig = inspect.signature(agent.get_action_and_value)
                        if 'evaluation_mode' in sig.parameters:
                            result = agent.get_action_and_value(masked_obs, lstm_state, next_done, evaluation_mode=True)
                        else:
                            result = agent.get_action_and_value(masked_obs, lstm_state, next_done)
                        
                        if len(result) == 5:
                            action, _, _, _, lstm_state = result
                        elif len(result) == 6:
                            action, _, _, _, lstm_state, _ = result
                        elif len(result) == 7:
                            action, _, _, _, lstm_state, _, _ = result
                        else:
                            action = result[0]
                    else:
                        # Check if agent supports evaluation_mode parameter
                        import inspect
                        sig = inspect.signature(agent.get_action_and_value)
                        if 'evaluation_mode' in sig.parameters:
                            result = agent.get_action_and_value(masked_obs, evaluation_mode=True)
                        else:
                            result = agent.get_action_and_value(masked_obs)
                        
                        if len(result) == 4:
                            action, _, _, _ = result
                        else:
                            action = result[0]
                else:
                    action = agent.get_action(masked_obs)

            # Step environment
            action_np = action.cpu().numpy()
            if hasattr(agent, 'is_discrete') and agent.is_discrete:
                action_np = action_np.reshape(eval_envs.num_envs, -1)
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            obs_dict = eval_envs.step(acts)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)

            # Track episode returns and lengths
            for env_idx in range(eval_envs.num_envs):
                env_returns[env_idx] += obs_dict['reward'][env_idx]
                env_lengths[env_idx] += 1
                if obs_dict['is_last'][env_idx]:
                    if len(env_episode_returns) < num_episodes:
                        env_episode_returns.append(env_returns[env_idx])
                        env_episode_lengths.append(env_lengths[env_idx])
                    env_returns[env_idx] = 0
                    env_lengths[env_idx] = 0

        # Compute metrics
        env_episode_returns = np.array(env_episode_returns)
        env_episode_lengths = np.array(env_episode_lengths)
        env_metrics[env_name] = {
            'mean_return': np.mean(env_episode_returns),
            'std_return': np.std(env_episode_returns),
            'mean_length': np.mean(env_episode_lengths),
            'std_length': np.std(env_episode_lengths)
        }
        print(f"  {env_name}: mean_return={env_metrics[env_name]['mean_return']:.2f}, std_return={env_metrics[env_name]['std_return']:.2f}")
        
        # Log metrics to wandb and tensorboard
        if use_wandb or writer is not None:
            log_metrics = {
                f"full_eval_return/{env_name}": env_metrics[env_name]['mean_return'],
                f"full_eval/{env_name}/std_return": env_metrics[env_name]['std_return'],
                f"full_eval/{env_name}/mean_length": env_metrics[env_name]['mean_length'],
                f"full_eval/{env_name}/std_length": env_metrics[env_name]['std_length'],
            }
            log_evaluation_metrics(log_metrics, global_step, use_wandb, writer)
        
        eval_envs.close()

    # Compute overall metrics across all environments
    all_returns = [metrics['mean_return'] for metrics in env_metrics.values()]
    all_lengths = [metrics['mean_length'] for metrics in env_metrics.values()]
    
    overall_metrics = {
        'full_eval_return/mean': np.mean(all_returns),
        'full_eval_return/std': np.std(all_returns),
        'full_eval/length/mean': np.mean(all_lengths),
        'full_eval/length/std': np.std(all_lengths),
    }
    
    # Log overall metrics to wandb and tensorboard
    if use_wandb or writer is not None:
        log_evaluation_metrics(overall_metrics, global_step, use_wandb, writer)
    
    print(f"  Overall: mean_return={overall_metrics['full_eval_return/mean']:.2f}, std_return={overall_metrics['full_eval_return/std']:.2f}")

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
                          make_envs_func=None, writer=None, use_wandb=False, debug=False, skip_subset_eval=False):
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
        skip_subset_eval: Whether to skip subset evaluation entirely
        
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
        
        # Run subset evaluation (observation subsets) only if not skipped
        if not skip_subset_eval:
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


def run_initial_evaluation(agent, config, device, make_envs_func, writer=None, use_wandb=False, debug=False, skip_subset_eval=False):
    """Run initial evaluation at the start of training.
    
    Args:
        agent: Agent to evaluate
        config: Configuration object
        device: Device to run evaluation on
        make_envs_func: Function to create evaluation environments
        writer: TensorBoard writer (optional)
        use_wandb: Whether to use wandb logging
        debug: Whether to print detailed debugging information
        skip_subset_eval: Whether to skip subset evaluation entirely
        
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
    
    # Run initial subset evaluation (observation subsets) only if not skipped
    if not skip_subset_eval:
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