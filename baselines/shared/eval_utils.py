"""
Shared evaluation utilities for all baselines.
This module provides evaluation functions that can be used across different baseline implementations.
"""

import numpy as np
import torch
import re
from thesis.shared.evaluation import evaluate_policy, log_evaluation_metrics, log_evaluation_videos


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
    
    # Process MLP keys
    for key, value in obs_dict.items():
        if matches_pattern(key, mlp_keys_pattern):
            filtered_obs[key] = value
        elif key.endswith('_unprivileged'):
            # Check if the privileged version is in the pattern
            privileged_key = key[:-13]  # Remove '_unprivileged' suffix
            if matches_pattern(privileged_key, mlp_keys_pattern):
                # Substitute unprivileged for privileged
                filtered_obs[privileged_key] = value
    
    # Process CNN keys (same logic)
    for key, value in obs_dict.items():
        if matches_pattern(key, cnn_keys_pattern):
            filtered_obs[key] = value
        elif key.endswith('_unprivileged'):
            privileged_key = key[:-13]
            if matches_pattern(privileged_key, cnn_keys_pattern):
                filtered_obs[privileged_key] = value
    
    return filtered_obs


def evaluate_agent_with_observation_subsets(agent, envs, device, config, make_envs_func=None, 
                                          writer=None, use_wandb=False, global_step=0):
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
        
    Returns:
        dict: Combined evaluation metrics
    """
    if not hasattr(config, 'eval') or not hasattr(config.eval, 'num_eval_configs'):
        print("Warning: No eval.num_eval_configs found in config, skipping subset evaluation")
        return {}
    
    num_eval_configs = config.eval.num_eval_configs
    all_episode_returns = []
    all_episode_lengths = []
    env_metrics = {}
    
    print(f"Running evaluation across {num_eval_configs} observation subsets...")
    
    for env_idx in range(1, num_eval_configs + 1):
        env_name = f"env{env_idx}"
        
        # Get eval keys for this environment
        if hasattr(config, 'eval_keys') and hasattr(config.eval_keys, env_name):
            eval_keys = getattr(config.eval_keys, env_name)
            mlp_keys_pattern = eval_keys.mlp_keys
            cnn_keys_pattern = eval_keys.cnn_keys
        else:
            print(f"Warning: No eval_keys.{env_name} found, using default patterns")
            mlp_keys_pattern = '.*'
            cnn_keys_pattern = '.*'
        
        # Create evaluation environments
        if make_envs_func is None:
            raise ValueError("make_envs_func must be provided")
        eval_envs = make_envs_func(config, num_envs=config.eval.eval_envs)
        eval_envs.num_envs = config.eval.eval_envs
        
        # Run evaluation with observation filtering
        env_episode_returns = []
        env_episode_lengths = []
        num_episodes = config.eval.num_eval_episodes
        
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
        
        # Filter and substitute observations
        filtered_obs_dict = filter_observations_by_keys(obs_dict, mlp_keys_pattern, cnn_keys_pattern)
        
        for key in all_keys:
            if key in filtered_obs_dict:
                next_obs[key] = torch.Tensor(filtered_obs_dict[key].astype(np.float32)).to(device)
            else:
                # Zero out missing observations
                if key in obs:
                    next_obs[key] = torch.zeros_like(obs[key])
        
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
        
        # Track episode returns and lengths
        env_returns = np.zeros(eval_envs.num_envs)
        env_lengths = np.zeros(eval_envs.num_envs)
        
        # Run evaluation until we have enough episodes
        while len(env_episode_returns) < num_episodes:
            # Get action from policy
            with torch.no_grad():
                if hasattr(agent, 'get_action_and_value'):
                    action, _, _, _ = agent.get_action_and_value(next_obs)
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
            
            # Filter and substitute observations
            filtered_obs_dict = filter_observations_by_keys(obs_dict, mlp_keys_pattern, cnn_keys_pattern)
            
            # Process observations
            for key in all_keys:
                if key in filtered_obs_dict:
                    next_obs[key] = torch.Tensor(filtered_obs_dict[key].astype(np.float32)).to(device)
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
                        all_episode_returns.append(env_returns[env_idx])
                        all_episode_lengths.append(env_lengths[env_idx])
                    
                    env_returns[env_idx] = 0
                    env_lengths[env_idx] = 0
        
        # Calculate metrics for this environment
        env_episode_returns = np.array(env_episode_returns)
        env_episode_lengths = np.array(env_episode_lengths)
        
        env_metrics[env_name] = {
            'mean_return': np.mean(env_episode_returns),
            'std_return': np.std(env_episode_returns),
            'mean_length': np.mean(env_episode_lengths),
            'std_length': np.std(env_episode_lengths)
        }
        
        print(f"  {env_name}: mean_return={env_metrics[env_name]['mean_return']:.2f}, "
              f"std_return={env_metrics[env_name]['std_return']:.2f}")
    
    # Calculate overall metrics (averaged across all environments)
    all_episode_returns = np.array(all_episode_returns)
    all_episode_lengths = np.array(all_episode_lengths)
    
    overall_metrics = {
        'mean_return': np.mean(all_episode_returns),
        'std_return': np.std(all_episode_returns),
        'mean_length': np.mean(all_episode_lengths),
        'std_length': np.std(all_episode_lengths)
    }
    
    # Log individual environment metrics
    for env_name, metrics in env_metrics.items():
        log_evaluation_metrics({
            f"full_eval/{env_name}/mean_return": metrics['mean_return'],
            f"full_eval/{env_name}/std_return": metrics['std_return'],
            f"full_eval/{env_name}/mean_length": metrics['mean_length'],
            f"full_eval/{env_name}/std_length": metrics['std_length'],
        }, global_step, use_wandb, writer)
    
    # Log overall metrics
    log_evaluation_metrics({
        "full_eval/overall/mean_return": overall_metrics['mean_return'],
        "full_eval/overall/std_return": overall_metrics['std_return'],
        "full_eval/overall/mean_length": overall_metrics['mean_length'],
        "full_eval/overall/std_length": overall_metrics['std_length'],
    }, global_step, use_wandb, writer)
    
    return overall_metrics


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
    if log_video and 'video_frames' in eval_metrics:
        log_evaluation_videos(eval_metrics['video_frames'], global_step, use_wandb)
    
    return eval_metrics


def run_periodic_evaluation(agent, config, device, global_step, last_eval, eval_envs, 
                          make_envs_func=None, writer=None, use_wandb=False):
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
            global_step=global_step
        )
        
        last_eval = global_step
    
    return last_eval, eval_envs


def run_initial_evaluation(agent, config, device, make_envs_func, writer=None, use_wandb=False):
    """Run initial evaluation at the start of training.
    
    Args:
        agent: Agent to evaluate
        config: Configuration object
        device: Device to run evaluation on
        make_envs_func: Function to create evaluation environments
        writer: TensorBoard writer (optional)
        use_wandb: Whether to use wandb logging
        
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
        global_step=0
    )
    
    return eval_envs, eval_metrics 