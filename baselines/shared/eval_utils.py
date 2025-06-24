"""
Shared evaluation utilities for all baselines.
This module provides evaluation functions that can be used across different baseline implementations.
"""

import numpy as np
import torch
from thesis.shared.evaluation import evaluate_policy, log_evaluation_metrics, log_evaluation_videos


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
        eval_metrics = evaluate_agent(
            agent, eval_envs, device, config, 
            log_video=log_video, 
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
    
    # Run initial evaluation
    print("Running initial evaluation...")
    eval_metrics = evaluate_agent(
        agent, eval_envs, device, config, 
        log_video=True, 
        make_envs_func=make_envs_func,
        writer=writer,
        use_wandb=use_wandb,
        global_step=0
    )
    
    return eval_envs, eval_metrics 