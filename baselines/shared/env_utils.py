import gymnasium as gym
import numpy as np
import time
from typing import Dict, Any
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id: str, idx: int, capture_video: bool = False, run_name: str = None):
    """
    Create a single environment with wrappers.
    
    Args:
        env_id: Environment ID
        idx: Environment index
        capture_video: Whether to capture video
        run_name: Run name for video recording
        
    Returns:
        Environment creation function
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        return env
    return thunk

def make_envs(config: Dict[str, Any]):
    """
    Create vectorized environments based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Vectorized environment
    """
    env_id = config['env_id']
    num_envs = config['num_envs']
    capture_video = config.get('capture_video', False)
    run_name = f"{env_id}__ppo_rnn__{config['seed']}__{int(time.time())}"
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    
    return envs

def get_env_info(envs):
    """
    Get environment information.
    
    Args:
        envs: Vectorized environment
        
    Returns:
        Dictionary with environment information
    """
    return {
        'single_observation_space': envs.single_observation_space,
        'single_action_space': envs.single_action_space,
        'obs_space': envs.observation_space,
        'act_space': envs.action_space,
    } 