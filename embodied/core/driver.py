import collections
from typing import Any

import numpy as np

from .basics import convert
from dataclasses import dataclass


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  @dataclass
  class StepAnneal:
    callback: Any
    start_freq: int
    end_freq: int
    start: float
    end: float
    steps: int

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_steps_anneal = []
    self._on_steps_conditional = []
    self._on_episodes = []
    self._on_calls = []
    self._steps = 0
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None
    self._call_stats = collections.defaultdict(list)  # List of scores to be populated for every episode

  def on_step(self, callback):
    self._on_steps.append(callback)
  
  def on_step_anneal(self, callback, start_freq, end_freq, start, end, steps):
    self._on_steps_anneal.append(self.StepAnneal(callback, start_freq, end_freq, start, end, steps))
  
  def on_step_conditional(self, callback):
    self._on_steps_conditional.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)
  
  def on_call(self, callback):
    self._on_calls.append(callback)

  def __call__(self, policy, steps=0, episodes=0, custom_logging=False, mode=""):
    """Advances all environments until steps and episodes are both met.

    Args:
        policy (Any): function to convert observations -> actions
        steps (int, optional): Number of steps to take. Defaults to 0.
        episodes (int, optional): Number of episodes to run. Defaults to 0.
        custom_logging (bool, optional): Whether to use custom logging (e.g. mean score, etc). Defaults to False.
    """
    # print("BUFFER: Call mode", mode)
    step, episode = 0, 0
    while step < steps or episode < episodes:
      # if mode != "": print("BUFFER: Step", step, "Episode", episode)
      step, episode = self._step(policy, step, episode, mode=mode)
    
    # Call on_episode callbacks
    [fn(self._call_stats, **self._kwargs) for fn in self._on_calls]
    self._call_stats = collections.defaultdict(list)  # Reset call stats

  def _step(self, policy, step, episode, mode=""):
    """Step all environments forward by one step.

    Args:
        policy (Any): policy that takes observations and returns actions
        step (int): step counter
        episode (int): episode counter

    Returns:
        Tuple[int, int]: step, episode updated counts
    """
    assert all(len(x) == len(self._env) for x in self._acts.values())  # Check action shapes
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}  # Filter out keys starting with 'log_'
    obs = self._env.step(acts)  # Perform actions in environment --> get new observations
    obs = {k: convert(v) for k, v in obs.items()}  # Convert to numpy

    assert all(len(x) == len(self._env) for x in obs.values()), obs  # Check shapes
    acts, self._state = policy(obs, self._state, **self._kwargs)  # Get next actions from the policy
    acts = {k: convert(v) for k, v in acts.items()}  # Convert to numpy

    # Handle terminal (done) environments
    if obs['is_last'].any():  # If any of the environments are done
      mask = 1 - obs['is_last']  # Create a mask for the environments that are not done
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}  # Mask the actions
    acts['reset'] = obs['is_last'].copy()  # Set reset action for done environments
    self._acts = acts  # Store the actions for next time
    trns = {**obs, **acts}  # Combine the observations and actions (transition dict)

    # Handle newly initialized environments
    if obs['is_first'].any():  # If any of the environments are new
      for i, first in enumerate(obs['is_first']):  # For each environment
        if first:  # If the environment is new
          self._eps[i].clear()  # Clear the episode dictionary

    # Append transition data for each environment
    for i in range(len(self._env)):  # For each environment
      trn = {k: v[i] for k, v in trns.items()}  # extract transition data
      [self._eps[i][k].append(v) for k, v in trn.items()]  # append transition data to episode dictionary
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]  # call on_step callbacks
      if mode == "skip_on_step_conditional":
        # print("BUFFER: Not adding teacher data to train replay buffer")
        pass
      else:
        # print("BUFFER: Mode is not teacher full_train_no_replay, run on_steps_conditional")
        [fn(trn, i, **self._kwargs) for fn in self._on_steps_conditional]  # call conditional on_step callbacks if not in full_train mode

      # call on_step_anneal callbacks
      for anneal in self._on_steps_anneal:
        if self._steps < anneal.start * anneal.steps:
          prob = anneal.start_freq
        elif self._steps < anneal.end * anneal.steps:
          # Linearly interpolate probability between start_freq and end_freq
          progress = (self._steps - anneal.start * anneal.steps) / ((anneal.end - anneal.start) * anneal.steps)
          prob = anneal.start_freq + progress * (anneal.end_freq - anneal.start_freq)
        else:
          prob = anneal.end_freq

        if np.random.rand() < prob:  # Call the function with probability `prob`
          anneal.callback(trn, i, **self._kwargs)
        
        print("PROB", prob)
      step += 1  # increment step count

    # Handle environments that are done
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):  # For every environment index
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}  # Convert episode data to numpy
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]  # Call on_episode callbacks
          episode += 1  # Increment episode count
          
          # CUSTOM LOGGING: store the score as a class variable
          score = float(ep['reward'].astype(np.float64).sum())  # score = sum of rewards
          self._call_stats['score'].append(score)  # Append score to list of scores

          # CUSTOM LOGGING: store the last reward of this episode
          last_reward = float(ep['reward'][-1])
          self._call_stats['last_reward'].append(last_reward)

          # CUSTOM LOGGING: store the heatmap of this episode
          if 'heatmap' in obs:
            self._call_stats['heatmap'].append(obs['heatmap'])
    
    # Increment steps
    self._steps += 1
    
    # Return the updated step and episode counts
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
