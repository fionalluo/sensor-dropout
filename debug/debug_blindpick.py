from train_blindpick import SENSOR_CONFIGS, _make_env
import os
import numpy as np
from enum import Enum, auto

# Environment registration & constants
# -----------------------------------------------------------------------------

os.environ.setdefault("MUJOCO_GL", "egl")  # Off-screen rendering on headless nodes



difficulty = 0.07
seed = 0
sensor_config = "oracle_state"

config = SENSOR_CONFIGS[sensor_config]
print("config:", config)
env = _make_env(difficulty, config, seed)

print(env.observation_space)

"""
obs includes "obj_state" and 'robot_state'
obj_state is (3,) array of x,y,z position of the object
robot_state is (10,) array of gripper position, velocity, and state
    - pos (3,) array of gripper position
    - velp (3,) array of gripper velocity
    - gripper_state (2,) array of gripper state
    - gripper_vel (2,) array of gripper velocity
touch is (2,) array of boolean touch sensor data
"""

class Phase(Enum):
    REACH_ABOVE = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()

class OptimalPolicy:
    def __init__(self, env):
        self.env = env
        self.phase = Phase.REACH_ABOVE
        self.goal = env.unwrapped.goal.copy()

    def compute_action(self, obs):
        obj_pos = obs["obj_state"]
        grip_pos = obs["robot_state"][:3]
        
        action = np.zeros(self.env.action_space.shape)
        
        if self.phase == Phase.REACH_ABOVE:
            target_pos = obj_pos + np.array([0., 0., 0.1])
            action[:3] = (target_pos - grip_pos) * 10
            action[3] = 1.0  # Open gripper
            if np.linalg.norm(target_pos - grip_pos) < 0.02:
                self.phase = Phase.DESCEND
        
        elif self.phase == Phase.DESCEND:
            target_pos = obj_pos
            action[:3] = (target_pos - grip_pos) * 10
            action[3] = 1.0 # Keep gripper open
            if np.linalg.norm(target_pos - grip_pos) < 0.01:
                self.phase = Phase.GRASP
        
        elif self.phase == Phase.GRASP:
            action[3] = -1.0  # Close gripper
            if obs["touch"].all():
                self.phase = Phase.LIFT
                
        elif self.phase == Phase.LIFT:
            target_pos = self.goal
            action[:3] = (target_pos - grip_pos) * 10
            action[3] = -1.0 # Keep gripper closed

        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

policy = OptimalPolicy(env)
obs, _ = env.reset()
cum_reward = 0
for i in range(200):
    action = policy.compute_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    cum_reward += reward
    print(f"Step {i}: Phase: {policy.phase}, Reward: {cum_reward:.3f}, Terminated: {terminated}, Success: {info.get('is_success')}")

    if terminated:
        print(f"Success! at step {i}") ## should succeed rather early
        ## like step 18
        break
