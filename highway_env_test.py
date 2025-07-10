import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np

# Test combined sensors using DictObservation
print("\n=== Testing Combined Sensors (DictObservation) ===")

env = gym.make("parkingDict-v0")
obs, info = env.reset(seed=42)

print(f"Combined observation space: {env.observation_space}")
print(f"Number of sensors: {len(obs)}")
for sensor_name, sensor_obs in obs.items():
    print(f"Sensor {sensor_name} shape: {sensor_obs.shape}")

# Visualize all sensors in a grid
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
sensor_names = list(obs.keys())

for i, (ax, sensor_name) in enumerate(zip(axes.flat, sensor_names)):
    # Show the latest frame from the stack
    last_frame = obs[sensor_name][-1]
    # last_frame = last_frame.transpose(1, 0)
    ax.imshow(last_frame, cmap='gray')
    ax.set_title(f"{sensor_name}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("debug/combined_sensors_view.png")
plt.close()

# Take a step and show how observations change
print("\n=== Testing Action Step ===")
action = env.action_space.sample()
obs_next, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

env.close()
print("\nAll tests completed! Check the saved PNG files to visualize the sensor views.")
