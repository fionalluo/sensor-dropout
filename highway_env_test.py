import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import time

# # Test combined sensors using DictObservation
# print("\n=== Testing Combined Sensors (DictObservation) ===")

# env = gym.make("parkingDict-v0")
# obs, info = env.reset(seed=42)

# print(f"Combined observation space: {env.observation_space}")
# print(f"Number of sensors: {len(obs)}")
# for sensor_name, sensor_obs in obs.items():
#     print(f"Sensor {sensor_name} shape: {sensor_obs.shape}")

# # Visualize all sensors in a grid
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# sensor_names = list(obs.keys())

# for i, (ax, sensor_name) in enumerate(zip(axes.flat, sensor_names)):
#     # Show the latest frame from the stack
#     last_frame = obs[sensor_name][-1]
#     # last_frame = last_frame.transpose(1, 0)
#     ax.imshow(last_frame, cmap='gray')
#     ax.set_title(f"{sensor_name}")
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig("debug/combined_sensors_view.png")
# plt.close()

# # Take a step and show how observations change
# print("\n=== Testing Action Step ===")
# action = env.action_space.sample()
# obs_next, reward, terminated, truncated, info = env.step(action)

# print(f"Reward: {reward}")
# print(f"Terminated: {terminated}")
# print(f"Truncated: {truncated}")

# env.close()
# print("\nAll tests completed! Check the saved PNG files to visualize the sensor views.")


def test_env_speed(env_name, num_steps=1000, warmup_steps=100):
    """Test the speed of an environment by running steps and measuring FPS."""
    print(f"\n=== Testing {env_name} ===")
    
    # Create environment
    env = gym.make(env_name)
    obs, info = env.reset(seed=42)
    
    # Warmup phase
    print(f"Warming up with {warmup_steps} steps...")
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=42)
    
    # Performance test
    print(f"Running performance test with {num_steps} steps...")
    start_time = time.time()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=42)
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_steps / total_time
    
    env.close()
    
    print(f"Total time: {total_time:.3f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Average time per step: {total_time/num_steps*1000:.2f} ms")
    
    return fps, total_time

def main():
    print("=== Environment Speed Comparison ===")
    
    # Test both environments
    fps_normal, time_normal = test_env_speed("parking-v0")
    fps_dict, time_dict = test_env_speed("parkingDict-v0")
    
    # Comparison
    print("\n=== Performance Comparison ===")
    print(f"parking-v0:      {fps_normal:.2f} FPS ({time_normal:.3f}s)")
    print(f"parkingDict-v0:  {fps_dict:.2f} FPS ({time_dict:.3f}s)")
    
if __name__ == "__main__":
    main()



