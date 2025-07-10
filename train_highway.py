
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import HerReplayBuffer, SAC
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio.v2 as imageio
from pathlib import Path


# Individual sensor configurations





def record_rollout(
    model_path: str = "models/model.zip",
    env_id: str = "highway_parking-v0",
    output_path: str = "rollout.mp4",
    n_steps: int = 1000,
    deterministic: bool = True,
    sensor_name = "overhead"
) -> None:
    """Load a trained SB3 model, run a rollout and save it as a video.

    Parameters
    ----------
    model_path : str
        Path to the ``.zip`` file produced by ``model.save``.
    env_id : str
        Gymnasium environment ID to create.
    output_path : str
        Target ``.mp4`` or ``.gif`` filename.
    n_steps : int
        Maximum number of environment steps to record.
    deterministic : bool
        Use the deterministic policy when predicting actions.
    """

    # Create env in RGB‐array render mode so env.render() returns numpy frames
    env = gym.make(env_id, render_mode="rgb_array",)

    render_env = gym.make("parkingDict-v0", render_mode="rgb_array",)
    # Load model – we pass env so SB3 can access observation / action spaces
    model = SAC.load(Path(model_path), env=env)


    obs, info = env.reset(seed=42)
    render_obs, info = render_env.reset(seed=42)
    frames = []

    for _ in range(n_steps):
        # Capture frame *before* the step so we also see the initial state
        # frame = env.render()
        # frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        render_obs, reward, terminated, truncated, info = render_env.step(action)

        last_frame = render_obs[sensor_name][-1]
        ## switch x and y
        # last_frame = last_frame.transpose(1, 0)
        frames.append(last_frame)

        if terminated or truncated:
            break

    env.close()

    if len(frames) == 0:
        raise RuntimeError("No frames captured – check render_mode or env compatibility.")

    fps = env.metadata.get("render_fps", 30)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gif":
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        # Default to MP4 (libx264)
        writer = imageio.get_writer(output_path, format="ffmpeg", fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(f"Saved rollout video to {output_path}")


#@title Training

# LEARNING_STEPS = 5e4 # @param {type: "number"}

# env = gym.make('highway_parking-v0')

# # Initialise Weights & Biases run
# wandb.init(project="highway_parking_sac", config={
#     "algo": "SAC",
#     "env": "parking-v0",
#     "learning_steps": LEARNING_STEPS,
#     "buffer_size": 1e6,
#     "learning_rate": 1e-3,
#     "gamma": 0.95,
#     "batch_size": 1024,
#     "tau": 0.05,
#     "net_arch": [512, 512, 512]
# }, sync_tensorboard=True, monitor_gym=True, save_code=True)

# her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')
# model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
#             replay_buffer_kwargs=her_kwargs, verbose=1, 
#             tensorboard_log="logs", 
#             buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.95, batch_size=1024, tau=0.05,
#             learning_starts=1000,
#             policy_kwargs=dict(net_arch=[512, 512, 512]))

# # Add WandB callback to log training metrics
# wandb_callback = WandbCallback(model_save_path="models")

# model.learn(int(LEARNING_STEPS), callback=wandb_callback)



for sensor_name in ["overhead", "left_mirror", "right_mirror", "rear_camera"]:
    record_rollout(
    "models/model.zip",
    "parking-v0",
    f"debug/rollout_{sensor_name}.mp4",
        n_steps=100,
        deterministic=True,
            sensor_name=sensor_name
    )