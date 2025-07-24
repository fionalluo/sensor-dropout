# Isaac Lab + Sensor Dropout: Quickstart (Docker)

This guide will get you up and running with Isaac Lab and the sensor-dropout repo using Docker, following the official Isaac Lab [Docker Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html).

---

## 1. Prerequisites

- **Follow the [Docker and Docker Compose section](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#docker-and-docker-compose) of the Isaac Lab Docker Guide.**
    - If all four bullet pointed items are already installed on your machine, you can proceed forward.

- **Next, try running the container as described in the Running the Container section of the [Isaac Lab Docker Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#running-the-container).**
    - If you can start and enter the container, you are ready to continue. 
    - Make sure that you have cloned the [IsaacLab](https://github.com/isaac-sim/IsaacLab) repo, the docker launcher is inside. You must clone the repo under any subfolder of your home directory.

---

## 2. Test Isaac Lab Installation

Inside the container, in the isaaclab directory, test that Isaac Lab works by running a standard RL-Games training script (headless mode for SSH):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Ant-v0 --headless
```
This command is from [Isaac Lab RL Existing Scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html), see the link for more details. If you are using SSH probably only the headless ones work, at least on Fiona's setup.

---

## 3. Mount sensor-dropout into the Isaac Lab Docker Container

This is not necessary for testing out any Isaac Lab examples, but eventually you want to mount our sensor-dropout directory into the Docker. You can come back and do this step later. 
Edit `IsaacLab/docker/docker-compose.yaml` and add a bind mount for your sensor-dropout repo.  
**Below these lines:**
```yaml
  - type: bind
    source: .isaac-lab-docker-history
    target: ${DOCKER_USER_HOME}/.bash_history
```
**Add:**
```yaml
  - type: bind
    source: /absolute/path/to/your/sensor-dropout
    target: /workspace/sensor-dropout
```
Replace `/absolute/path/to/your/sensor-dropout` with the actual path to your cloned sensor-dropout repo.

---

## 4. Sensor Dropout: Try the Scripts

Now let's test that the PPO baselines I wrote in sensor dropout work.

- I created a new folder: `baselines_isaac`. There are **two baselines implemented**:
  - `ppo`: Standard PPO baseline.
  - `ppo_dropout_any`: PPO with dropout applied on a per-episode basis for any of 2^n subsets of sensors.
- **Both have a training and a separate evaluation phase.**
  - The evaluation phase loads the epoch checkpoints and logs results to the same Weights & Biases (wandb) project.
  - **To view evaluation correctly in wandb, set the x-axis to `global_step` instead of `Step`.**
- The evaluation code is still WIP. It runs correctly, but is slow (several minutes per evaluation) because I didn't parallelize it yet. Currently evaluating not over specific pre-defined subsets, but over per-key dropout probabilities.

You can try the scripts:
```bash
./ppo_isaac.sh
./ppo_dropout_any_isaac.sh
```

**Before running them, open the `.sh` files and change the wandb user and project name to your own.**

- You may need to install additional Python packages for sensor-dropout. Just install them as you go though or use requirements.txt

---

## 4. Adding New Environments

Isaac Lab has many different environments. Since the observation is returned as one long tensor (not a dictionary with keys), we need to manually specify the key indices for masking in sensor-dropout.

**How to add a new environment:**

1. Run the `train.py` script in IsaacLab, from the tutorial, with the environment you want to use. This will print out the observation space, including the keys and their corresponding indices.
2. Copy the printed keys and indices into `sensor-dropout/baselines_isaac/config.yaml` under a new entry for your environment.
Now you can use your new environment with the sensor-dropout baselines.

---

## 6. Notes

These are things I realized when coding/debugging
- If you are running anything related to the simulation, you need to do it through the launcher `./isaaclab.sh`. All my scripts currently do this
- Only one environment instance can exist at a time, so you cannot easily create separate train/test environments. Instead, you want to make use of wrappers and resetting.