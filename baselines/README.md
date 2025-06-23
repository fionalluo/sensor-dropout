# Sensor Dropout Baselines

This directory contains implementations of various baselines for the sensor dropout problem. The goal is to train agents that can perform well when deployed with limited sensor information, even when trained with full privileged information.

## Baseline Types

1. **PPO RNN** - Train on all privileged information, deploy on subsets of observations
2. **PPO** - Train on all privileged information, deploy on subsets of observations  
3. **PPO Distill** - Train multiple teachers on all subsets of privileged information, save those teacher agents, then distill them into one student
4. **PPO MoE** - Train multiple teachers on all subsets of privileged information, then train a softmax function to select which teacher to run based on RL rewards

## Directory Structure

```
baselines/
├── shared/                 # Shared components across all baselines
│   ├── agent.py           # Base agent class
│   ├── nets.py            # Neural network architectures
│   └── __init__.py
├── ppo/                   # Standard PPO baseline
│   ├── agent.py           # PPO agent implementation
│   ├── ppo.py             # PPO training algorithm
│   ├── train.py           # Training script
│   ├── config.yaml        # Configuration file
│   ├── test_ppo.py        # Test script
│   └── __init__.py
├── ppo_rnn/               # PPO with RNN (TODO)
├── ppo_distill/           # PPO with distillation (TODO)
├── ppo_moe/               # PPO with Mixture of Experts (TODO)
└── README.md              # This file
```

## Usage

### Running PPO Baseline

```bash
# Direct execution (recommended)
cd baselines/ppo
python train.py

# Training with custom parameters
python train.py --task robopianist_robopianist-v0 --num_envs 4 --num_iterations 500

# Training with specific observation subset
python train.py --configs student_only

# Training with teacher (full observations)
python train.py --configs teacher_only

# Training without wandb logging
python train.py --no_wandb

# Training with custom wandb project
python train.py --wandb_project my-experiment
```

### Configuration

The `config.yaml` file contains all hyperparameters and can be extended with different configurations:

- `defaults`: Default configuration
- `student_only`: Limited observations (for deployment)
- `teacher_only`: Full privileged information (for training)
- `image_only`: Only image observations
- `state_only`: Only state observations
- `no_wandb`: Disable wandb logging

### Logging

The framework supports both TensorBoard and Weights & Biases (wandb) logging:

#### Wandb Logging
- **Enabled by default** in the configuration
- Logs all training metrics, hyperparameters, and experiment configuration
- Can be disabled with `--no_wandb` flag
- Custom project name with `--wandb_project` flag

#### Logged Metrics
- **Training metrics**: Policy loss, value loss, entropy, KL divergence
- **Performance metrics**: Episodic return, episodic length, steps per second
- **Hyperparameters**: Learning rate, batch size, network architecture, etc.

#### Example wandb usage:
```bash
# Enable wandb (default)
python train.py

# Disable wandb
python train.py --no_wandb

# Custom project name
python train.py --wandb_project sensor-dropout-experiments
```

### Key Components

#### BaseAgent (`shared/agent.py`)
- Abstract base class for all agents
- Handles observation encoding and action selection
- Supports both discrete and continuous action spaces

#### PPOAgent (`ppo/agent.py`)
- Implements PPO-specific agent
- Can use either teacher keys (full observations) or student keys (limited observations)
- Handles both image and MLP encoders

#### PPOTrainer (`ppo/ppo.py`)
- Implements the PPO algorithm
- Handles rollout collection and policy updates
- Includes GAE advantage estimation and value function clipping
- Supports both TensorBoard and wandb logging

## Training Process

1. **Teacher Training**: Train agents with full privileged information
2. **Student Training**: Train agents with limited observations
3. **Evaluation**: Deploy trained agents on different observation subsets

## Observation Keys

The system uses regex patterns to filter observations:

- `cnn_keys`: Image observations (e.g., "image")
- `mlp_keys`: State observations (e.g., "state", "position", "velocity")

## Extending the Framework

To add a new baseline:

1. Create a new directory (e.g., `ppo_rnn/`)
2. Implement agent class inheriting from `BaseAgent`
3. Implement training algorithm
4. Create training script and configuration
5. Update this README

## Dependencies

- PyTorch
- NumPy
- embodied (for environments)
- ruamel.yaml (for configuration)
- tensorboard (for logging)
- wandb (for experiment tracking)

