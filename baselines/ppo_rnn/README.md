# PPO RNN Implementation

This directory contains a PPO (Proximal Policy Optimization) implementation with RNN (Recurrent Neural Network) support for handling sequential observations. The implementation is based on the existing PPO baseline but adds LSTM layers to process temporal information.

## Features

- **PPO Algorithm**: Standard PPO implementation with GAE (Generalized Advantage Estimation)
- **RNN Support**: LSTM layers for processing sequential observations
- **Flexible Encoders**: Support for both CNN and MLP encoders
- **Multi-Environment Support**: Works with tiger key door, maze, and blind pick environments
- **Evaluation**: Comprehensive evaluation with observation subset testing
- **Logging**: TensorBoard and wandb integration

## Architecture

The PPO RNN agent consists of:

1. **Observation Encoders**: 
   - CNN encoders for image observations (heavyweight for large images, lightweight for small images)
   - MLP encoders for non-image observations
2. **LSTM Layer**: Processes encoded observations sequentially
3. **Actor-Critic Networks**: Policy and value networks operating on LSTM outputs

## Usage

### Training

```bash
# Basic training
python train.py --configs gymnasium_tigerkeydoor

# With custom seed
python train.py --configs gymnasium_tigerkeydoor --seed 42

# With wandb tracking
python train.py --configs gymnasium_tigerkeydoor --track

# Debug mode
python train.py --configs gymnasium_tigerkeydoor --debug
```

### Configuration

The training uses a YAML configuration file (`config.yaml`) with named configs for different environments:

- `gymnasium_tigerkeydoor`: Tiger Door Key environment
- `gymnasium_maze`: Maze environment  
- `gymnasium_blindpick`: Blind Pick environment

### Key Parameters

- `rnn.hidden_size`: LSTM hidden state size (default: 128)
- `rnn.num_layers`: Number of LSTM layers (default: 1)
- `learning_rate`: Learning rate for optimization
- `num_steps`: Number of steps per rollout
- `num_envs`: Number of parallel environments
- `total_timesteps`: Total training timesteps

## Implementation Details

### LSTM Integration

The LSTM layer is integrated into the PPO training loop:

1. **State Management**: LSTM states are maintained across rollout steps
2. **Reset Handling**: LSTM states are reset when episodes end
3. **Mini-batch Processing**: LSTM states are properly handled during policy updates

### Observation Processing

The agent supports multiple observation types:

- **Large Images** (>7x7): Processed by heavyweight ResNet encoder
- **Small Images** (≤7x7): Processed by lightweight CNN encoder  
- **Non-image Data**: Processed by MLP encoder

### Training Loop

1. **Rollout Collection**: Collect experience with LSTM state tracking
2. **Advantage Computation**: Compute advantages using GAE
3. **Policy Updates**: Update policy with LSTM state management
4. **Evaluation**: Periodic evaluation with observation subset testing

## Dependencies

- PyTorch
- NumPy
- Gymnasium
- Embodied
- TensorBoard
- Wandb (optional)

## Files

- `train.py`: Main training script
- `agent.py`: PPO RNN agent implementation
- `ppo_rnn.py`: PPO RNN trainer class
- `config.yaml`: Configuration file with named configs
- `ppo_rnn.sh`: Shell script for running experiments
- `test_agent.py`: Test script for verifying agent functionality

## Example Output

```
Configuration:
--------------------------------------------------
task: gymnasium_TigerDoorKey-v0
total_timesteps: 500000
learning_rate: 0.0001
num_envs: 8
num_steps: 256
rnn:
  hidden_size: 128
  num_layers: 1
--------------------------------------------------

Starting PPO RNN training on gymnasium_TigerDoorKey-v0 with 8 environments
Training for 244 iterations
Wandb logging enabled - project: sensor-dropout

Episode 1: reward=0.00, length=100
Episode 2: reward=0.00, length=100
...
```

## Notes

- The implementation reuses much of the existing PPO codebase
- LSTM states are properly managed during training and evaluation
- The agent supports both discrete and continuous action spaces
- Evaluation includes observation subset testing for sensor dropout analysis 