# Subset Policy Training and Deployment

This directory contains scripts to train PPO RNN policies for each subset of eval_keys and easily deploy them for inference.

## Overview

For configurations like `gymnasium_tigerkeydoor`, this system trains 4 separate policies:
- `env1`: Full privileged observations
- `env2`: Unprivileged door observations  
- `env3`: Unprivileged key observations
- `env4`: All unprivileged observations

Each policy is trained on its specific subset of observations and saved in a structured format for easy deployment.

## Files

- `train_subset_policies.py`: Main training script
- `load_subset_policy.py`: Utility to load and use trained policies
- `example_usage.py`: Example showing how to use the trained policies
- `README.md`: This documentation

## Training Policies

### Quick Start

```bash
# Train policies for gymnasium_tigerkeydoor (from project root)
./train_subset_policies.sh
```

### Manual Training

```bash
# Train for a specific config (from project root)
python subset_policies/train_subset_policies.py \
  --configs gymnasium_tigerkeydoor \
  --seed 12345 \
  --output_dir ~/policies \
  --cuda \
  --debug
```

### Output Structure

Policies are saved in the following structure:
```
policies/
└── tigerkeydoor_12345/
    ├── env1/
    │   └── policy.pt
    ├── env2/
    │   └── policy.pt
    ├── env3/
    │   └── policy.pt
    ├── env4/
    │   └── policy.pt
    └── metadata.yaml
```

Each `policy.pt` contains:
- `agent_state_dict`: Trained model weights
- `config`: Configuration used for training
- `eval_keys`: Observation filtering patterns
- `subset_name`: Name of the subset (env1, env2, etc.)

## Loading and Using Policies

### List Available Policies

```bash
python subset_policies/load_subset_policy.py \
  --policy_dir ~/policies/tigerkeydoor_12345 \
  --list
```

### Load Specific Policy

```bash
python subset_policies/load_subset_policy.py \
  --policy_dir ~/policies/tigerkeydoor_12345 \
  --subset env1
```

### Programmatic Usage

```python
from subset_policies.load_subset_policy import SubsetPolicyLoader

# Load all policies
loader = SubsetPolicyLoader("~/policies/tigerkeydoor_12345", device='cpu')

# Get action from a specific policy
action, lstm_state = loader.get_action('env1', observation_dict, lstm_state)

# Load specific policy for direct access
agent, config, eval_keys = loader.load_policy('env1')
```

### Example Usage

```bash
# Run the example script (update policy_dir first)
python subset_policies/example_usage.py
```

## Configuration

The training uses the same configuration system as the main PPO RNN training, with the following key components:

### eval_keys Configuration

```yaml
eval_keys:
  env1:
    mlp_keys: '\b(neighbors|door|doors_unlocked|position|has_key)\b'
    cnn_keys: '^$'
  env2:
    mlp_keys: '\b(neighbors_unprivileged|door_unprivileged|doors_unlocked|position|has_key)\b'
    cnn_keys: '^$'
  # ... more environments
```

### Training Parameters

- Uses the same PPO RNN architecture and training loop
- Each subset is trained independently
- Policies are saved in a format optimized for quick loading
- LSTM states are preserved for sequential inference

## Key Features

1. **Independent Training**: Each subset is trained separately with its own observation filtering
2. **Easy Deployment**: Policies can be loaded and used without retraining
3. **LSTM Support**: Full support for LSTM states in inference
4. **Observation Filtering**: Automatic filtering based on eval_keys patterns
5. **Metadata Tracking**: Complete metadata about training configuration and results

## Troubleshooting

### Common Issues

1. **Policy directory not found**: Make sure to run training first or check the path
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Import errors**: Ensure all dependencies are installed and paths are correct

### Debug Mode

Use the `--debug` flag to get detailed information about:
- Observation filtering
- Policy loading
- Training progress

## Integration with Existing Code

The subset policies are designed to be drop-in replacements for the main PPO RNN agent:

```python
# Instead of training a new agent
# agent = train_ppo_rnn(envs, config, seed, num_iterations)

# Load a pre-trained subset policy
from subset_policies.load_subset_policy import SubsetPolicyLoader
loader = SubsetPolicyLoader(policy_dir, device='cpu')
agent, config, eval_keys = loader.load_policy('env1')

# Use the agent normally
action, lstm_state = agent.get_action(obs, lstm_state)
```

This makes it easy to compare performance across different observation subsets or deploy specific policies in production environments. 