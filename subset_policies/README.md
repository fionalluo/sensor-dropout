# Subset Policy Training and Deployment

This directory contains utilities for training and deploying subset policies for sensor dropout experiments.

## Overview

For configurations like `gymnasium_tigerdoorkey`, this system trains 4 separate policies:

- **env1**: Policies trained with door and key observations only
- **env2**: Policies trained with tiger and key observations only  
- **env3**: Policies trained with tiger and door observations only
- **env4**: Policies trained with tiger, door, and key observations

## Training Subset Policies

### Quick Start

```bash
# Train policies for gymnasium_tigerdoorkey (from project root)
./train_subset_policies.sh
```

### Manual Training

```bash
# Train subset policies manually
python3 subset_policies/train_subset_policies.py \
    --configs gymnasium_tigerdoorkey \
    --output_dir ./policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --debug
```

### Output Structure

```
policies/
└── ppo_rnn/
    └── tigerdoorkey/
        ├── env1/
        │   └── policy_20241201_143022.pt
        ├── env2/
        │   └── policy_20241201_143156.pt
        ├── env3/
        │   └── policy_20241201_143245.pt
        ├── env4/
        │   └── policy_20241201_143334.pt
        └── metadata.yaml
```

## Using Trained Policies

### List Available Policies

```bash
python subset_policies/load_subset_policy.py \
    --policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --list
```

### Load Specific Policy

```bash
python subset_policies/load_subset_policy.py \
    --policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --subset env1
```

### Programmatic Usage

```python
from subset_policies.load_subset_policy import SubsetPolicyLoader

# Load a pre-trained subset policy
loader = SubsetPolicyLoader("~/policies/ppo_rnn/tigerdoorkey", device='cpu')
agent, config, eval_keys = loader.load_policy('env1')
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