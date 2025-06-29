# Subset Policy Training and Deployment

This directory contains utilities for training and deploying subset policies for sensor dropout experiments.

## Overview

For configurations like `gymnasium_tigerdoorkey`, this system trains 4 separate policies:

- **env1**: Policies trained with door and key observations only
- **env2**: Policies trained with tiger and key observations only  
- **env3**: Policies trained with tiger and door observations only
- **env4**: Policies trained with tiger, door, and key observations

**Supported Policy Types:**
- **PPO**: Standard PPO without recurrent networks
- **PPO-RNN**: PPO with LSTM recurrent networks for sequential inference

## Training Subset Policies

### Quick Start

```bash
# Train PPO-RNN policies (default)
./train_subset_policies.sh

# Train PPO policies
POLICY_TYPE=ppo ./train_subset_policies.sh
```

### Manual Training

```bash
# Train PPO-RNN subset policies manually
python3 subset_policies/train_subset_policies.py \
    --configs gymnasium_tigerdoorkey \
    --policy_type ppo_rnn \
    --output_dir ./policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --debug

# Train PPO subset policies manually
python3 subset_policies/train_subset_policies.py \
    --configs gymnasium_tigerdoorkey \
    --policy_type ppo \
    --output_dir ./policies/ppo/tigerdoorkey \
    --cuda \
    --debug
```

### Output Structure

```
policies/
├── ppo/
│   └── tigerdoorkey/
│       ├── env1/
│       │   └── policy_20241201_143022.pt
│       ├── env2/
│       │   └── policy_20241201_143156.pt
│       ├── env3/
│       │   └── policy_20241201_143245.pt
│       ├── env4/
│       │   └── policy_20241201_143334.pt
│       └── metadata.yaml
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
# List PPO policies
python subset_policies/load_subset_policy.py \
    --policy_dir ~/policies/ppo/tigerdoorkey \
    --list

# List PPO-RNN policies
python subset_policies/load_subset_policy.py \
    --policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --list
```

### Load Specific Policy

```bash
# Load PPO policy
python subset_policies/load_subset_policy.py \
    --policy_dir ~/policies/ppo/tigerdoorkey \
    --subset env1

# Load PPO-RNN policy
python subset_policies/load_subset_policy.py \
    --policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --subset env1
```

### Programmatic Usage

```python
from subset_policies.load_subset_policy import SubsetPolicyLoader

# Load a pre-trained PPO subset policy
ppo_loader = SubsetPolicyLoader("~/policies/ppo/tigerdoorkey", device='cpu')
agent, config, eval_keys = ppo_loader.load_policy('env1')

# Load a pre-trained PPO-RNN subset policy
ppo_rnn_loader = SubsetPolicyLoader("~/policies/ppo_rnn/tigerdoorkey", device='cpu')
agent, config, eval_keys = ppo_rnn_loader.load_policy('env1')

# Get actions (PPO doesn't use LSTM states)
action, _ = ppo_loader.get_action('env1', obs)

# Get actions (PPO-RNN uses LSTM states)
action, lstm_state = ppo_rnn_loader.get_action('env1', obs, lstm_state)
```

## Configuration

The training uses the same configuration system as the main PPO and PPO-RNN training, with the following key components:

### Policy Type Selection

- **PPO**: Uses `baselines/ppo/config.yaml` and `train_ppo` function
- **PPO-RNN**: Uses `baselines/ppo_rnn/config.yaml` and `train_ppo_rnn` function

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

- Uses the same PPO/PPO-RNN architecture and training loop as the main baselines
- Each subset is trained independently
- Policies are saved in a format optimized for quick loading
- LSTM states are preserved for PPO-RNN sequential inference

## Key Features

1. **Multiple Policy Types**: Support for both PPO and PPO-RNN policies
2. **Independent Training**: Each subset is trained separately with its own observation filtering
3. **Easy Deployment**: Policies can be loaded and used without retraining
4. **LSTM Support**: Full support for LSTM states in PPO-RNN inference
5. **Observation Filtering**: Automatic filtering based on eval_keys patterns
6. **Metadata Tracking**: Complete metadata about training configuration and results

## Shell Script Usage

The `train_subset_policies.sh` script supports both policy types:

```bash
# Train PPO-RNN policies (default)
./train_subset_policies.sh

# Train PPO policies
POLICY_TYPE=ppo ./train_subset_policies.sh

# Train PPO policies with custom environment variable
export POLICY_TYPE=ppo
./train_subset_policies.sh
```

## Troubleshooting

### Common Issues

1. **Policy directory not found**: Make sure to run training first or check the path
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Import errors**: Ensure all dependencies are installed and paths are correct
4. **Wrong policy type**: Make sure the policy type matches between training and loading

### Debug Mode

Use the `--debug` flag to get detailed information about:
- Observation filtering
- Policy loading
- Training progress

## Integration with Existing Code

The subset policies are designed to be drop-in replacements for the main PPO and PPO-RNN agents:

```python
# Instead of training a new agent
# agent = train_ppo(envs, config, seed, num_iterations)  # for PPO
# agent = train_ppo_rnn(envs, config, seed, num_iterations)  # for PPO-RNN

# Load a pre-trained subset policy
from subset_policies.load_subset_policy import SubsetPolicyLoader
loader = SubsetPolicyLoader(policy_dir, device='cpu')
agent, config, eval_keys = loader.load_policy('env1')

# Use the agent normally
if loader.policy_type == 'ppo':
    action, _ = loader.get_action('env1', obs)
else:  # ppo_rnn
    action, lstm_state = loader.get_action('env1', obs, lstm_state)
``` 