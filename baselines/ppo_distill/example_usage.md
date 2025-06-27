# PPO Distill Example Usage

This document provides comprehensive examples of how to use the PPO Distill baseline for sensor dropout experiments.

## Prerequisites

First, ensure you have expert subset policies trained:

```bash
# Train subset policies
./train_subset_policies.sh

# Verify policies exist
ls ~/policies/tigerdoorkey/
# Should show: env1/ env2/ env3/ env4/ metadata.yaml
```

## Basic Training Examples

### Default Training

```bash
# Basic distillation training with default settings
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### High Distillation Coefficient

```bash
# Train with high distillation coefficient (more emphasis on expert guidance)
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey high_distill \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Batch Cycling Mode

```bash
# Train with batch-level configuration cycling
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey batch_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Network Architecture Examples

### Large Network

```bash
# Train with larger network architecture
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey large_network high_entropy \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Small Network

```bash
# Train with smaller network architecture
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey small_network low_entropy \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Large RNN

```bash
# Train with larger RNN hidden size
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey large_rnn episode_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Distillation Parameter Examples

### High Distillation

```bash
# High distillation coefficient (more expert guidance)
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey high_distill \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Low Distillation

```bash
# Low distillation coefficient (less expert guidance)
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey low_distill \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### No Distillation (PPO Only)

```bash
# Train without distillation (standard PPO)
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Cycling Mode Examples

### Episode Cycling (Default)

```bash
# Episode-level configuration cycling
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey episode_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Batch Cycling

```bash
# Batch-level configuration cycling
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey batch_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Training Parameter Examples

### Fast Learning

```bash
# Higher learning rate for faster training
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Slow Learning

```bash
# Lower learning rate for more stable training
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### High Entropy

```bash
# Higher entropy coefficient for more exploration
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Low Entropy

```bash
# Lower entropy coefficient for more exploitation
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Combined Configuration Examples

### Large Network with Large RNN

```bash
# Large network architecture with large RNN
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey large_network large_rnn \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Fast Learning with Batch Cycling

```bash
# Fast learning with batch-level cycling
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey fast_learning batch_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Slow Learning with High Entropy

```bash
# Slow learning with high entropy for exploration
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey slow_learning high_entropy \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Small Network with Low Distillation

```bash
# Small network with low distillation coefficient
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey small_network low_distill \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Debugging Examples

### Debug Mode

```bash
# Enable debug mode for detailed logging
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track \
    --debug
```

### No Wandb Logging

```bash
# Train without wandb logging
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey no_wandb \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda
```

## Advanced Examples

### Custom Seed

```bash
# Train with specific random seed
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey episode_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --seed 42 \
    --cuda \
    --track
```

### Multiple Configurations

```bash
# Combine multiple configuration options
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey high_distill episode_cycle \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Large Network with Large RNN and High Distillation

```bash
# Large architecture with high distillation
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey large_network large_rnn high_distill \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

## Monitoring Training

### Wandb Metrics

The training logs the following metrics to wandb:

- `train/distill_loss`: Distillation loss
- `train/current_config`: Current subset configuration
- `train/total_loss`: Combined loss (PPO + distillation)
- `eval/mean_return`: Average episode return
- `full_eval/env*/mean_return`: Performance on each subset

### Expected Behavior

- **Distillation loss** should decrease over time
- **Configuration cycling** should show different subsets being used
- **Performance** should improve across all observation subsets
- **Total loss** should be stable and not explode

## Troubleshooting

### Expert Policies Not Found

```
Error: Expert policy directory not found: ~/policies/tigerdoorkey
```

**Solution**: Train subset policies first:

```bash
./train_subset_policies.sh
```

### Memory Issues

If you encounter CUDA out of memory errors:

```bash
# Use smaller network
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey small_network \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track
```

### Debugging

Enable debug mode for detailed logging:

```bash
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/tigerdoorkey \
    --cuda \
    --track \
    --debug
```

## Expert Policy Directory Structure

```
policies/
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

Expert policy directory: ~/policies/tigerdoorkey 