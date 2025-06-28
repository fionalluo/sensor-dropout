# PPO Distill Baseline

A PPO (Proximal Policy Optimization) implementation with distillation learning from expert subset policies for sensor dropout experiments.

## Overview

The PPO Distill baseline trains a student policy that learns from multiple expert subset policies through distillation. The student policy:

1. **Receives full observations** but with masked values based on current subset configuration
2. **Cycles through different observation subsets** during training (episode or batch mode)
3. **Learns from expert actions** via distillation loss combined with standard PPO objectives
4. **Maintains episode coherence** by cycling configurations appropriately

## Quick Start

### Prerequisites

First, train expert subset policies:

```bash
# Train subset policies
./train_subset_policies.sh

# Verify policies exist
ls ~/policies/ppo_rnn/tigerdoorkey/
# Should show: env1/ env2/ env3/ env4/ metadata.yaml
```

### Basic Training

```bash
# Basic distillation training
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --track
```

### Advanced Configurations

```bash
# High distillation coefficient
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey high_distill \
    --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --track

# Batch cycling mode
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey batch_cycle \
    --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --track

# Large network with high entropy
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey large_network high_entropy \
    --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --track
```

## Configuration Options

### Distillation Parameters

- `distill_coef`: Distillation loss coefficient (default: 0.1)
- `expert_coef`: Expert action coefficient (default: 0.5)
- `cycle_mode`: Configuration cycling mode ("episode" or "batch")

### Network Architecture

- `large_network`: Use larger MLP/CNN hidden sizes (512 instead of 256)
- `large_rnn`: Use larger LSTM hidden size (512 instead of 256)
- `small_network`: Use smaller hidden sizes (128 instead of 256)

### Training Parameters

- `high_entropy`: Higher entropy coefficient (0.05 instead of 0.01)
- `low_entropy`: Lower entropy coefficient (0.005 instead of 0.01)
- `fast_learning`: Higher learning rate (1e-3 instead of 3e-4)
- `slow_learning`: Lower learning rate (1e-4 instead of 3e-4)

## Architecture

### Student Policy

The student policy inherits from `PPORnnAgent` and adds:

1. **Expert Policy Manager**: Loads and manages expert subset policies
2. **Configuration Scheduler**: Cycles through different observation subsets
3. **Observation Masking**: Masks observations based on current subset
4. **Distillation Loss**: KL divergence between student and expert action distributions

### Training Process

1. **Configuration Cycling**: Student cycles through different observation subsets
2. **Observation Masking**: Full observations are masked based on current subset
3. **Expert Action Collection**: Expert policies provide target actions
4. **Distillation Loss**: Student learns to match expert action distributions
5. **Combined Training**: Distillation loss + standard PPO objectives

## Example Usage

### Basic Training

```bash
# Train with default settings
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey \
    --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
    --cuda \
    --track
```

### Custom Distillation Settings

```bash
# High distillation coefficient
python3 baselines/ppo_distill/train.py \
    --configs gymnasium_tigerdoorkey high_distill \
# Train with high distillation coefficient
python baselines/ppo_distill/train.py \
  --configs gymnasium_tigerdoorkey high_distill \
  --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
  --cuda

# Train with batch-level cycling
python baselines/ppo_distill/train.py \
  --configs gymnasium_tigerdoorkey batch_cycle \
  --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
  --cuda

# Train with custom network architecture
python baselines/ppo_distill/train.py \
  --configs gymnasium_tigerdoorkey large_network high_entropy \
  --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
  --cuda
```

## Configuration

### **Distillation Parameters**

```yaml
# Distillation settings
distill_coef: 0.1    # Coefficient for distillation loss
expert_coef: 0.5     # Coefficient for expert guidance
cycle_mode: "episode" # How to cycle: "episode" or "batch"
```

### **Predefined Configurations**

- **`high_distill`**: Higher distillation coefficient (0.5)
- **`low_distill`**: Lower distillation coefficient (0.05)
- **`batch_cycle`**: Batch-level configuration cycling
- **`episode_cycle`**: Episode-level configuration cycling (default)

### **Network Variations**

- **`large_network`**: Larger encoder (1024 dim, 512 units)
- **`small_network`**: Smaller encoder (256 dim, 128 units)
- **`large_rnn`**: Larger LSTM (256 hidden, 2 layers)
- **`small_rnn`**: Smaller LSTM (64 hidden, 1 layer)

## Training Process

### **1. Expert Policy Loading**
- Loads all subset policies (env1, env2, env3, env4)
- Validates policy compatibility
- Initializes expert action generation

### **2. Configuration Cycling**
- Cycles through observation configurations
- Maintains episode coherence
- Tracks current configuration state

### **3. Distillation Training**
- Collects rollouts with expert actions
- Computes distillation loss
- Combines with standard PPO objectives
- Updates student policy

### **4. Loss Components**
```
Total Loss = Policy Loss + 
             vf_coef * Value Loss + 
             ent_coef * Entropy Loss + 
             distill_coef * Distillation Loss
```

## Expected Benefits

### **Compared to Individual Subset Policies**
- **Single policy** instead of 4 separate policies
- **Unified representation** that generalizes across subsets
- **Better sample efficiency** through expert guidance
- **Easier deployment** with one model

### **Compared to Standard PPO**
- **Expert guidance** improves learning speed
- **Structured exploration** through configuration cycling
- **Better generalization** across observation subsets
- **More robust** to sensor dropout scenarios

## Troubleshooting

### **Common Issues**

1. **Expert policy directory not found**
   ```
   Error: Expert policy directory not found: ~/policies/ppo_rnn/tigerdoorkey
   ```
   **Solution**: Run `./train_subset_policies.sh` first

2. **CUDA out of memory**
   **Solution**: Reduce batch size or use CPU
   ```bash
   python baselines/ppo_distill/train.py \
     --configs gymnasium_tigerdoorkey \
     --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey
   ```

3. **High distillation loss**
   **Solution**: Reduce `distill_coef` or increase `expert_coef`

### **Debug Mode**

```bash
python baselines/ppo_distill/train.py \
  --configs gymnasium_tigerdoorkey \
  --expert_policy_dir ~/policies/ppo_rnn/tigerdoorkey \
  --debug
```

## Integration

The PPO Distill baseline integrates seamlessly with the existing evaluation framework:

- **Same configuration system** as PPO RNN
- **Compatible evaluation metrics**
- **Standard logging and tracking**
- **Easy comparison** with other baselines

## Files

- **`agent.py`**: PPO Distill agent with expert policy integration
- **`ppo_distill.py`**: Main training loop with distillation logic
- **`train.py`**: Training script with configuration management
- **`config.yaml`**: Configuration file with distillation parameters
- **`README.md`**: This documentation

## Example Output

```
Starting PPO Distill training...
Expert policy directory: ~/policies/ppo_rnn/tigerdoorkey
Cycle mode: episode
Distillation coefficient: 0.1
Expert coefficient: 0.5
Loaded expert policy: env1
Loaded expert policy: env2
Loaded expert policy: env3
Loaded expert policy: env4
Loaded 4 expert policies

Iteration 0
  FPS: 1250.3
  Total Loss: 2.3456
  Policy Loss: 1.2345
  Value Loss: 0.5678
  Entropy Loss: 0.1234
  Distill Loss: 0.4199
  Approx KL: 0.0123
  Clip Frac: 0.2345
  Current Config: env1
```

This baseline provides a powerful approach to learning unified policies that can handle sensor dropout scenarios through expert-guided distillation. 