# PPO RNN Baseline - Example Usage

This document provides examples of how to use the PPO RNN baseline for training and evaluation.

## Basic Training

### Standard Training (Full Observations)

```bash
# Train on full observations (default)
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor

# Train on full observations with specific seed
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor --seed 42

# Train without wandb logging
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor no_wandb
```

### Training on Specific Observation Subsets

```bash
# Train on env1 (full privileged observations)
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor --training_env env1

# Train on env2 (must press button to reveal tiger/treasure)
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor --training_env env2

# Train on env3 (must get key to reveal unlocked doors)
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor --training_env env3

# Train on env4 (must press button and get key)
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor --training_env env4
```

## Maze Environment

```bash
# Train on env1 (full privileged)
python baselines/ppo_rnn/train.py --configs gymnasium_maze --training_env env1

# Train on env2 (only neighbors_3x3)
python baselines/ppo_rnn/train.py --configs gymnasium_maze --training_env env2

# Train on env3 (only neighbors_5x5)
python baselines/ppo_rnn/train.py --configs gymnasium_maze --training_env env3

# Train on env4 (neighbors_3x3 and goal position)
python baselines/ppo_rnn/train.py --configs gymnasium_maze --training_env env4

# Train on env5 (neighbors_5x5 and goal position)
python baselines/ppo_rnn/train.py --configs gymnasium_maze --training_env env5

# Train on env6 (both neighbors, no goal)
python baselines/ppo_rnn/train.py --configs gymnasium_maze --training_env env6
```

## Blind Pick Environment

```bash
# Train on env1 (full privileged)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env1

# Train on env2 (robot state, NO touch, all cameras)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env2

# Train on env3 (NO robot state, touch, all cameras)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env3

# Train on env4 (NO robot state, NO touch, only cameras)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env4

# Train on env5 (robot state, touch, only wrist camera)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env5

# Train on env6 (no robot state, touch, only front camera)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env6

# Train on env7 (robot state, touch, only side camera)
python baselines/ppo_rnn/train.py --configs gymnasium_blindpick --training_env env7
```

## Observation Subsets

### Tiger Door Key Environment

- **env1**: Full privileged observations
  - MLP: `neighbors`, `door`, `doors_unlocked`, `position`, `has_key`
  - CNN: `^$` (empty)

- **env2**: Must press button to reveal tiger/treasure
  - MLP: `neighbors_unprivileged_key`, `door_unprivileged`, `doors_unlocked`, `position`, `has_key`
  - CNN: `^$` (empty)
  - **Substitution**: `neighbors` → `neighbors_unprivileged_key`, `door` → `door_unprivileged`

- **env3**: Must get key to reveal unlocked doors
  - MLP: `neighbors_unprivileged_button`, `door`, `doors_unlocked_unprivileged`, `position`, `has_key`
  - CNN: `^$` (empty)
  - **Substitution**: `neighbors` → `neighbors_unprivileged_button`, `doors_unlocked` → `doors_unlocked_unprivileged`

- **env4**: Must press button and get key
  - MLP: `neighbors_unprivileged`, `door_unprivileged`, `doors_unlocked_unprivileged`, `position`, `has_key`
  - CNN: `^$` (empty)
  - **Substitution**: `neighbors` → `neighbors_unprivileged`, `door` → `door_unprivileged`, `doors_unlocked` → `doors_unlocked_unprivileged`

### Prefix-Based Substitution System

The evaluation system uses a **prefix-based substitution** approach:

1. **For each privileged key** that the agent expects (e.g., `neighbors`)
2. **Look for unprivileged keys** that start with `{key}_unprivileged` (e.g., `neighbors_unprivileged_key`)
3. **Substitute** the unprivileged key for the privileged key
4. **If no match found**, zero out the observation

**Examples:**
- `neighbors` → `neighbors_unprivileged_key` (if available)
- `neighbors` → `neighbors_unprivileged_button` (if available)
- `neighbors` → `neighbors_unprivileged` (if available)
- `door` → `door_unprivileged` (if available)

**Benefits:**
- **Flexible naming**: Any suffix after `_unprivileged` is allowed
- **Unique mapping**: Only one unprivileged key per privileged key in each eval config
- **Automatic substitution**: No need for explicit mappings
- **Backward compatible**: Works with existing `_unprivileged` suffix convention

### Maze Environment

- **env1**: Full privileged observations
  - MLP: `goal_position`, `position`, `neighbors_3x3`
  - CNN: `neighbors_5x5`

- **env2**: Only using neighbors_3x3
  - MLP: `goal_position_unprivileged`, `position`, `neighbors_3x3`
  - CNN: `neighbors_5x5_unprivileged`
  - **Substitution**: `goal_position` → `goal_position_unprivileged`, `neighbors_5x5` → `neighbors_5x5_unprivileged`

- **env3**: Only using neighbors_5x5
  - MLP: `goal_position_unprivileged`, `position`, `neighbors_3x3_unprivileged`
  - CNN: `neighbors_5x5`
  - **Substitution**: `goal_position` → `goal_position_unprivileged`, `neighbors_3x3` → `neighbors_3x3_unprivileged`

- **env4**: Use neighbors_3x3 and goal position
  - MLP: `goal_position`, `position`, `neighbors_3x3`
  - CNN: `neighbors_5x5_unprivileged`
  - **Substitution**: `neighbors_5x5` → `neighbors_5x5_unprivileged`

- **env5**: Use neighbors_5x5 and goal position
  - MLP: `goal_position`, `position`, `neighbors_3x3_unprivileged`
  - CNN: `neighbors_5x5`
  - **Substitution**: `neighbors_3x3` → `neighbors_3x3_unprivileged`

- **env6**: Only using both sets of neighbors, no goal position
  - MLP: `goal_position_unprivileged`, `position`, `neighbors_3x3`
  - CNN: `neighbors_5x5`
  - **Substitution**: `goal_position` → `goal_position_unprivileged`

### Blind Pick Environment

- **env1**: Full privileged observations
  - MLP: `.*` (all robot_state, touch, etc.)
  - CNN: `.*` (all cameras)

- **env2**: Robot state, NO touch, but using all cameras
  - MLP: `robot_state`
  - CNN: `.*` (all cameras)

- **env3**: NO robot state, touch, but using all cameras
  - MLP: `touch`
  - CNN: `.*` (all cameras)

- **env4**: NO robot state, NO touch, ONLY all cameras
  - MLP: `^$` (empty)
  - CNN: `.*` (all cameras)

- **env5**: Robot state, touch, but only wrist camera
  - MLP: `.*` (all)
  - CNN: `gripper_camera_rgb`

- **env6**: No robot state, touch, but only front camera
  - MLP: `.*` (all)
  - CNN: `camera_front`

- **env7**: Robot state, touch, but only side camera
  - MLP: `.*` (all)
  - CNN: `camera_side`

## Logging

When using the `--training_env` parameter, the run name will include the environment subset:

- Default: `ppo_rnn_gymnasium_TigerDoorKey-v0_0`
- env1: `ppo_rnn_gymnasium_TigerDoorKey-v0_0_env1`
- env2: `ppo_rnn_gymnasium_TigerDoorKey-v0_0_env2`
- etc.

## Configuration

The system supports various configuration options:

- **Network size**: `large_network`, `small_network`
- **RNN size**: `large_rnn`, `small_rnn`
- **Entropy**: `high_entropy`, `low_entropy`
- **Learning rate**: `fast_learning`, `slow_learning`

Example:
```bash
python baselines/ppo_rnn/train.py --configs gymnasium_tigerkeydoor large_network high_entropy --training_env env2
``` 