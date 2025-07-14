# PPO Dropout Any Baseline

This is a modified version of the original PPO Dropout baseline that implements **probabilistic key dropout** instead of cycling through predefined observation subsets.

## Key Differences from Original PPO Dropout

### Original PPO Dropout:
- Cycles through predefined evaluation subsets (e.g., env1, env2, env3, env4)
- Each episode uses a specific, predefined combination of observation keys
- Limited to the number of predefined subsets

### PPO Dropout Any:
- **Probabilistic dropout**: Every observation key can be dropped out with some probability
- **Dynamic combinations**: Generates 2^n - 1 possible combinations (any subset except empty)
- **Modular design**: Easily configurable probability schedules

## Core Implementation

### `ProbabilisticEpisodeMaskingWrapper`
The main wrapper that handles probabilistic key dropout:
- Replaces the original `EpisodeMaskingWrapper._select_env_subset()` method
- Implements `_select_probabilistic_keys()` for random key selection
- Ensures at least one key is always included (no empty observations)

### `DropoutScheduler`
Modular scheduler supporting different probability schedules:
- **Constant**: Fixed dropout probability throughout training
- **Linear**: Linearly interpolated dropout probability over episodes
- **Exponential**: Exponentially decaying dropout probability

## Configuration

### Basic Dropout Configuration
```yaml
dropout:
  schedule_type: 'constant'  # 'constant', 'linear', 'exponential'
  base_probability: 0.5      # 50% dropout rate
```

### Linear Schedule
```yaml
dropout:
  schedule_type: 'linear'
  start_probability: 0.8     # Start with 80% dropout
  end_probability: 0.2       # End with 20% dropout  
  total_episodes: 10000      # Over 10,000 episodes
```

### Exponential Schedule
```yaml
dropout:
  schedule_type: 'exponential'
  start_probability: 0.8     # Start with 80% dropout
  decay_rate: 0.995          # Decay rate per episode
  min_probability: 0.1       # Minimum dropout rate
```

## Usage

### Running the Baseline
```bash
# Run with default config
python baselines/ppo_dropout_any/train.py

# Run with specific config
python baselines/ppo_dropout_any/train.py --configs gymnasium_blindpick

# Run with custom seed
python baselines/ppo_dropout_any/train.py --configs gymnasium_blindpick --seed 42
```

### Using the Shell Script
```bash
# Run multiple seeds automatically
./ppo_dropout_any.sh
```

## Available Configurations

### Tiger Door Key
- **Constant dropout**: 30% dropout rate (70% inclusion)
- **Keys**: neighbors, door, doors_unlocked, position, has_key

### Maze Environments
- **Linear schedule**: 70% → 20% dropout over episodes
- **Keys**: goal_position, position, neighbors_3x3, neighbors_5x5

### Blind Pick
- **Exponential schedule**: 60% → 10% dropout with slow decay
- **Keys**: robot_state, touch, camera_front, camera_side, gripper_camera_rgb

## Modular Design Benefits

1. **Easy probability tuning**: Just change config values
2. **Schedulable**: Add new schedule types by extending `DropoutScheduler`
3. **Extensible**: Easy to add new dropout strategies
4. **Debuggable**: Clear logging of dropout decisions per episode

## Example Output

```
[PROB DROPOUT] Available keys: ['robot_state', 'touch', 'camera_front', 'camera_side', 'gripper_camera_rgb']
[PROB DROPOUT] Student keys: ['robot_state', 'touch', 'camera_front', 'camera_side', 'gripper_camera_rgb']
[PROB DROPOUT] Episode 1: dropout_prob=0.600, selected_keys=['robot_state', 'camera_front']
[PROB DROPOUT] Episode 2: dropout_prob=0.600, selected_keys=['touch', 'camera_side', 'gripper_camera_rgb']
[PROB DROPOUT] Episode 3: dropout_prob=0.600, selected_keys=['robot_state', 'touch', 'camera_front']
```

## Future Extensions

The modular design makes it easy to add:
- **Curriculum learning**: Gradually increase dropout difficulty
- **Adaptive dropout**: Adjust based on performance
- **Hierarchical dropout**: Different rates for different key types
- **Key-specific probabilities**: Different dropout rates per key 