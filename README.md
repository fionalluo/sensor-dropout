# sensor-dropout

A project for studying sensor dropout and observation space subsets in reinforcement learning environments.

## Environment Setup

This project requires setting up a conda environment and installing the necessary dependencies, along with two main components: the grid environments and the Gymnasium Robotics environments.

### Conda Environment Setup

1. Create a new conda environment with Python 3.9:
   ```bash
   conda create -n dropout python=3.9
   ```

2. Activate the conda environment:
   ```bash
   conda activate dropout
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the current project as a local package:
   ```bash
   pip install -e .
   ```

### Grid Setup

1. Clone the trailenv repository:
   ```bash
   git clone https://github.com/fionalluo/trailenv
   ```

2. Navigate to the trailenv directory:
   ```bash
   cd trailenv
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Gymnasium Robotics Setup

1. Clone the Gymnasium-Robotics repository:
   ```bash
   git clone https://github.com/fionalluo/Gymnasium-Robotics
   ```

2. Navigate to the Gymnasium-Robotics directory:
   ```bash
   cd Gymnasium-Robotics
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Workflow

This section describes how to run the different baselines and experiments in the sensor-dropout project.

### Run PPO Baseline

1. **Before running**, modify the environment and number of seeds in `./ppo.sh`:
   - Environment
   - Number of runs

2. **Before running**, configure PPO parameters in `baselines/ppo/config.yaml`:
   - Expert subset configurations are defined in these config.yaml files
   - Modify hyperparameters, network architectures, and training settings as needed

3. **Execute the PPO training script:**
   ```bash
   ./ppo.sh
   ```

### Run PPO Distill Baseline

#### Step 1: Save Expert Policies

1. **Before running**, modify the environment and policy type you want in `./train_subset_policies.sh`:
   - Environment
   - Policy Type

2. **Before running**, modify the policy type specific config (e.g., `ppo/config.yaml`) for the subset policy parameters:
   - Hyperparameters

3. **Execute the script to save all expert policies.** They will be saved to `policies/{policy_type}/{task}/env{n}/....pt`:
   ```bash
   ./train_subset_policies.sh
   ```

#### Step 1a (Optional): Test Policy Loading

1. **Verify that your subset policies can be loaded correctly:**
   ```bash
   python test/test_ppo_loading.py
   ```

#### Step 2: Run Distillation

1. **Before running**, modify the environment and number of seeds in `./ppo_distill.sh`:
   - Environment
   - Number of runs

2. **Before running**, configure the PPO settings in `baselines/ppo_distill/config.yaml`:
   - Hyperparameters

3. **Execute the distillation training:**
   ```bash
   ./ppo_distill.sh
   ```

### Other Baselines

**Note**: Other folders in `baselines` like `ppo_lstm`, `ppo_rnn`, etc., are currently out of date and not compatible with the current codebase. Use the main PPO and PPO Distill baselines for now. 

## Project Structure

```
sensor-dropout/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Poetry configuration
├── poetry.lock                 # Poetry lock file
│
├── baselines/                   # Main algorithm implementations
│   ├── ppo/                    # PPO baseline
│   │   ├── agent.py           # PPO agent implementation
│   │   ├── config.yaml        # PPO configuration
│   │   ├── ppo.py             # PPO training logic
│   │   └── train.py           # Training script
│   │
│   ├── ppo_distill/           # PPO distillation baseline
│   │   ├── agent.py           # Distillation agent
│   │   ├── config.yaml        # Distillation config
│   │   ├── ppo_distill.py     # Distillation logic
│   │   └── train.py           # Training script
│   │
│   ├── shared/                # Shared utilities
│   │   ├── agent.py           # Base agent class
│   │   ├── config_utils.py    # Configuration utilities
│   │   ├── env_utils.py       # Environment utilities
│   │   ├── eval_utils.py      # Evaluation utilities
│   │   ├── masking_utils.py   # Observation masking
│   │   ├── nets.py            # Neural network architectures
│   │   └── policy_utils.py    # Policy utilities
│   │
│   ├── ppo_lstm/              # LSTM variant (outdated)
│   ├── ppo_rnn/               # RNN variant (outdated)
│   └── ppo_moe/               # Mixture of experts (outdated)
│
├── embodied/                   # Embodied AI framework
│   ├── core/                  # Core utilities
│   ├── envs/                  # Environment wrappers
│   ├── replay/                # Replay buffer implementations
│   ├── run/                   # Training runners
│   └── scripts/               # Installation scripts
│
├── subset_policies/           # Subset policy training
│   ├── train_subset_policies.py
│   ├── load_subset_policy.py
│   └── example_usage.py
│
├── thesis/                    # Thesis-related code (OUTDATED)
│
├── test/                      # Testing utilities
│   └── test_ppo_loading.py   # Policy loading tests
│
├──ppo.sh                # PPO training script
├──ppo_distill.sh        # Distillation training script
├──train_subset_policies.sh
├──ppo_lstm.sh           # LSTM training (outdated)
├──ppo_rnn.sh            # RNN training (outdated)
```

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]

