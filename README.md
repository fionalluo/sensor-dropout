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
   - Change the environment variable to specify which environment to run on
   - Adjust the number of seeds for reproducibility

2. **Before running**, configure PPO parameters in `baselines/ppo/config.yaml`:
   - Expert subset configurations are defined in these config.yaml files
   - Modify hyperparameters, network architectures, and training settings as needed

3. Execute the PPO training script:
   ```bash
   ./ppo.sh
   ```

### Run PPO Distill Baseline

#### Step 1: Save Expert Policies

1. **Before running**, modify the environment and policy type you want in `./train_subset_policies.sh`:
   - Change the environment variable
   - Specify the policy type to train

2. **Before running**, modify the policy type specific config (e.g., `ppo/config.yaml`) for the subset policy parameters:
   - Adjust hyperparameters for the subset policies
   - Configure observation space subsets

3. Execute the script to save all expert policies. They will be saved to `policies/{policy_type}/{task}/env{n}/....pt`:
   ```bash
   ./train_subset_policies.sh
   ```

#### Step 1a (Optional): Test Policy Loading

1. Verify that your subset policies can be loaded correctly:
   ```bash
   python test/test_ppo_loading.py
   ```

#### Step 2: Run Distillation

1. **Before running**, modify the environment and number of seeds in `./ppo_distill.sh`:
   - Change the environment to run on
   - Adjust the number of seeds for reproducibility

2. **Before running**, configure the PPO settings in `baselines/ppo_distill/config.yaml`:
   - Configure distillation-specific parameters
   - Set up teacher-student learning settings

3. Execute the distillation training:
   ```bash
   ./ppo_distill.sh
   ```

### Other Baselines

**Note**: Other folders in `baselines` like `ppo_lstm`, `ppo_rnn`, etc., are currently out of date and not compatible with the current codebase. Use the main PPO and PPO Distill baselines for now. 

## Project Structure

[Add project structure information here]

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]

