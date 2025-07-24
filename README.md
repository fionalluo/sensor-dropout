# sensor-dropout

A project for studying sensor dropout and observation space subsets in reinforcement learning environments.

## Environment Setup

This project requires setting up a conda environment and installing the necessary dependencies, along with two main components: the grid environments and the Gymnasium Robotics environments.

### Repository Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/fionalluo/sensor-dropout
   cd sensor-dropout
   ```

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
All baselines have been replaced to run with SB3, however the commands below are the exact same.

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

### Run PPO Dropout Baseline
Note: The logging in the PPO dropout baseline on wandb may be outdated for now (like not showing the accurate number of steps) but it still runs fine. 

1. **Before running**, modify the environment and number of seeds in `./ppo_dropout.sh`:
   - Environment
   - Number of runs

2. **Before running**, configure PPO parameters in `baselines/ppo_dropout/config.yaml`:
   - Dropout rates and layers
   - Other hyperparameters and training settings

3. **Execute the PPO dropout training script:**
   ```bash
   ./ppo_dropout.sh
   ```


##### Understanding the Masking Logic

The observation masking process works as follows:

1. **Data Collection**
   - Full observations are collected from the environment
   - No masking is applied during collection

2. **Teacher Processing**
   - Teacher receives observations filtered according to its specific keys
   - Each teacher has its own set of observation keys defined in config.yaml

3. **Student Processing**
   - Student has its own set of defined observation keys
   - However, observations are first masked using the corresponding teacher's keys
   - Important: Even though student sees observations with student key names, the actual values come from teacher's masked observations
   - This means student's effective observation space matches the teacher it's learning from, regardless of its configured keys

This masking mechanism ensures the student learns from teacher demonstrations while maintaining consistent observation spaces during knowledge transfer.

### Other Baselines

**Note**: Previous baselines using cleanrl rather than SB3 have been moved to the branch fiona/draft. Use the SB3 baselines as default.

## Project Structure

[Add file tree here]

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]

