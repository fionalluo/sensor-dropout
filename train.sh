#  python train.py --baseline ppo_dropout --configs gymnasium_tigerdoorkey --num_seeds 4 --slurm
#  python train.py --baseline ppo_dropout --configs gymnasium_maze --num_seeds 4 --slurm
# python train.py --baseline ppo_dropout --configs gymnasium_maze11 --num_seeds 4 --partition batch --qos normal --slurm

# python train.py --baseline ppo_dropout --configs gymnasium_tigerdoorkey --num_seeds 4 --masking_strategy adaptive --slurm
# python train.py --baseline ppo_dropout --configs gymnasium_maze --num_seeds 4 --slurm --masking_strategy adaptive
# python train.py --baseline ppo_dropout --configs gymnasium_maze11 --num_seeds 4 --partition batch --qos normal --slurm --masking_strategy adaptive


# python plot/plot.py --projects ppo_dropout-gymnasium_tigerdoorkey ppo_dropout-adaptive-gymnasium_tigerdoorkey --run_dir wandb --ymin 0

# python plot/plot.py --projects ppo_dropout-gymnasium_maze ppo_dropout-adaptive-gymnasium_maze --run_dir wandb --ymin 0

# python plot/plot.py --projects ppo_dropout-gymnasium_maze11 ppo_dropout-adaptive-gymnasium_maze11 --run_dir wandb --ymin 0

# python plot/plot.py --projects  ppo_dropout-gymnasium_tigerdoorkey --run_dir wandb --metrics full_eval_return/env1/mean_return full_eval_return/env2/mean_return full_eval_return/env3/mean_return full_eval_return/env4/mean_return --ymin 0


# python train.py --baseline ppo_dropout --wandb_project ppo_dropout-cycle_debug_v3 --configs debug_adaptive_tigerdoorkey --masking_strategy cycle --num_seeds 4 --slurm --partition batch --qos normal
# python train.py --baseline ppo_dropout --wandb_project ppo_dropout-adaptive_debug_v3 --configs debug_adaptive_tigerdoorkey --masking_strategy adaptive --num_seeds 4 --slurm --partition batch --qos normal
# python train.py --baseline ppo_dropout --wandb_project ppo_dropout-uniform_debug_v3 --configs debug_adaptive_tigerdoorkey --masking_strategy uniform --num_seeds 4 --slurm --partition batch --qos normal

python train.py --baseline ppo_dropout --configs gymnasium_tigerdoorkey --num_seeds 4 --masking_strategy adaptive --slurm --wandb_project ppo_dropout-adaptive_v2
python train.py --baseline ppo_dropout --configs gymnasium_maze --num_seeds 4 --slurm --masking_strategy adaptive --wandb_project ppo_dropout-adaptive_v2