from slurm_utils import run_cmd


if __name__ == "__main__":
    num_seeds = 4
    version = 1
    num_jobs = 0
    for strat in ["cycle", "adaptive"]:
        # for config in ["gymnasium_tigerdoorkey", "gymnasium_maze", "gymnasium_maze11"]:
        for config in ["debug_gymnasium_tigerdoorkey"]:
            cmd = f"python train.py --baseline ppo_dropout --configs {config} --num_seeds {num_seeds} --masking_strategy {strat} --slurm --version {version}"
            num_jobs += 1
            # use_batch_partition = False
            # if num_jobs % 4 == 0:
            #     use_batch_partition = True
            # if use_batch_partition:
            #     cmd += " --partition batch --qos normal"
            run_cmd(cmd,
                    conda_env="rl_sensor")