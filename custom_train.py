from slurm_utils import run_cmd


if __name__ == "__main__":
    num_seeds = 4
    num_jobs = 0
    use_batch_freq = 2
    tiger_configs = [
               "debug_tiger_cycle", 
               "debug_tiger_adaptive_window1", 
               "debug_tiger_adaptive_warmup_window1",
               "debug_tiger_adaptive_warmup_window100",
               "debug_tiger_adaptive_warmup_window100_ema"
               ]
    maze_configs = [
        "debug_maze_cycle",
        "debug_maze_adaptive_window1",
    ]
    # version = 1 ## num_workers = 32
    version = 2 ## num_workers = 8

    configs = maze_configs
    
    # Use the new debug configs which have masking strategy embedded
    for config in configs:
        for seed in range(42, 42 + num_seeds):
            cmd = f"python train.py --baseline ppo_dropout --configs {config} --seed {seed} --slurm --version {version}"
            num_jobs += 1
            if num_jobs % use_batch_freq == 0:
                cmd += " --partition batch --qos normal"
            run_cmd(cmd,
                    conda_env="rl_sensor")