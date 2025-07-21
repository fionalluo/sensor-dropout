from slurm_utils import run_cmd

def debug():
    cmd = f"python train.py --baseline ppo_dropout --configs debug_parallel_fix --seed 42 --slurm --version 1"
    run_cmd(cmd,
            conda_env="rl_sensor")

def run_tiger():
    num_seeds = 4
    output_dir = "/mnt/kostas-graid/datasets/vlongle/rl_sensors"
    tiger_configs = [
        "debug_tiger_cycle", 
        "debug_tiger_adaptive_window1", 
        "debug_tiger_adaptive_warmup_window1",
        "debug_tiger_adaptive_warmup_window100",
        "debug_tiger_adaptive_warmup_window100_ema"
    ]
    version = 5 ## fix the bugs in parallel synchronization. LOOK AT THE EMPIRICAL PROBABILITIES
    ## TO CONFIRM!
    num_jobs = 0
    use_batch_freq = 2
    for config in tiger_configs:
        for seed in range(42, 42 + num_seeds):
            cmd = f"python train.py --baseline ppo_dropout --configs {config} --seed {seed} --slurm --version {version} --output_dir {output_dir}"
            num_jobs += 1
            if num_jobs % use_batch_freq == 0:
                cmd += " --partition batch --qos normal"
            run_cmd(cmd,
                    conda_env="rl_sensor")
        

if __name__ == "__main__":
    # debug()
    run_tiger()
    # num_seeds = 4
    # num_jobs = 0
    # use_batch_freq = 2
    # tiger_configs = [
    #            "debug_tiger_cycle", 
    #            "debug_tiger_adaptive_window1", 
    #            "debug_tiger_adaptive_warmup_window1",
    #            "debug_tiger_adaptive_warmup_window100",
    #            "debug_tiger_adaptive_warmup_window100_ema"
    #            ]
    # maze_configs = [
    #     "debug_maze_cycle",
    #     "debug_maze_adaptive_window1",
    # ]
    # # version = 1 ## num_workers = 32
    # # version = 2 ## num_workers = 8
    # version = 3 ## way longer training on maze for env2 to converge!

    # configs = maze_configs
    
    # # Use the new debug configs which have masking strategy embedded
    # for config in configs:
    #     for seed in range(42, 42 + num_seeds):
    #         cmd = f"python train.py --baseline ppo_dropout --configs {config} --seed {seed} --slurm --version {version}"
    #         num_jobs += 1
    #         if num_jobs % use_batch_freq == 0:
    #             cmd += " --partition batch --qos normal"
    #         run_cmd(cmd,
    #                 conda_env="rl_sensor")