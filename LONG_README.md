## Baselines
- PPO:
pick one observation-space from `ppo/config.yaml`. Default to train all obs
- PPO Distill
    - train multiple expert subset polciies under `policies/<policy_type>/<task>`
    - distill to a single student policy.

## Configs
`keys`: all possible sensors, `keys`: what being used by the env (student sensor). 

## Eval Wandb
`eval/mean_return` the result on the training sensor config, `full_eval_return/mean` the result on the average all possible config env combination, which is usually the key metric to track and is a superset of `eval/mean_return`.


## Running

```
python train.py --baseline ppo_dropout --configs gymnasium_tigerdoorkey --num_seeds 4 [--slurm]
```

## Plotting

```
python sensor-dropout/plot/plot.py --projects ppo_dropout-gymnasium_tigerdoorkey ppo_dropout-adaptive-gymnasium_tigerdoorkey --run_dir sensor-dropout/wandb --ymin 0
```

```
python sensor-dropout/plot/plot.py --projects ppo_dropout-gymnasium_maze ppo_dropout-adaptive-gymnasium_maze --run_dir sensor-dropout/wandb --ymin 0
```


```
python sensor-dropout/plot/plot.py --projects ppo_dropout-gymnasium_maze11 ppo_dropout-adaptive-gymnasium_maze11 --run_dir sensor-dropout/wandb --ymin 0
```


## Dropout fancier tricks

gymnasium_tigerdoorkey