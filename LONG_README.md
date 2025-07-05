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

For continuous control, SB3 PPO

```
python debug_blindpick.py --num_seeds 5 --slurm 2>&1 | tee error.logs
```


```
python run_ppo.py --slurm --configs gymnasium_blindpick --num_seeds 4 --wandb_project fiona-ppo-blindpick-with-oracle
```