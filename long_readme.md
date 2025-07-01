## Baselines
- PPO:
pick one observation-space from `ppo/config.yaml`. Default to train all obs
- PPO Distill
    - train multiple expert subset polciies under `policies/<policy_type>/<task>`
    - distill to a single student policy.