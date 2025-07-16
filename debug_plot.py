import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Debug script to check data fetching
api = wandb.Api()

# Specific run from the user's log
run_path = "vlongle/ppo_dropout_debug_v2/0zus7jh9"
metric = "full_eval_return/env2/mean_return"

print(f"Fetching run: {run_path}")
run = api.run(run_path)

# Test different sample sizes
for samples in [None, 1000, 5000, 10000, 50000]:
    if samples is None:
        hist = run.history()
        print(f"\nDefault history fetch:")
    else:
        hist = run.history(samples=samples)
        print(f"\nHistory with samples={samples}:")
    
    if metric in hist.columns:
        metric_data = hist[['global_step', metric]].dropna()
        print(f"  Total rows: {len(hist)}")
        print(f"  Rows with {metric}: {len(metric_data)}")
        print(f"  Step range: {metric_data['global_step'].min()} - {metric_data['global_step'].max()}")
        print(f"  Value range: {metric_data[metric].min():.2f} - {metric_data[metric].max():.2f}")
    else:
        print(f"  Metric {metric} not found!")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Default fetch
hist_default = run.history()
if metric in hist_default.columns:
    data_default = hist_default[['global_step', metric]].dropna()
    axes[0].plot(data_default['global_step'], data_default[metric], 'r-', label=f'Default ({len(data_default)} points)')
    axes[0].set_title('Default history() fetch')
    axes[0].legend()

# High sample fetch
hist_full = run.history(samples=50000)
if metric in hist_full.columns:
    data_full = hist_full[['global_step', metric]].dropna()
    axes[1].plot(data_full['global_step'], data_full[metric], 'b-', label=f'samples=50000 ({len(data_full)} points)')
    axes[1].set_title('Full history fetch')
    axes[1].legend()

for ax in axes:
    ax.set_xlabel('global_step')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_data_fetch.png')
print("\nSaved comparison plot to debug_data_fetch.png") 