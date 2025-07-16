#!/usr/bin/env python3
"""run_baseline.py

Runner script for PPO baselines with optional Slurm submission.

Usage examples:
---------------
Run ppo locally (sequentially):
    python train.py ppo --configs gymnasium_tigerdoorkey gymnasium_maze --num_seeds 3

Submit ppo_dropout as its own Slurm job:
    python train.py ppo_dropout --slurm --configs gymnasium_tigerdoorkey --num_seeds 5 \
        --job_name ppo_dropout_tiger --partition gpu --gpus 1 --time 04:00:00

The script mirrors ppo.sh behaviour, including a 4-hour timeout when running
locally and automatically setting ``MUJOCO_GL=egl``.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from pathlib import Path
from typing import List
from slurm_utils import submit_to_slurm, add_slurm_args, slurm_kwargs_from_args

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def generate_unique_seed() -> int:
    """Generate a (reasonably) unique 32-bit integer seed.

    The implementation mimics the shell script which hashed the current
    timestamp to obtain 8 hex digits. We simply mask the current nanosecond
    timestamp plus randomness into 32 bits.
    """
    ts_ns = time.time_ns() & 0xFFFF_FFFF
    return (ts_ns ^ random.getrandbits(32)) & 0xFFFF_FFFF




# -----------------------------------------------------------------------------
# Slurm helper
# -----------------------------------------------------------------------------

# (Removed the custom submit_to_slurm helper – we now rely on slurm_utils.submit_to_slurm)


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PPO baselines sequentially or submit them as Slurm jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core experiment parameters
    p.add_argument(
        "--baseline",
        choices=["ppo", "ppo_dropout"],
        required=True,
        help="The baseline to run ('ppo' or 'ppo_dropout').",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["gymnasium_maze"],
        help="List of config names passed to the training script.",
    )
    p.add_argument("--num_seeds", type=int, default=1, help="Number of consecutive seeds to run per config.")
    p.add_argument("--seed", type=int, help="Initial seed. If omitted, one is generated.")
    p.add_argument("--train_script", default=None, help="Path to training script (inferred from baseline if not set).")
    p.add_argument("--timeout_hours", type=float, default=72.0, help="Per-run timeout when running locally.")
    p.add_argument(
        "--base_logdir",
        default=None,
        help="Base directory for logs (inferred from baseline if not set).",
    )
    p.add_argument("--wandb_project", default=None, help="Wandb project name. If not set, a name is generated from baseline and configs.")

    # PPO Dropout specific arguments
    p.add_argument(
        "--masking_strategy",
        choices=["cycle", "uniform", "adaptive"],
        default="cycle",
        help="Masking strategy for ppo_dropout baseline (cycle, random, or adaptive).",
    )
    p.add_argument("--version", type=int, default=1, help="Version number for wandb project.")

    # Slurm flags (including --slurm) via shared helper
    add_slurm_args(p)
    p.set_defaults(job_name=None)  # Will be set based on baseline

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Set dynamic defaults based on baseline ---
    baseline = args.baseline
    if args.train_script is None:
        args.train_script = f"baselines/{baseline}/train.py"
    if args.base_logdir is None:
        args.base_logdir = f"~/logdir/baselines/{baseline}"
    if args.job_name is None:
        args.job_name = baseline

    if args.wandb_project is None:
        configs_str = "-".join(sorted(args.configs))
        # Include masking strategy in project name for ppo_dropout
        args.wandb_project = f"{baseline}-{args.masking_strategy}-{configs_str}-v{args.version}"
        print(f"INFO: --wandb_project not set, generated project name: '{args.wandb_project}'")
    # ---

    # Activate EGL rendering for MuJoCo on headless nodes
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Seed generation
    # initial_seed = args.seed if args.seed is not None else generate_unique_seed()
    initial_seed = 42 ## deterministic seed for reproducibility
    seeds = [initial_seed + i for i in range(args.num_seeds)]

    base_logdir = Path(os.path.expanduser(args.base_logdir))
    base_logdir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(args.timeout_hours * 3600)

    # Prepare common Slurm kwargs once (if needed)
    slurm_common = slurm_kwargs_from_args(args) | {"out_dir": f"slurm_outs/{baseline}"}

    cmd_idx = 0
    for config in args.configs:
        for seed in seeds:
            logdir = base_logdir / f"{config}_{seed}"
            logdir_str = str(logdir)
            print(f"Running {baseline.upper()} baseline with config {config} and seed {seed}, logging to {logdir_str}")

            cmd = (
                f"python -u {args.train_script} "
                f"--configs {config} --seed {seed} --wandb_project {args.wandb_project}"
            )

            # Add masking strategy for ppo_dropout
            if baseline == "ppo_dropout":
                cmd += f" --masking_strategy {args.masking_strategy}"

            if args.slurm:
                submit_to_slurm(cmd, idx=cmd_idx, **slurm_common)
                cmd_idx += 1
                continue

            subprocess.run(cmd, shell=True, check=True, timeout=timeout_seconds)

    if args.slurm:
        print("✅ All jobs submitted to Slurm. Run without --slurm after completion to execute additional steps if needed.")


if __name__ == "__main__":
    main() 