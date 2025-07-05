#!/usr/bin/env python3
"""run_ppo.py

Adaptation of ppo.sh into Python with optional Slurm submission.

Usage examples:
---------------
Run locally (sequentially):
    python run_ppo.py --configs gymnasium_tigerdoorkey gymnasium_maze --num_seeds 3

Submit each run as its own Slurm job:
    python run_ppo.py --slurm --configs gymnasium_tigerdoorkey --num_seeds 5 \
        --job_name ppo_tiger --partition gpu --gpus 1 --time 04:00:00

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

def submit_to_slurm(cmd: str, *, idx: int, args: argparse.Namespace) -> None:
    """Submit *cmd* to Slurm via *sbatch --wrap* with CLI-provided options."""
    out_dir = Path("slurm_outs/ppo")
    out_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{args.job_name}_{idx}"
    sbatch_cmd: List[str] = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={out_dir}/{job_name}-%j.out",
        f"--time={args.time}",
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        f"--gpus={args.gpus}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus}",
        "--wrap",
        cmd,
    ]

    print("[sbatch]", " ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd, check=True)


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PPO baselines sequentially or submit them as Slurm jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution mode
    p.add_argument("--slurm", action="store_true", help="Submit each command via Slurm instead of running locally.")

    # Core experiment parameters
    p.add_argument(
        "--configs",
        nargs="+",
        default=["gymnasium_maze"],
        help="List of config names passed to baselines/ppo/train.py",
    )
    p.add_argument("--num_seeds", type=int, default=1, help="Number of consecutive seeds to run per config.")
    p.add_argument("--seed", type=int, help="Initial seed. If omitted, one is generated.")
    p.add_argument("--train_script", default="baselines/ppo/train.py", help="Path to training script.")
    p.add_argument("--timeout_hours", type=float, default=72.0, help="Per-run timeout when running locally.")
    p.add_argument(
        "--base_logdir",
        default="~/logdir/baselines/ppo",
        help="Base directory for logs (informational only – train.py controls actual output).",
    )
    p.add_argument("--wandb_project", default="sensor-dropout", help="Wandb project name.")

    # Slurm options (used only when --slurm is provided)
    p.add_argument("--job_name", default="ppo", help="Base Slurm job name.")
    p.add_argument("--time", default="72:00:00", help="Slurm time limit (HH:MM:SS)")
    p.add_argument("--partition", default="eaton-compute", help="Slurm partition")
    p.add_argument("--qos", default="ee-high", help="Quality of service")
    p.add_argument("--gpus", default="1", help="GPUs per job (value passed to --gpus)")
    p.add_argument("--mem", default="32G", help="Memory per job")
    p.add_argument("--cpus", default="64", help="CPUs per task")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Activate EGL rendering for MuJoCo on headless nodes
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Seed generation
    initial_seed = args.seed if args.seed is not None else generate_unique_seed()
    seeds = [initial_seed + i for i in range(args.num_seeds)]

    base_logdir = Path(os.path.expanduser(args.base_logdir))
    base_logdir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(args.timeout_hours * 3600)

    cmd_idx = 0
    for config in args.configs:
        for seed in seeds:
            logdir = base_logdir / f"{config}_{seed}"
            logdir_str = str(logdir)
            print(f"Running PPO baseline with config {config} and seed {seed}, logging to {logdir_str}")

            cmd = (
                f"python -u {args.train_script} "
                f"--configs {config} --seed {seed} --wandb_project {args.wandb_project}"
            )

            if args.slurm:
                submit_to_slurm(cmd, idx=cmd_idx, args=args)
                cmd_idx += 1
                continue

            subprocess.run(cmd, shell=True, check=True, timeout=timeout_seconds)

    if args.slurm:
        print("✅ All jobs submitted to Slurm. Run without --slurm after completion to execute additional steps if needed.")


if __name__ == "__main__":
    main() 