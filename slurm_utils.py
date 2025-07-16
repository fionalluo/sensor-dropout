#!/usr/bin/env python3
"""slurm_utils.py – reusable helper for submitting commands to Slurm.

Import *submit_to_slurm* from this module wherever you need to queue a shell
command via ``sbatch --wrap``.  The signature matches the resource flags we
commonly use across scripts (job name, GPUs, memory, CPUs, etc.).
"""
from __future__ import annotations
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import List

def get_conda_env(env_name):
    conda_init = "source $(conda info --base)/etc/profile.d/conda.sh"
    acv = f"bash -c '{conda_init} && conda activate {env_name} &&"
    return acv

def run_cmd(cmd, step_name="", *, conda_env=None, allow_error=False, dry_run=False):
    """Execute a shell command with optional conda-environment wrapping.

    Parameters
    ----------
    cmd : str | list
        The command to run. Lists are automatically joined by spaces for
        convenience, so you can write ``["python", "script.py", "--arg", "val"]``
        instead of hand-crafting long f-strings.
    step_name : str, optional
        Human-readable label used in logs.
    conda_env : str, optional
        If given, the command is executed inside that conda environment using
        the same *conda activate* wrapper used elsewhere in the codebase.
    allow_error : bool, default False
        When *False*, a non-zero exit status will terminate the program.
    dry_run : bool, default False
        Log the command but do not execute it. Useful for debugging.
    """

    # Convert list-based commands to a single string.
    if isinstance(cmd, list):
        cmd = " ".join(map(str, cmd))

    if conda_env:
        conda_cmd = get_conda_env(conda_env)
        cmd = f"{conda_cmd} {cmd}'"  # close the single quote opened in get_conda_env

    logging.info(f"[RUN] {step_name} | {cmd}")

    if dry_run:
        logging.info("[RUN] Dry-run enabled; command not executed.")
        return True

    exit_code = os.system(cmd)
    if exit_code != 0 and not allow_error:
        logging.error(f"[RUN] Stopping pipeline at step: {step_name}")
        sys.exit(exit_code)

    logging.info(f"[RUN] Successfully completed step: {step_name}")
    return exit_code == 0


def submit_to_slurm(
    cmd: str,
    *,
    idx: int = 0,
    job_name: str = "job",
    time: str = "72:00:00",
    partition: str = "eaton-compute",
    qos: str = "ee-high",
    gpus: str = "1",
    mem: str = "32G",
    cpus: str = "32",
    out_dir: str | Path = "slurm_outs",
    exclude: str | None = None,
) -> None:
    """Submit *cmd* to Slurm using ``sbatch --wrap``.

    Parameters
    ----------
    cmd : str
        Shell command to execute inside the job.
    idx : int, optional
        Index appended to *job_name* so that parallel submissions are unique.
    job_name : str
        Base Slurm job name.
    time : str
        Slurm wall-time limit (``HH:MM:SS``).
    partition, qos, gpus, mem, cpus : str
        Standard Slurm resource flags.
    out_dir : str | Path
        Directory where ``sbatch`` stdout files are written (will be created).
    exclude : str, optional
        Comma-separated list of nodes to exclude from job allocation.
    """
    out_path = Path(out_dir) / job_name
    out_path.mkdir(parents=True, exist_ok=True)

    job_full_name = f"{job_name}_{idx}"
    sbatch_cmd: List[str] = [
        "sbatch",
        f"--job-name={job_full_name}",
        f"--output={out_path}/{job_full_name}-%j.out",
        f"--time={time}",
        f"--partition={partition}",
        f"--qos={qos}",
        f"--gpus={gpus}",
        f"--mem={mem}",
        f"--cpus-per-task={cpus}",
    ]
    if exclude:
        sbatch_cmd.append(f"--exclude={exclude}")

    sbatch_cmd.extend(["--wrap", cmd])

    print("[sbatch]", " ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd, check=True)

def add_slurm_args(parser):
    """Attach common Slurm CLI flags to an ``argparse.ArgumentParser``.

    Call this in every script that supports ``--slurm`` execution to avoid
    repeating boilerplate option definitions.
    """
    group = parser.add_argument_group("Slurm options")
    group.add_argument('--slurm', action='store_true', help='Submit job via Slurm instead of running locally.')
    group.add_argument('--job_name', default='job', help='Base Slurm job name')
    group.add_argument('--time', default='72:00:00', help='Slurm time limit (HH:MM:SS)')
    group.add_argument('--partition', default='eaton-compute', help='Slurm partition')
    group.add_argument('--qos', default='ee-high', help='Quality of service')
    group.add_argument('--gpus', default='1', help='GPUs per job')
    group.add_argument('--mem', default='32G', help='Memory per job')
    group.add_argument('--cpus', default='32', help='CPUs per task')
    group.add_argument(
        '--exclude',
        default='kd-2080ti-[1-4].grasp.maas,dj-2080ti-0.grasp.maas,mp-2080ti-0.grasp.maas',
        help='Comma-separated list of nodes to exclude.'
    )
    return group

def slurm_kwargs_from_args(args):
    """Return dict of kwargs accepted by :func:`submit_to_slurm` extracted from *args*."""
    keys = ('job_name', 'time', 'partition', 'qos', 'gpus', 'mem', 'cpus', 'exclude')
    return {k: getattr(args, k) for k in keys} 