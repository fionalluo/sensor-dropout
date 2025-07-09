#!/usr/bin/env python3
"""slurm_utils.py – reusable helper for submitting commands to Slurm.

Import *submit_to_slurm* from this module wherever you need to queue a shell
command via ``sbatch --wrap``.  The signature matches the resource flags we
commonly use across scripts (job name, GPUs, memory, CPUs, etc.).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

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
        "--wrap",
        cmd,
    ]

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
    return group

def slurm_kwargs_from_args(args):
    """Return dict of kwargs accepted by :func:`submit_to_slurm` extracted from *args*."""
    keys = ('job_name', 'time', 'partition', 'qos', 'gpus', 'mem', 'cpus')
    return {k: getattr(args, k) for k in keys} 