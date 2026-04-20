"""HPC job submission and monitoring for Slurm and PBS clusters.

Manages the submission, tracking, and result collection of GPU-intensive
pipeline jobs (AlphaFold, ProteinMPNN, RFdiffusion) on HPC schedulers.

Supports three execution modes:
- slurm: SBATCH-based submission via submitit or direct sbatch
- pbs: qsub-based submission
- local: direct subprocess execution (for development/debugging)
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Scheduler(Enum):
    SLURM = "slurm"
    PBS = "pbs"
    LOCAL = "local"


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class HPCJob:
    """Represents a submitted HPC job."""

    job_id: str
    job_name: str
    command: list[str]
    work_dir: Path
    scheduler: Scheduler
    status: JobStatus = JobStatus.PENDING
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    exit_code: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None

    @property
    def wall_time_hours(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) / 3600
        return 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class HPCManager:
    """Manage job submission and monitoring on HPC clusters.

    Abstracts Slurm and PBS behind a unified interface. For local
    execution (development), runs jobs as subprocesses.

    Usage:
        manager = HPCManager(config.compute)
        job = manager.submit(
            command=["python", "scripts/run_design_round.py", ...],
            job_name="af3_round_01",
            work_dir=Path("data/results/round_01"),
        )
        manager.wait(job)
    """

    def __init__(self, config: DictConfig) -> None:
        self.scheduler = Scheduler(config.scheduler)
        self.partition = config.partition
        self.gpus_per_node = config.gpus_per_node
        self.cpus_per_task = config.cpus_per_task
        self.memory_gb = config.memory_gb
        self.time_limit_hours = config.time_limit_hours
        self.container = config.get("container")

        self._active_jobs: dict[str, HPCJob] = {}

        logger.info(
            "HPC manager: scheduler=%s, partition=%s, gpus=%d",
            self.scheduler.value, self.partition, self.gpus_per_node,
        )

    def submit(
        self,
        command: list[str],
        job_name: str,
        work_dir: Path,
        gpus: int | None = None,
        cpus: int | None = None,
        memory_gb: int | None = None,
        time_hours: int | None = None,
        array_size: int | None = None,
        dependencies: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> HPCJob:
        """Submit a job to the scheduler.

        Args:
            command: Command to execute (list of strings).
            job_name: Human-readable job name.
            work_dir: Working directory for the job.
            gpus: Number of GPUs (default from config).
            cpus: CPUs per task (default from config).
            memory_gb: Memory in GB (default from config).
            time_hours: Time limit in hours (default from config).
            array_size: If set, submit as a job array of this size.
            dependencies: Job IDs this job depends on (afterok).
            env_vars: Additional environment variables.

        Returns:
            HPCJob with the submitted job's ID and metadata.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        gpus = gpus or self.gpus_per_node
        cpus = cpus or self.cpus_per_task
        memory_gb = memory_gb or self.memory_gb
        time_hours = time_hours or self.time_limit_hours

        if self.scheduler == Scheduler.SLURM:
            job = self._submit_slurm(
                command, job_name, work_dir, gpus, cpus,
                memory_gb, time_hours, array_size, dependencies, env_vars,
            )
        elif self.scheduler == Scheduler.PBS:
            job = self._submit_pbs(
                command, job_name, work_dir, gpus, cpus,
                memory_gb, time_hours, dependencies, env_vars,
            )
        else:
            job = self._submit_local(command, job_name, work_dir, env_vars)

        self._active_jobs[job.job_id] = job
        logger.info("Submitted job %s: %s", job.job_id, job_name)
        return job

    def status(self, job: HPCJob) -> JobStatus:
        """Query the current status of a job."""
        if job.is_terminal:
            return job.status

        if self.scheduler == Scheduler.SLURM:
            return self._status_slurm(job)
        elif self.scheduler == Scheduler.PBS:
            return self._status_pbs(job)
        return job.status

    def wait(
        self,
        job: HPCJob,
        poll_interval: int = 30,
        timeout_hours: float | None = None,
    ) -> HPCJob:
        """Block until a job completes, fails, or times out.

        Args:
            job: Job to wait for.
            poll_interval: Seconds between status checks.
            timeout_hours: Maximum wait time (None = unlimited).

        Returns:
            Updated HPCJob with final status.
        """
        start = time.time()
        timeout_sec = timeout_hours * 3600 if timeout_hours else float("inf")

        while not job.is_terminal:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                logger.warning("Job %s timed out after %.1fh", job.job_id, elapsed / 3600)
                job.status = JobStatus.CANCELLED
                break

            job.status = self.status(job)
            if not job.is_terminal:
                logger.debug(
                    "Job %s: %s (%.0fs elapsed)", job.job_id, job.status.value, elapsed
                )
                time.sleep(poll_interval)

        job.end_time = time.time()
        logger.info(
            "Job %s finished: %s (%.1fh wall time)",
            job.job_id, job.status.value, job.wall_time_hours,
        )
        return job

    def wait_all(
        self,
        jobs: list[HPCJob],
        poll_interval: int = 30,
    ) -> list[HPCJob]:
        """Wait for multiple jobs to complete."""
        pending = list(jobs)
        while pending:
            still_pending = []
            for job in pending:
                job.status = self.status(job)
                if not job.is_terminal:
                    still_pending.append(job)
            pending = still_pending
            if pending:
                logger.info(
                    "Waiting: %d/%d jobs still running", len(pending), len(jobs)
                )
                time.sleep(poll_interval)

        completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
        logger.info("All jobs done: %d completed, %d failed", completed, failed)
        return jobs

    # ------------------------------------------------------------------
    # Slurm implementation
    # ------------------------------------------------------------------

    def _submit_slurm(
        self,
        command: list[str],
        job_name: str,
        work_dir: Path,
        gpus: int,
        cpus: int,
        memory_gb: int,
        time_hours: int,
        array_size: int | None,
        dependencies: list[str] | None,
        env_vars: dict[str, str] | None,
    ) -> HPCJob:
        """Submit via sbatch."""
        # Write batch script
        script_path = work_dir / f"{job_name}.sbatch"
        stdout_path = work_dir / f"{job_name}_%j.out"
        stderr_path = work_dir / f"{job_name}_%j.err"

        time_str = f"{time_hours}:00:00"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={memory_gb}G",
            f"#SBATCH --time={time_str}",
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
            f"#SBATCH --chdir={work_dir}",
        ]

        if array_size is not None:
            lines.append(f"#SBATCH --array=0-{array_size - 1}")
        if dependencies:
            dep_str = ":".join(dependencies)
            lines.append(f"#SBATCH --dependency=afterok:{dep_str}")

        lines.append("")

        # Environment setup
        if env_vars:
            for key, val in env_vars.items():
                lines.append(f"export {key}={val}")
            lines.append("")

        # Container wrapper
        if self.container:
            cmd_str = " ".join(command)
            lines.append(f"singularity exec --nv {self.container} {cmd_str}")
        else:
            lines.append(" ".join(command))

        script_path.write_text("\n".join(lines) + "\n")

        # Submit
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, cwd=str(work_dir),
        )

        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        # Parse job ID from "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]

        return HPCJob(
            job_id=job_id,
            job_name=job_name,
            command=command,
            work_dir=work_dir,
            scheduler=Scheduler.SLURM,
            submit_time=time.time(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def _status_slurm(self, job: HPCJob) -> JobStatus:
        """Query job status via sacct."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job.job_id, "--format=State", "--noheader", "-P"],
                capture_output=True, text=True, timeout=10,
            )
            state = result.stdout.strip().split("\n")[0].strip()
            return _SLURM_STATE_MAP.get(state, JobStatus.UNKNOWN)
        except Exception:
            return JobStatus.UNKNOWN

    # ------------------------------------------------------------------
    # PBS implementation
    # ------------------------------------------------------------------

    def _submit_pbs(
        self,
        command: list[str],
        job_name: str,
        work_dir: Path,
        gpus: int,
        cpus: int,
        memory_gb: int,
        time_hours: int,
        dependencies: list[str] | None,
        env_vars: dict[str, str] | None,
    ) -> HPCJob:
        """Submit via qsub."""
        script_path = work_dir / f"{job_name}.pbs"
        stdout_path = work_dir / f"{job_name}.o"
        stderr_path = work_dir / f"{job_name}.e"

        time_str = f"{time_hours}:00:00"

        lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -q {self.partition}",
            f"#PBS -l select=1:ncpus={cpus}:ngpus={gpus}:mem={memory_gb}gb",
            f"#PBS -l walltime={time_str}",
            f"#PBS -o {stdout_path}",
            f"#PBS -e {stderr_path}",
            f"#PBS -d {work_dir}",
        ]

        if dependencies:
            dep_str = ":".join(dependencies)
            lines.append(f"#PBS -W depend=afterok:{dep_str}")

        lines.append("")
        lines.append(f"cd {work_dir}")

        if env_vars:
            for key, val in env_vars.items():
                lines.append(f"export {key}={val}")
            lines.append("")

        lines.append(" ".join(command))

        script_path.write_text("\n".join(lines) + "\n")

        result = subprocess.run(
            ["qsub", str(script_path)],
            capture_output=True, text=True, cwd=str(work_dir),
        )

        if result.returncode != 0:
            raise RuntimeError(f"qsub failed: {result.stderr}")

        job_id = result.stdout.strip()

        return HPCJob(
            job_id=job_id,
            job_name=job_name,
            command=command,
            work_dir=work_dir,
            scheduler=Scheduler.PBS,
            submit_time=time.time(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def _status_pbs(self, job: HPCJob) -> JobStatus:
        """Query job status via qstat."""
        try:
            result = subprocess.run(
                ["qstat", "-f", job.job_id],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "job_state" in line:
                    state = line.split("=")[-1].strip()
                    return _PBS_STATE_MAP.get(state, JobStatus.UNKNOWN)
            return JobStatus.COMPLETED  # qstat returns nothing for finished jobs
        except Exception:
            return JobStatus.UNKNOWN

    # ------------------------------------------------------------------
    # Local execution (development mode)
    # ------------------------------------------------------------------

    def _submit_local(
        self,
        command: list[str],
        job_name: str,
        work_dir: Path,
        env_vars: dict[str, str] | None,
    ) -> HPCJob:
        """Run job as a local subprocess (blocking)."""
        env = dict(os.environ)
        if env_vars:
            env.update(env_vars)

        stdout_path = work_dir / f"{job_name}.out"
        stderr_path = work_dir / f"{job_name}.err"

        logger.info("Running locally: %s", " ".join(command))

        with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
            result = subprocess.run(
                command, stdout=fout, stderr=ferr,
                cwd=str(work_dir), env=env,
            )

        job = HPCJob(
            job_id=f"local_{job_name}_{int(time.time())}",
            job_name=job_name,
            command=command,
            work_dir=work_dir,
            scheduler=Scheduler.LOCAL,
            submit_time=time.time(),
            start_time=time.time(),
            end_time=time.time(),
            exit_code=result.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            status=JobStatus.COMPLETED if result.returncode == 0 else JobStatus.FAILED,
        )

        return job


# State mapping tables
_SLURM_STATE_MAP: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "FAILED": JobStatus.FAILED,
    "CANCELLED": JobStatus.CANCELLED,
    "CANCELLED+": JobStatus.CANCELLED,
    "TIMEOUT": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "PREEMPTED": JobStatus.CANCELLED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
}

_PBS_STATE_MAP: dict[str, JobStatus] = {
    "Q": JobStatus.PENDING,
    "R": JobStatus.RUNNING,
    "C": JobStatus.COMPLETED,
    "E": JobStatus.RUNNING,   # exiting
    "H": JobStatus.PENDING,   # held
    "F": JobStatus.COMPLETED, # finished
}
