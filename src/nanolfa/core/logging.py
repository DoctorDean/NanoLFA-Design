"""Structured logging configuration for the NanoLFA-Design pipeline.

Configures log formatting, file output, and per-round log rotation
so that HPC cluster runs produce clean, parseable log files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    run_name: str = "nanolfa",
    console: bool = True,
) -> logging.Logger:
    """Configure structured logging for the pipeline.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: If set, write logs to this directory.
        run_name: Base name for log files.
        console: Whether to also log to stderr.

    Returns:
        Root logger configured with handlers.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{run_name}.log", mode="a",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "urllib3", "rdkit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def setup_round_log(
    round_num: int,
    log_dir: Path,
    run_name: str = "nanolfa",
) -> logging.FileHandler:
    """Add a per-round log file handler.

    Each round gets its own log file for easy debugging of individual
    rounds on HPC clusters.

    Args:
        round_num: Current design round number.
        log_dir: Directory for round log files.
        run_name: Base name prefix.

    Returns:
        The FileHandler (caller can remove it after the round).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    round_path = log_dir / f"{run_name}_round_{round_num:02d}.log"

    handler = logging.FileHandler(round_path, mode="w")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logging.getLogger().addHandler(handler)
    logging.getLogger(__name__).info(
        "Round %d log: %s", round_num, round_path,
    )
    return handler
