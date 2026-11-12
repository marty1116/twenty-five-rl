"""Experiment metrics, manifests, hashes, and bounded checkpoint retention."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import torch

from .. import rules
from ..observations import OBSERVATION_SCHEMA_VERSION
from .protocol import source_identity


class MetricLogger:
    """Always write JSONL metrics and optionally mirror scalar values to TensorBoard."""

    def __init__(self, checkpoint: Path, tensorboard_dir: Path | None = None, append: bool = False):
        self.path = checkpoint.with_suffix(".metrics.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append:
            self.path.write_text("")
        self._tensorboard = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError("TensorBoard logging requested. Install it with: python -m pip install -e '.[tracking]'") from exc
            self._tensorboard = SummaryWriter(log_dir=str(tensorboard_dir))

    def log(self, step: int, **metrics: float | int | str | bool) -> None:
        row: dict[str, Any] = {"step": step, "wall_time": time.time(), **metrics}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self._tensorboard is not None:
            for name, value in metrics.items():
                if isinstance(value, bool):
                    self._tensorboard.add_scalar(name, int(value), step)
                elif isinstance(value, (int, float)):
                    self._tensorboard.add_scalar(name, value, step)
            self._tensorboard.flush()

    def close(self) -> None:
        if self._tensorboard is not None:
            self._tensorboard.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TerminalProgress:
    """Dependency-free, TTY-only progress display for long training runs."""

    def __init__(
        self,
        total: int,
        *,
        initial: int = 0,
        label: str = "Training",
        enabled: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.total = total
        self.initial = initial
        self.label = label
        self.stream = stream or sys.stderr
        self.enabled = enabled and self.stream.isatty()
        self.started = time.perf_counter()
        self._last_width = 0
        self._closed = False

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def update(
        self,
        completed: int,
        *,
        episodes: int | None = None,
        speed: float | None = None,
        evaluation: float | None = None,
    ) -> None:
        if not self.enabled or self._closed:
            return
        completed = min(max(completed, self.initial), self.total)
        run_completed = completed - self.initial
        run_total = max(self.total - self.initial, 1)
        fraction = run_completed / run_total
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (run_total - run_completed) / run_completed if run_completed else 0.0
        filled = min(20, round(20 * fraction))
        fields = [
            f"{self.label} {completed}/{self.total}",
            f"[{'#' * filled}{'-' * (20 - filled)}]",
            f"{fraction:6.1%}",
            f"elapsed {self._duration(elapsed)}",
            f"ETA {self._duration(eta) if run_completed else '--:--'}",
        ]
        if episodes is not None:
            fields.append(f"episodes {episodes:,}")
        if speed is not None:
            fields.append(f"{speed:,.0f} trans/s")
        if evaluation is not None:
            fields.append(f"eval {evaluation:.1%}")
        line = "  ".join(fields)
        padding = " " * max(0, self._last_width - len(line))
        self.stream.write(f"\r{line}{padding}")
        self.stream.flush()
        self._last_width = len(line)

    def close(self) -> None:
        if self.enabled and not self._closed:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    checkpoint: Path,
    algorithm: str,
    config: dict[str, Any],
    started_at: float,
    summary: dict[str, Any],
) -> Path:
    path = checkpoint.with_suffix(".manifest.json")
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    payload = {
        "format_version": 1,
        **source_identity(),
        "ruleset_version": rules.RULESET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "algorithm": algorithm,
        "config": config,
        "summary": summary,
        "command": sys.argv,
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "duration_seconds": time.time() - started_at,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def prune_snapshots(directory: Path, prefix: str, keep: int) -> list[Path]:
    """Remove old numbered snapshots while preserving separately named best/latest files."""
    if keep < 1:
        raise ValueError("keep must be positive")
    snapshots = sorted(directory.glob(f"{prefix}.step-*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    removed = snapshots[keep:]
    for path in removed:
        path.unlink(missing_ok=True)
    return removed
