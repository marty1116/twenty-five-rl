"""Experimental frozen checkpoint registry and matchup-cycle utilities."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .policies import FrozenCheckpointPolicy, Policy


@dataclass(frozen=True)
class CheckpointRecord:
    identifier: str
    filename: str
    algorithm: str
    training_steps: int
    seed: int
    sha256: str


class CheckpointRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"

    def _records(self) -> list[dict[str, object]]:
        if not self.index.exists():
            return []
        data = json.loads(self.index.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("checkpoint registry must contain a JSON list")
        return data

    def register(self, source: Path, algorithm: str, training_steps: int, seed: int) -> CheckpointRecord:
        if not source.is_file():
            raise FileNotFoundError(source)
        if training_steps < 0:
            raise ValueError("training_steps must be non-negative")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        identifier = f"{algorithm}-{training_steps}-{digest[:8]}"
        destination = self.root / f"{identifier}{source.suffix}"
        shutil.copy2(source, destination)
        record = CheckpointRecord(identifier, destination.name, algorithm, training_steps, seed, digest)
        records = [row for row in self._records() if row.get("identifier") != identifier]
        records.append(asdict(record))
        temporary = self.index.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.index)
        return record

    def load_pool(self, weights: list[float] | None = None) -> "FrozenPool":
        records = self._records()
        policies = [FrozenCheckpointPolicy(self.root / str(record["filename"])) for record in records]
        return FrozenPool(policies, weights)


class FrozenPool:
    def __init__(self, policies: list[Policy], weights: list[float] | None = None):
        if not policies:
            raise ValueError("a frozen pool requires at least one policy")
        if weights is not None and len(weights) != len(policies):
            raise ValueError("weights must match policies")
        self.policies = tuple(policies)
        self.weights = None if weights is None else tuple(weights)

    def sample(self, rng: random.Random) -> Policy:
        return rng.choices(self.policies, weights=self.weights, k=1)[0]


def find_cycles(matrix: dict[str, dict[str, float]]) -> list[tuple[str, str, str]]:
    names = list(matrix)
    cycles: list[tuple[str, str, str]] = []
    for first in names:
        for second in names:
            for third in names:
                candidate = (first, second, third)
                if len(set(candidate)) != 3:
                    continue
                if (
                    matrix[first][second] > 0.5
                    and matrix[second][third] > 0.5
                    and matrix[third][first] > 0.5
                    and not any(set(existing) == set(candidate) for existing in cycles)
                ):
                    cycles.append(candidate)
    return cycles

