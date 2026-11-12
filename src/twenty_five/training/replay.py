"""Memory-efficient prioritized replay storage for neural value learners."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_masks: np.ndarray
    discounts: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.states = np.empty((capacity, observation_size), dtype=np.uint8)
        self.next_states = np.empty((capacity, observation_size), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.next_masks = np.empty((capacity, 52), dtype=np.bool_)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
        discount: float,
    ) -> None:
        index = self.position
        self.states[index] = np.rint(np.clip(state, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = np.rint(np.clip(next_state, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.dones[index] = done
        self.next_masks[index] = next_mask
        self.discounts[index] = discount
        self.priorities[index] = float(self.priorities[: self.size].max()) if self.size else 1.0
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> ReplayBatch:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("not enough replay entries")
        if not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0 or not 0 <= beta <= 1:
            raise ValueError('PER requires finite alpha >= 0 and beta in [0, 1]')
        scaled = np.power(self.priorities[: self.size] + 1e-6, alpha)
        probabilities = scaled / scaled.sum()
        indices = rng.choice(self.size, size=batch_size, replace=True, p=probabilities)
        weights = np.power(self.size * probabilities[indices], -beta)
        # A buffer-wide normalization is independent of the sampled batch.
        weights /= np.power(self.size * probabilities.min(), -beta)
        return ReplayBatch(
            self.states[indices].astype(np.float32) / 255.0,
            self.actions[indices].astype(np.int64),
            self.rewards[indices],
            self.next_states[indices].astype(np.float32) / 255.0,
            self.dones[indices],
            self.next_masks[indices],
            self.discounts[indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        indices, errors = np.asarray(indices), np.asarray(errors)
        if indices.ndim != 1 or errors.shape != indices.shape or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('priority indices and errors must be equal-length vectors with integer indices')
        if not np.isfinite(errors).all() or (indices < 0).any() or (indices >= self.size).any():
            raise ValueError('priority errors must be finite and indices must identify live records')
        unique, inverse = np.unique(indices, return_inverse=True)
        maximum = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(maximum, inverse, np.abs(errors).astype(np.float32))
        self.priorities[unique] = maximum + 1e-5

    def save(self, path: Path, *, metadata: dict[str, object]) -> None:
        """Persist valid entries compactly; uint8 sparse observations compress well."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.asarray(1, dtype=np.int16),
            capacity=np.asarray(self.capacity, dtype=np.int64),
            position=np.asarray(self.position, dtype=np.int64),
            size=np.asarray(self.size, dtype=np.int64),
            states=self.states[: self.size],
            next_states=self.next_states[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            dones=self.dones[: self.size],
            next_masks=self.next_masks[: self.size],
            discounts=self.discounts[: self.size],
            priorities=self.priorities[: self.size],
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> tuple["ReplayBuffer", dict[str, object]]:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["format_version"]) != 1:
                raise ValueError("unsupported replay sidecar format")
            states = payload["states"]
            capacity = int(payload["capacity"])
            replay = cls(capacity, int(states.shape[1]))
            size = int(payload["size"])
            if size != len(states) or not 0 <= size <= capacity:
                raise ValueError("replay sidecar has inconsistent size")
            replay.size = size
            replay.position = int(payload["position"])
            if not 0 <= replay.position < capacity:
                raise ValueError("replay sidecar has invalid position")
            for name in (
                "states", "next_states", "actions", "rewards", "dones",
                "next_masks", "discounts", "priorities",
            ):
                getattr(replay, name)[:size] = payload[name]
            metadata = json.loads(str(payload["metadata"]))
        return replay, metadata
