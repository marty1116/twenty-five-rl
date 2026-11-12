"""Generic masked tabular Monte Carlo and temporal-difference control."""

from __future__ import annotations

import ast
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Hashable

from .. import rules
from ..observations import OBSERVATION_SCHEMA_VERSION


def discounted_returns(rewards: list[float], gamma: float, bootstrap: float = 0.0) -> list[float]:
    out = [0.0] * len(rewards)
    running = bootstrap
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


class TabularControl:
    def __init__(self, alpha: float = 0.1, gamma: float = 1.0, seed: int = 0):
        self.alpha, self.gamma = alpha, gamma
        self.q: dict[Hashable, dict[int, float]] = defaultdict(dict)
        self.counts: dict[tuple[Hashable, int], int] = defaultdict(int)
        self.rng = random.Random(seed)

    def value(self, state: Hashable, action: int) -> float:
        return self.q[state].get(action, 0.0)

    def act(self, state: Hashable, legal: list[int], epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)
        best = max(self.value(state, a) for a in legal)
        return self.rng.choice([a for a in legal if self.value(state, a) == best])

    def monte_carlo_update(
        self,
        states: list[Hashable],
        actions: list[int],
        rewards: list[float],
        first_visit: bool = True,
        bootstrap: float = 0.0,
    ) -> None:
        returns = discounted_returns(rewards, self.gamma, bootstrap)
        visited: set[tuple[Hashable, int]] = set()
        for state, action, ret in zip(states, actions, returns):
            key = (state, action)
            if first_visit and key in visited:
                continue
            visited.add(key)
            self.counts[key] += 1
            old = self.value(state, action)
            self.q[state][action] = old + (ret - old) / self.counts[key]

    def td_update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable | None,
        next_legal: list[int],
        method: str,
        epsilon: float = 0.0,
        next_action: int | None = None,
    ) -> None:
        if next_state is None:
            target = reward
        elif method == "sarsa":
            if next_action is None:
                raise ValueError("SARSA requires next_action")
            target = reward + self.gamma * self.value(next_state, next_action)
        elif method == "expected_sarsa":
            n = len(next_legal)
            best = max(self.value(next_state, action) for action in next_legal)
            greedy = [action for action in next_legal if self.value(next_state, action) == best]
            expected = sum(
                (epsilon / n + ((1 - epsilon) / len(greedy) if action in greedy else 0.0))
                * self.value(next_state, action)
                for action in next_legal
            )
            target = reward + self.gamma * expected
        elif method == "q_learning":
            target = reward + self.gamma * max(self.value(next_state, a) for a in next_legal)
        else:
            raise ValueError("method must be sarsa, q_learning, or expected_sarsa")
        self.q[state][action] = self.value(state, action) + self.alpha * (target - self.value(state, action))

    def save(self, path: str | Path, *, episodes: int = 0, metadata: dict | None = None, training_state: dict | None = None) -> None:
        payload = {
            "format_version": 2,
            "ruleset_version": rules.RULESET_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "episodes": episodes,
            "rng_state": repr(self.rng.getstate()),
            "q": {self._key(state): {str(action): value for action, value in row.items()} for state, row in self.q.items()},
            "counts": {f"{self._key(state)}|{action}": count for (state, action), count in self.counts.items()},
            'metadata': metadata or {},
            'training_state': training_state or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def _key(state: Hashable) -> str:
        return json.dumps(list(state) if isinstance(state, tuple) else state, separators=(",", ":"))

    @staticmethod
    def _parse_key(value: str) -> Hashable:
        decoded = json.loads(value)
        return tuple(decoded) if isinstance(decoded, list) else decoded

    @classmethod
    def load(cls, path: str | Path) -> "TabularControl":
        data = json.loads(Path(path).read_text())
        if data.get("format_version") != 2:
            raise ValueError("unsupported tabular checkpoint format")
        if data.get("ruleset_version") != rules.RULESET_VERSION:
            raise ValueError("tabular checkpoint was trained under an incompatible or unversioned ruleset")
        if data.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("tabular checkpoint was trained under an incompatible or unversioned observation schema")
        obj = cls(data["alpha"], data["gamma"])
        for state, row in data["q"].items():
            key = cls._parse_key(state)
            obj.q[key] = {int(a): float(v) for a, v in row.items()}
        for raw, count in data.get("counts", {}).items():
            state, action = raw.rsplit("|", 1)
            key = cls._parse_key(state)
            obj.counts[(key, int(action))] = int(count)
        if "rng_state" in data:
            obj.rng.setstate(ast.literal_eval(data["rng_state"]))
        obj.episodes = int(data.get("episodes", 0))
        obj.metadata = data.get('metadata', {})
        obj.training_state = data.get('training_state', {})
        return obj
