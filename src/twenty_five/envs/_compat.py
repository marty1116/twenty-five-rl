"""Optional framework imports with tiny fallbacks for engine-only installations."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GymBase = gym.Env
    HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover - used only by dependency-light installs
    class GymBase:
        def reset(self, *, seed=None, options=None):
            return None

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self, mask: np.ndarray | None = None) -> int:
            choices = np.flatnonzero(mask) if mask is not None else np.arange(self.n)
            return int(np.random.choice(choices))

        def contains(self, value: object) -> bool:
            return isinstance(value, (int, np.integer)) and 0 <= int(value) < self.n

        def seed(self, seed: int | None = None):
            np.random.seed(seed)

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

        def contains(self, value: object) -> bool:
            return isinstance(value, np.ndarray) and value.shape == self.shape

    class _Dict:
        def __init__(self, values: dict[str, object]):
            self.spaces = values

        def contains(self, value: object) -> bool:
            return isinstance(value, dict) and all(key in value for key in self.spaces)

        def seed(self, seed: int | None = None):
            return None

    class spaces:  # type: ignore[no-redef]
        Discrete, Box, Dict = _Discrete, _Box, _Dict

    HAS_GYMNASIUM = False

try:
    from pettingzoo import AECEnv as PettingZooAECEnv

    HAS_PETTINGZOO = True
except ImportError:  # pragma: no cover - used only by dependency-light installs
    class PettingZooAECEnv:
        def _clear_rewards(self) -> None:
            for agent in self.rewards:
                self.rewards[agent] = 0.0

        def _accumulate_rewards(self) -> None:
            for agent, reward in self.rewards.items():
                self._cumulative_rewards[agent] += reward

        def _deads_step_first(self) -> None:
            dead = [agent for agent in self.agents if self.terminations[agent] or self.truncations[agent]]
            if dead:
                self.agent_selection = dead[0]

        def _was_dead_step(self, action) -> None:
            if action is not None:
                raise ValueError("terminated agents must be stepped with None")
            dead = self.agent_selection
            self.agents.remove(dead)
            if self.agents:
                self._deads_step_first()

    HAS_PETTINGZOO = False
