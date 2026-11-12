"""Gymnasium adapter exposing one controlled seat over the canonical AEC game."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ..observations import NEURAL_OBSERVATION_SIZE
from ..policies import Policy, RandomPolicy
from ._compat import GymBase, HAS_GYMNASIUM, spaces
from .aec_env import TwentyFiveAECEnv
from .events import INFO_SCHEMA_VERSION


class TwentyFiveEnv(GymBase):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        num_players: int = 4,
        controlled_seat: int | None = 0,
        opponents: dict[int, Policy] | None = None,
        reward: str = "terminal",
        validate: bool = False,
        render_mode: str | None = None,
        illegal_action_mode: str = "truncate",
    ):
        if illegal_action_mode not in {"truncate", "raise", "penalize"}:
            raise ValueError("illegal_action_mode must be 'truncate', 'raise', or 'penalize'")
        self.num_players = num_players
        self.controlled_seat = controlled_seat
        self._randomize_seat = controlled_seat is None
        self.opponents = opponents or {}
        self.validate = validate
        self.render_mode = render_mode
        self.illegal_action_mode = illegal_action_mode
        self.action_space = spaces.Discrete(52)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(0.0, 1.0, (NEURAL_OBSERVATION_SIZE,), np.float32),
                "action_mask": spaces.Box(0, 1, (52,), np.int8),
            }
        )
        self.aec = TwentyFiveAECEnv(num_players, reward, validate=validate, render_mode=render_mode)
        self.rng = random.Random()
        self.invalid_actions = 0
        self._truncated = False

    @property
    def game(self):
        return self.aec.game

    def _agent(self) -> str:
        if self.controlled_seat is None:
            raise RuntimeError("controlled seat has not been selected")
        return self.aec.possible_agents[self.controlled_seat]

    def _obs(self) -> dict[str, np.ndarray]:
        observation = self.aec.observe(self._agent())
        if self._truncated:
            observation['action_mask'] = np.zeros(52, dtype=np.int8)
        return observation

    def _policy(self, seat: int) -> Policy:
        return self.opponents.get(seat, RandomPolicy())

    def _advance_opponents(self) -> tuple[float, list[dict[str, Any]]]:
        reward = 0.0
        events: list[dict[str, Any]] = []
        controlled = self._agent()
        while (self.aec.agents and not self.aec.terminations[controlled]
               and not self.aec.truncations[controlled] and self.aec.agent_selection != controlled):
            actor_name = self.aec.agent_selection
            seat = self.aec.agent_name_mapping[actor_name]
            action = self._policy(seat).act(self.aec.raw_view(actor_name), self.rng)
            self.aec.step(action)
            reward += self.aec.rewards[controlled]
            transition = self.aec.infos[controlled].get("transition")
            if transition is not None:
                events.append(dict(transition))
        self._truncated = bool(self.aec.truncations[controlled])
        return reward, events

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if HAS_GYMNASIUM:
            super().reset(seed=seed)
        self.rng.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        if self._randomize_seat:
            self.controlled_seat = self.rng.randrange(self.num_players)
        if self.controlled_seat is None or not 0 <= self.controlled_seat < self.num_players:
            raise ValueError("controlled_seat outside player range")
        dealer = int(options["dealer"]) if options and "dealer" in options else 0
        initial_deck = options.get("initial_deck") if options else None
        self.aec.reset(seed=seed, options={"dealer": dealer, "initial_deck": initial_deck})
        self.invalid_actions = 0
        self._truncated = False
        initial_reward, events = self._advance_opponents()
        return self._obs(), self._info(events, initial_reward)

    def step(self, action: int):
        if self.game is None:
            raise RuntimeError("reset must be called before step")
        if self._truncated:
            raise RuntimeError("reset must be called after truncation")
        if self.game.terminal:
            raise RuntimeError('reset must be called after termination')
        mask = self._obs()["action_mask"]
        if not self.action_space.contains(action) or not mask[int(action)]:
            self.invalid_actions += 1
            if self.illegal_action_mode == "raise":
                raise ValueError(f"illegal action {action}; episode state was not changed")
            if self.illegal_action_mode == "penalize":
                return self._obs(), -1.0, False, False, self._info([], -1.0)
            self._truncated = True
            return self._obs(), -1.0, False, True, self._info([], -1.0)
        controlled = self._agent()
        self.aec.step(int(action))
        reward = self.aec.rewards[controlled]
        transition = self.aec.infos[controlled].get("transition")
        events = [dict(transition)] if transition is not None else []
        other_reward, other_events = self._advance_opponents()
        reward += other_reward
        events.extend(other_events)
        terminated = bool(self.aec.terminations[controlled])
        return self._obs(), float(reward), terminated, self._truncated, self._info(events, reward)

    def _info(self, events: list[dict[str, Any]], reward: float) -> dict[str, Any]:
        if self.game is None:
            raise RuntimeError("reset must be called before info")
        return {
            'info_schema_version': INFO_SCHEMA_VERSION,
            'truncated': self._truncated,
            "player": self.controlled_seat,
            "phase": self.game.phase.value,
            "winner": self.game.winner,
            "scores": list(self.game.scores),
            "round": self.game.round_number,
            "events": events,
            "reward_delivered": reward,
            "invalid_action_count": self.invalid_actions,
            "legal_actions": list(np.flatnonzero(self._obs()["action_mask"])),
        }

    def render(self) -> str:
        return self.aec.render()

    def close(self) -> None:
        self.aec.close()
