"""Canonical PettingZoo AEC environment for multiplayer Twenty-Five."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ..engine import TwentyFiveGame
from ..observations import NEURAL_OBSERVATION_SIZE, PlayerView
from ..rewards import RewardSpec, reward_spec
from ._compat import PettingZooAECEnv, spaces
from .events import INFO_SCHEMA_VERSION, player_event


class TwentyFiveAECEnv(PettingZooAECEnv):
    """One action per agent turn with PettingZoo-compatible reward accounting."""

    metadata = {"name": "twenty_five_v3", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 4,
        reward: str = "terminal",
        random_dealer: bool = False,
        validate: bool = False,
        render_mode: str | None = None,
        illegal_action_mode: str = "truncate",
    ):
        if illegal_action_mode not in {"truncate", "raise"}:
            raise ValueError("illegal_action_mode must be 'truncate' or 'raise'")
        self.num_players = num_players
        self.possible_agents = [f"player_{seat}" for seat in range(num_players)]
        self.agent_name_mapping = {agent: seat for seat, agent in enumerate(self.possible_agents)}
        self.reward_spec: RewardSpec = reward_spec(reward)
        self.random_dealer = random_dealer
        self.validate = validate
        self.render_mode = render_mode
        self.illegal_action_mode = illegal_action_mode
        self.rng = random.Random()
        self.game: TwentyFiveGame | None = None
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection = self.possible_agents[0]
        self._action_spaces = {agent: spaces.Discrete(52) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(0.0, 1.0, (NEURAL_OBSERVATION_SIZE,), np.float32),
                    "action_mask": spaces.Box(0, 1, (52,), np.int8),
                }
            )
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str):
        if agent not in self.agent_name_mapping:
            raise KeyError(agent)
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        if agent not in self.agent_name_mapping:
            raise KeyError(agent)
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self.rng.seed(seed)
        for agent in self.possible_agents:
            self.action_space(agent).seed(seed)
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = dict(self.rewards)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {'info_schema_version': INFO_SCHEMA_VERSION} for agent in self.agents}
        dealer = None
        if options and "dealer" in options:
            dealer = int(options["dealer"])
        elif self.random_dealer:
            dealer = self.rng.randrange(self.num_players)
        self.game = TwentyFiveGame(
            self.num_players,
            seed=seed,
            debug=self.validate,
            initial_dealer=0 if dealer is None else dealer,
            initial_deck=options.get("initial_deck") if options else None,
        )
        self.agent_selection = self.possible_agents[self.game.current_player]

    def raw_view(self, agent: str) -> PlayerView:
        if self.game is None:
            raise RuntimeError("reset must be called before observe")
        return PlayerView.from_game(self.game, self.agent_name_mapping[agent])

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        view = self.raw_view(agent)
        mask = view.action_mask
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            mask = np.zeros(52, dtype=np.int8)
        return {"observation": view.neural, "action_mask": mask}

    def last(self, observe: bool = True):
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("active agents require a card action")
        if self.game is None:
            raise RuntimeError("reset must be called before step")
        actor = self.agent_name_mapping[agent]
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < 52 or not self.game.legal_action_mask(actor)[int(action)]:
            if self.illegal_action_mode == "raise":
                raise ValueError(f"illegal action {action}; episode state was not changed")
            self._cumulative_rewards[agent] = 0.0
            self._clear_rewards()
            for name in self.possible_agents:
                self.truncations[name] = True
                self.rewards[name] = 1.0 / (self.num_players - 1)
                self.infos[name] = {'info_schema_version': INFO_SCHEMA_VERSION, "illegal_action_by": agent, "scores": tuple(self.game.scores)}
            self.rewards[agent] = -1.0
            self._deads_step_first()
            self._accumulate_rewards()
            return
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        transition = self.game.step(actor, int(action))
        delivered = self.reward_spec.event_rewards(transition, self.num_players)
        for seat, name in enumerate(self.possible_agents):
            self.rewards[name] = delivered[seat]
            self.infos[name] = {'info_schema_version': INFO_SCHEMA_VERSION, "transition": player_event(transition, seat), "scores": tuple(self.game.scores)}
        if self.game.terminal:
            for name in self.possible_agents:
                self.terminations[name] = True
            self._deads_step_first()
        else:
            self.agent_selection = self.possible_agents[self.game.current_player]
        self._accumulate_rewards()

    def agent_iter(self, max_iter: int | None = None):
        count = 0
        while self.agents and (max_iter is None or count < max_iter):
            yield self.agent_selection
            count += 1

    def render(self) -> str:
        if self.game is None:
            raise RuntimeError("reset must be called before render")
        return str(self.game.public_state())

    def close(self) -> None:
        return None
