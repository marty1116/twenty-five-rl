"""Small, explicit training reward layer separate from authoritative rules."""

from __future__ import annotations

from dataclasses import dataclass

from . import rules
from .engine import Transition

REWARD_SCHEMES = ("terminal", "score_difference", "terminal_plus_score")


@dataclass(frozen=True)
class RewardSpec:
    scheme: str = "terminal"
    score_scale: float = 1.0 / rules.WIN_SCORE

    def event_rewards(self, transition: Transition, num_players: int) -> list[float]:
        if num_players < 2:
            raise ValueError("rewards require at least two players")
        rewards = [0.0] * num_players
        if self.scheme in {"score_difference", "terminal_plus_score"} and transition.points_awarded:
            scaled = [points * self.score_scale for points in transition.points_awarded]
            mean = sum(scaled) / num_players
            rewards = [value - mean for value in scaled]
        if transition.game_winner is not None and self.scheme in {"terminal", "terminal_plus_score"}:
            terminal = [-1.0 / (num_players - 1)] * num_players
            terminal[transition.game_winner] = 1.0
            rewards = [dense + outcome for dense, outcome in zip(rewards, terminal)]
        return rewards


def reward_spec(name: str) -> RewardSpec:
    if name not in REWARD_SCHEMES:
        raise ValueError(f"reward must be one of {', '.join(REWARD_SCHEMES)}")
    return RewardSpec(name)
