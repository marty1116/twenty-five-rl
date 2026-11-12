"""Small, balanced held-out evaluations used during training."""

from __future__ import annotations

from pathlib import Path

from ..evaluation import evaluation_schedule, play_match
from ..policies import FrozenCheckpointPolicy, Policy, make_policy


def policy_win_rate(
    learned: Policy,
    players: int,
    seed: int,
    games: int = 16,
    opponent: str = "random",
) -> float:
    opponent_policy = make_policy(opponent)
    wins = 0
    for deal_seed, seat, dealer in evaluation_schedule(games, [900_000 + seed], players):
        lineup = [opponent_policy for _ in range(players)]
        lineup[seat] = learned
        game, _ = play_match(lineup, deal_seed, dealer)
        wins += int(game.winner == seat)
    return wins / games


def checkpoint_win_rate(
    path: Path,
    players: int,
    seed: int,
    games: int = 16,
    opponent: str = "random",
) -> float:
    return policy_win_rate(FrozenCheckpointPolicy(path), players, seed, games, opponent)
