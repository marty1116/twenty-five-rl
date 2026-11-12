"""Experimental duplicate-deal evaluation and Markdown/CSV/JSON reports."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from numbers import Integral
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .cards import Card
from .engine import Phase, TwentyFiveGame
from .policies import Policy, PolicyView, make_policy
from .valuation import discard_value


@dataclass
class PolicyStats:
    games: int = 0
    wins: int = 0
    final_scores: list[float] = field(default_factory=list)
    differentials: list[float] = field(default_factory=list)
    tricks: list[float] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    by_seat: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    by_dealer: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    by_players: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    tricks_by_trump: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    by_deal: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    rob_opportunities: int = 0
    rob_matches: int = 0
    rob_match_wins: int = 0
    rob_discards: list[dict[str, float | int]] = field(default_factory=list)
    invalid_actions: int = 0
    mask_disagreements: int = 0
    completed_games: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)


class PolicyFailure(RuntimeError):
    def __init__(self, policy, seat, action, kind, message):
        super().__init__(message)
        self.details = {'policy': policy, 'seat': seat, 'action': repr(action), 'kind': kind, 'message': message}


def _mean(values):
    return statistics.fmean(values) if values else None


def _fmt(value, spec='.3f'):
    return '-' if value is None else format(value, spec)


EVALUATION_SUITES: dict[str, tuple[str, ...]] = {
    "smoke": ("random", "lowest"),
    "baseline": ("random", "greedy", "conservative", "counting"),
}


def _wilson(wins: int, games: int) -> tuple[float, float]:
    if not games:
        return 0.0, 0.0
    z = 1.96
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return center - half_width, center + half_width


def _cluster_bootstrap_interval(by_deal: dict[int, list[int]], seed: int = 1729) -> tuple[float, float]:
    """Bootstrap whole duplicate-deal clusters instead of treating rotations as independent."""
    clusters = list(by_deal.values())
    if len(clusters) < 2:
        values = [value for cluster in clusters for value in cluster]
        return _wilson(sum(values), len(values))
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(2_000):
        sample = [rng.choice(clusters) for _ in clusters]
        values = [value for cluster in sample for value in cluster]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    return estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates))]


def evaluation_schedule(games: int, seeds: list[int], num_players: int) -> list[tuple[int, int, int]]:
    """Return exactly ``games`` balanced (deal seed, target seat, dealer) entries."""
    if games < 1:
        raise ValueError("games must be positive")
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    rotations = num_players * num_players
    schedule: list[tuple[int, int, int]] = []
    for game_index in range(games):
        deal_index, rotation = divmod(game_index, rotations)
        base = seeds[deal_index % len(seeds)]
        generation = deal_index // len(seeds)
        deal_seed = base + generation * 1_000_003
        target_seat, dealer = divmod(rotation, num_players)
        schedule.append((deal_seed, target_seat, dealer))
    return schedule


def play_match(
    policies: list[Policy], seed: int, dealer: int = 0, *, validate: bool = False
) -> tuple[TwentyFiveGame, dict[str, object]]:
    game = TwentyFiveGame(len(policies), seed=seed, debug=validate, initial_dealer=dealer)
    rng = random.Random(seed ^ 0xA5A5)
    latency: dict[int, float] = defaultdict(float)
    calls: dict[int, int] = defaultdict(int)
    actions = 0
    mask_disagreements = 0
    rob_records: list[dict[str, float | int]] = []
    while not game.terminal:
        seat = game.current_player
        policy = policies[seat]
        view = PolicyView.from_game(game, seat)
        started = time.perf_counter()
        try:
            action = policy.act(view, rng)
        except Exception as exc:
            raise PolicyFailure(policy.name, seat, None, 'policy_exception', str(exc)) from exc
        latency[seat] += time.perf_counter() - started
        calls[seat] += 1
        mask = game.legal_action_mask(seat)
        mask_actions = {index for index, legal in enumerate(mask) if legal}
        card_actions = {card.index for card in game.legal_cards(seat)}
        mask_disagreements += int(mask_actions != card_actions)
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 52 or not mask[action]:
            raise PolicyFailure(policy.name, seat, action, 'invalid_action', f"policy {policy.name} returned illegal action {action}")
        if game.phase is Phase.ROB:
            rob_records.append(
                {
                    "seat": seat,
                    "card": action,
                    "estimated_value": discard_value(Card.from_index(action), list(view.hand), game.trump),
                }
            )
        game.step(seat, action)
        actions += 1
    return game, {
        "latency_ms": {str(seat): 1_000 * latency[seat] / max(1, calls[seat]) for seat in range(len(policies))},
        "actions": actions,
        "rob_records": rob_records,
        "mask_disagreements": mask_disagreements,
    }


def _rate_pairwise(matchups: dict[str, dict[str, list[int]]]) -> dict[str, float]:
    """Order-independent Bradley-Terry ratings, expressed on an Elo-like scale."""
    names = list(matchups)
    strengths = {name: 1.0 for name in names}
    for _ in range(200):
        updated: dict[str, float] = {}
        for name in names:
            wins = 0.0
            denominator = 0.0
            for opponent in names:
                if opponent == name:
                    continue
                observed_wins, games = matchups[name][opponent]
                if not games:
                    continue
                wins += observed_wins + 0.5
                adjusted_games = games + 1.0
                denominator += adjusted_games / (strengths[name] + strengths[opponent])
            updated[name] = wins / denominator if denominator else 1.0
        geometric_mean = math.exp(statistics.fmean(math.log(max(value, 1e-12)) for value in updated.values()))
        strengths = {name: value / geometric_mean for name, value in updated.items()}
    return {name: 1_500 + 400 * math.log10(max(value, 1e-12)) for name, value in strengths.items()}


def _homogeneous_matchup_matrix(
    named: dict[str, Policy], schedule: list[tuple[int, int, int]], num_players: int, failures=None,
) -> dict[str, dict[str, float | None]]:
    """Measure each focal policy against a table filled with one opponent policy."""
    matrix: dict[str, dict[str, float | None]] = {}
    for target_name, target in named.items():
        matrix[target_name] = {}
        for opponent_name, opponent in named.items():
            if target_name == opponent_name:
                matrix[target_name][opponent_name] = None
                continue
            wins = 0
            completed = 0
            for seed, target_seat, dealer in schedule:
                lineup = [opponent for _ in range(num_players)]
                lineup[target_seat] = target
                try:
                    game, _ = play_match(lineup, seed, dealer)
                except PolicyFailure as exc:
                    if failures is not None:
                        failures.append({'stage': 'homogeneous', 'seed': seed, 'dealer': dealer, **exc.details})
                    continue
                completed += 1
                wins += int(game.winner == target_seat)
            matrix[target_name][opponent_name] = wins / completed if completed else None
    return matrix


def evaluate(
    named: dict[str, Policy],
    games_per_policy: int,
    seeds: list[int],
    output_dir: Path,
    num_players: int = 4,
    training_metadata: dict[str, dict[str, object]] | None = None,
    matchup_games: int | None = None,
    benchmark_suite: str = "baseline",
) -> dict[str, object]:
    names = list(named)
    if not names:
        raise ValueError("at least one policy is required")
    if benchmark_suite not in EVALUATION_SUITES:
        raise ValueError(f"unknown benchmark suite {benchmark_suite!r}; choose {sorted(EVALUATION_SUITES)}")
    suite_names = EVALUATION_SUITES[benchmark_suite]
    suite_policies = {name: make_policy(name) for name in suite_names}
    schedule = evaluation_schedule(games_per_policy, seeds, num_players)
    used_seeds = list(dict.fromkeys(seed for seed, _, _ in schedule))
    stats = {name: PolicyStats() for name in names}
    diagnostic_failures = []

    # Every target receives the same schedule and the same candidate-independent opponents.
    for target_name in names:
        for game_index, (seed, seat, dealer) in enumerate(schedule):
            lineup = [
                suite_policies[suite_names[(position + game_index) % len(suite_names)]]
                for position in range(num_players)
            ]
            lineup[seat] = named[target_name]
            stat = stats[target_name]
            stat.games += 1
            try:
                game, timing = play_match(lineup, seed, dealer)
            except PolicyFailure as exc:
                stat.failures.append({'seed': seed, 'dealer': dealer, **exc.details})
                stat.invalid_actions += exc.details['kind'] == 'invalid_action'
                stat.by_seat[seat].append(0)
                stat.by_dealer[dealer].append(0)
                stat.by_players[num_players].append(0)
                stat.by_deal[seed].append(0)
                continue
            winner = game.winner
            assert winner is not None
            won = int(seat == winner)
            stat.completed_games += 1
            stat.wins += won
            stat.final_scores.append(game.scores[seat])
            stat.differentials.append(game.scores[seat] - max(value for index, value in enumerate(game.scores) if index != seat))
            stat.rounds.append(len(game.match_history))
            stat.returns.append(1.0 if won else -1 / (num_players - 1))
            stat.lengths.append(int(timing["actions"]))
            stat.by_seat[seat].append(won)
            stat.by_dealer[dealer].append(won)
            stat.by_players[num_players].append(won)
            stat.by_deal[seed].append(won)
            total_tricks = sum(record["trick_winners"].count(seat) for record in game.match_history)
            stat.tricks.append(total_tricks / max(1, len(game.match_history)))
            stat.latencies_ms.append(timing["latency_ms"][str(seat)])  # type: ignore[index]
            stat.mask_disagreements += int(timing["mask_disagreements"])
            for record in game.match_history:
                winners = record["trick_winners"]
                stat.tricks_by_trump[record["trump"]].append(winners.count(seat) / max(1, len(winners)))
            rob_count = sum(record["robber"] == seat for record in game.match_history)
            stat.rob_opportunities += rob_count
            if rob_count:
                stat.rob_matches += 1
                stat.rob_match_wins += won
            stat.rob_discards.extend(
                {"card": row["card"], "estimated_value": row["estimated_value"]}
                for row in timing["rob_records"]  # type: ignore[union-attr]
                if row["seat"] == seat
            )

    head_to_head_schedule = evaluation_schedule(games_per_policy, seeds, 2)
    head_to_head_seeds = list(dict.fromkeys(seed for seed, _, _ in head_to_head_schedule))
    matchups = {first: {second: [0, 0] for second in names} for first in names}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            for seed, first_seat, dealer in head_to_head_schedule:
                lineup = [named[second], named[second]]
                lineup[first_seat] = named[first]
                try:
                    game, _ = play_match(lineup, seed, dealer)
                except PolicyFailure as exc:
                    diagnostic_failures.append({'stage': 'head_to_head', 'seed': seed, 'dealer': dealer, **exc.details})
                    continue
                first_won = int(game.winner == first_seat)
                matchups[first][second][0] += first_won
                matchups[first][second][1] += 1
                matchups[second][first][0] += 1 - first_won
                matchups[second][first][1] += 1
    pairwise = {
        first: {
            second: (matchups[first][second][0] / matchups[first][second][1] if matchups[first][second][1] else None)
            for second in names
        }
        for first in names
    }
    ratings = _rate_pairwise(matchups)

    cycles: list[tuple[str, str, str]] = []
    for first in names:
        for second in names:
            for third in names:
                if len({first, second, third}) != 3:
                    continue
                if all(value is not None and value > .5 for value in (pairwise[first][second], pairwise[second][third], pairwise[third][first])):
                    cycle = (first, second, third)
                    if not any(set(existing) == set(cycle) for existing in cycles):
                        cycles.append(cycle)

    multiplayer_games = matchup_games if matchup_games is not None else min(games_per_policy, 100)
    multiplayer_schedule = evaluation_schedule(multiplayer_games, seeds, num_players)
    multiplayer_matrix = _homogeneous_matchup_matrix(named, multiplayer_schedule, num_players, diagnostic_failures) if len(names) > 1 else {}

    rows: list[dict[str, object]] = []
    for name, stat in stats.items():
        wilson_low, wilson_high = _wilson(stat.wins, stat.games)
        ci_low, ci_high = _cluster_bootstrap_interval(stat.by_deal, seeds[0])
        metadata = (training_metadata or {}).get(name, {})
        rows.append(
            {
                "policy": name,
                "total_games": stat.games,
                'attempted_games': stat.games,
                'completed_games': stat.completed_games,
                'failed_games': len(stat.failures),
                'failures': stat.failures,
                'valid_for_comparison': not stat.failures,
                'win_rate_denominator': 'attempted games; failed trials are non-wins',
                'ci95_method': 'cluster bootstrap over deal seeds' if len(stat.by_deal) >= 2 else 'Wilson fallback (fewer than two deal clusters)',
                "wins": stat.wins,
                "win_rate": stat.wins / max(1, stat.games),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilson_ci95_low": wilson_low,
                "wilson_ci95_high": wilson_high,
                "average_final_score": _mean(stat.final_scores),
                "average_score_differential": _mean(stat.differentials),
                "average_tricks_per_round": _mean(stat.tricks),
                "average_rounds_per_match": _mean(stat.rounds),
                "win_rate_by_seat": {key: statistics.fmean(value) for key, value in stat.by_seat.items()},
                "win_rate_by_dealer": {key: statistics.fmean(value) for key, value in stat.by_dealer.items()},
                "win_rate_by_player_count": {key: statistics.fmean(value) for key, value in stat.by_players.items()},
                "average_trick_share_by_trump_suit": {
                    key: statistics.fmean(value) for key, value in stat.tricks_by_trump.items()
                },
                "two_player_win_rate_against": pairwise[name],
                "multiplayer_win_rate_against_homogeneous_table": multiplayer_matrix.get(name, {}),
                "rob_opportunities": stat.rob_opportunities,
                "rob_discard_choices": stat.rob_discards,
                "matches_with_rob": stat.rob_matches,
                "win_rate_in_matches_with_rob": stat.rob_match_wins / max(1, stat.rob_matches),
                "invalid_action_count": stat.invalid_actions,
                "illegal_mask_action_disagreement_count": stat.mask_disagreements,
                "mean_episodic_return": _mean(stat.returns),
                "return_variance": statistics.pvariance(stat.returns) if stat.returns else None,
                "mean_game_length": _mean(stat.lengths),
                "inference_latency_ms": _mean(stat.latencies_ms),
                "bradley_terry_rating": ratings[name],
                "training_steps": metadata.get("training_steps"),
                "training_wall_clock_seconds": metadata.get("training_wall_clock_seconds"),
                "model_parameter_count": metadata.get("model_parameter_count", 0),
                "checkpoint_identifier": metadata.get("checkpoint_identifier", "baseline"),
                "training_seed": metadata.get("training_seed"),
                'evaluation_mode': metadata.get('evaluation_mode', 'policy-defined'),
                'evaluation_epsilon': metadata.get('evaluation_epsilon'),
                "evaluation_seeds": used_seeds,
            }
        )

    output: dict[str, object] = {
        "requested_games_per_policy": games_per_policy,
        "benchmark_suite": benchmark_suite,
        "benchmark_opponents": list(suite_names),
        "actual_games_per_policy": {name: stats[name].completed_games for name in names},
        'attempted_games_per_policy': {name: stats[name].games for name in names},
        'diagnostic_failures': diagnostic_failures,
        'complete_rotation_blocks': games_per_policy % (num_players*num_players) == 0,
        "held_out_evaluation_seeds": used_seeds,
        "two_player_evaluation_seeds": head_to_head_seeds,
        "duplicate_deals": True,
        "seat_rotation": True,
        "dealer_rotation": True,
        "confidence_interval": "cluster bootstrap over deal seeds" if len(used_seeds) >= 2 else 'Wilson fallback (fewer than two deal clusters)',
        "rows": rows,
        "two_player_head_to_head_matrix": pairwise,
        "multiplayer_homogeneous_table_matrix": multiplayer_matrix,
        "multiplayer_matchup_games_per_cell": multiplayer_games,
        "cyclic_matchups": cycles,
        "note": "Main win rates use a fixed candidate-independent suite; candidate matchups are separate diagnostics.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.json").write_text(json.dumps(output, indent=2))
    with (output_dir / "leaderboard.csv").open("w", newline="") as stream:
        simple = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
        writer = csv.DictWriter(stream, fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)

    markdown = [
        "# Twenty-Five evaluation leaderboard",
        "",
        f"> Held-out duplicate-deal evaluation against the fixed `{benchmark_suite}` suite: "
        + ", ".join(suite_names) + ".",
        f"> 95% intervals: {output['confidence_interval']}. Candidate matchups are separate diagnostics.",
        '> Win rates divide by attempted trials; failures count as non-wins and invalidate comparison eligibility. Score/return metrics use completed matches only.',
        "",
        "| Policy | Games | Wins | Win rate (95% CI) | Avg score | Avg diff | Tricks/round | 2P BT rating | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: value["win_rate"], reverse=True):
        markdown.append(
            f"| {row['policy']} | {row['total_games']} | {row['wins']} | {row['win_rate']:.3f} "
            f"({row['ci95_low']:.3f}-{row['ci95_high']:.3f}) | {_fmt(row['average_final_score'], '.2f')} | "
            f"{_fmt(row['average_score_differential'], '.2f')} | {_fmt(row['average_tricks_per_round'], '.2f')} | "
            f"{row['bradley_terry_rating']:.0f} | {_fmt(row['inference_latency_ms'])} |"
        )
    markdown += [
        "",
        f"## {num_players}-player focal policy versus homogeneous opponent table",
        "",
        "| Policy | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for first in names:
        markdown.append(
            "| " + first + " | " + " | ".join("-" if first == second else _fmt(multiplayer_matrix[first][second]) for second in names) + " |"
        )
    markdown += [
        "",
        "## Two-player head-to-head matrix",
        "",
        "| Policy | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for first in names:
        markdown.append(
            "| " + first + " | " + " | ".join("-" if first == second else _fmt(pairwise[first][second]) for second in names) + " |"
        )
    markdown += [
        "",
        "## Diagnostics",
        "",
        f"- Requested attempts per policy: {games_per_policy}; completed: {output['actual_games_per_policy']}",
        f"- Multiplayer matchup games per matrix cell: {multiplayer_games}",
        f"- Cyclic two-player matchups: {cycles or 'none detected'}",
        f"- Evaluation deal seeds: {used_seeds}",
        f"- Failed main trials: { {name: len(stat.failures) for name,stat in stats.items()} }; diagnostic failures: {len(diagnostic_failures)}",
    ]
    (output_dir / "leaderboard.md").write_text("\n".join(markdown) + "\n")
    return output
