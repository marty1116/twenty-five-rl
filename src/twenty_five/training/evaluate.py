"""Command-line entry point for reproducible policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation import EVALUATION_SUITES, evaluate
from ..policies import FrozenCheckpointPolicy, make_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["random", "lowest", "greedy", "conservative", "counting"])
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH; repeatable")
    parser.add_argument("--games", type=int, default=80, help="exact trials attempted for every leaderboard policy")
    parser.add_argument("--matchup-games", type=int, help="games per multiplayer matchup cell; defaults to min(games, 100)")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9001, 9002, 9003, 9004, 9005])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--suite", choices=sorted(EVALUATION_SUITES), default="baseline")
    parser.add_argument('--policy-mode', choices=['greedy','sampled','both'], default='both')
    parser.add_argument('--evaluation-epsilon', type=float, default=.05,
                        help='epsilon-soft evaluation for sampled tabular/Q checkpoints; actors use their categorical policy')
    args = parser.parse_args()

    named = {name: make_policy(name) for name in args.policies}
    metadata: dict[str, dict[str, object]] = {}
    for item in args.checkpoint:
        if "=" not in item:
            parser.error("--checkpoint must use NAME=PATH")
        name, path = item.split("=", 1)
        modes = ('greedy','sampled') if args.policy_mode == 'both' else (args.policy_mode,)
        for mode in modes:
            checkpoint = FrozenCheckpointPolicy(path, mode=mode, epsilon=args.evaluation_epsilon)
            key = f'{name}:{mode}'
            named[key] = checkpoint
            metadata[key] = checkpoint.training_metadata
    result = evaluate(
        named,
        args.games,
        args.seeds,
        args.output_dir,
        args.players,
        training_metadata=metadata,
        matchup_games=args.matchup_games,
        benchmark_suite=args.suite,
    )
    print(
        json.dumps(
            {
                "policies": len(result["rows"]),
                "games_per_policy": result["actual_games_per_policy"],
                "output": str(args.output_dir),
                "benchmark_suite": args.suite,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
