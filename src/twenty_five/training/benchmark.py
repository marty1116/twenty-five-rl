"""Measure simulator throughput and batched neural inference on this machine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..agents.neural import LinearActorCritic, resolve_device
from ..evaluation import play_match
from ..observations import NEURAL_OBSERVATION_SIZE
from ..policies import make_policy


def benchmark(games: int, players: int, device: str, batch_sizes: list[int]) -> dict[str, object]:
    if games < 1 or any(size < 1 for size in batch_sizes):
        raise ValueError("games and batch sizes must be positive")
    policies = [make_policy("random") for _ in range(players)]
    started = time.perf_counter()
    actions = 0
    for index in range(games):
        _, timing = play_match(policies, seed=700_000 + index, dealer=index % players)
        actions += int(timing["actions"])
    simulator_seconds = time.perf_counter() - started

    compute = resolve_device(device)
    model = LinearActorCritic(NEURAL_OBSERVATION_SIZE, phase_heads=True).to(compute).eval()
    inference: dict[str, dict[str, float]] = {}
    for batch_size in batch_sizes:
        observations = torch.randn((batch_size, NEURAL_OBSERVATION_SIZE), device=compute)
        masks = torch.ones((batch_size, 52), dtype=torch.bool, device=compute)
        with torch.no_grad():
            for _ in range(10):
                model.policy_and_value(observations, masks)
            if compute.type == "cuda":
                torch.cuda.synchronize()
            repeats = max(20, 2_000 // batch_size)
            inference_started = time.perf_counter()
            for _ in range(repeats):
                model.policy_and_value(observations, masks)
            if compute.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - inference_started
        decisions = repeats * batch_size
        inference[str(batch_size)] = {
            "decisions_per_second": decisions / elapsed,
            "milliseconds_per_batch": 1_000 * elapsed / repeats,
        }

    return {
        "device": str(compute),
        "cuda_device": torch.cuda.get_device_name(0) if compute.type == "cuda" else None,
        "simulator": {
            "games": games,
            "players": players,
            "seconds": simulator_seconds,
            "games_per_second": games / simulator_seconds,
            "actions_per_second": actions / simulator_seconds,
        },
        "inference_by_batch_size": inference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.games, args.players, args.device, args.batch_sizes)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
