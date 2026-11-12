"""Run a reproducible DQN or PPO experiment from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .dqn import train as train_dqn
from .ppo import train as train_ppo


def _paths(config: dict[str, Any]) -> dict[str, Any]:
    converted = dict(config)
    for key in ("output", "resume", "tensorboard_dir"):
        if converted.get(key) is not None:
            converted[key] = Path(converted[key])
    if "opponent_checkpoints" in converted:
        converted["opponent_checkpoints"] = [Path(value) for value in converted["opponent_checkpoints"]]
    return converted


def run_config(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text())
    algorithm = document.get("algorithm")
    trainers: dict[str, Callable[..., dict[str, object]]] = {"dqn": train_dqn, "ppo": train_ppo}  # type: ignore[dict-item]
    if algorithm not in trainers:
        raise ValueError("config algorithm must be 'dqn' or 'ppo'")
    train_config = document.get("train")
    if not isinstance(train_config, dict):
        raise ValueError("config requires a 'train' object")
    return trainers[algorithm](**_paths(train_config))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_config(args.config), indent=2))


if __name__ == "__main__":
    main()
