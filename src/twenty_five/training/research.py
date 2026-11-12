"""Plan or execute independent research runs; no training without --execute."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

from ..evaluation import evaluate
from ..policies import FrozenCheckpointPolicy, make_policy
from .protocol import source_identity


def trainer(algorithm):
    from . import dqn, policy_gradient, ppo, tabular_twentyfive

    if algorithm in {"mc", "sarsa", "q_learning", "expected_sarsa"}:
        return tabular_twentyfive.train, {"algorithm": algorithm}, ".json"
    if algorithm in {"reinforce", "reinforce_baseline", "n_step_actor_critic"}:
        return policy_gradient.train, {"variant": algorithm}, ".pt"
    if algorithm == "dqn":
        return dqn.train, {}, ".pt"
    if algorithm == "ppo":
        return ppo.train, {}, ".pt"
    raise ValueError(f"unknown algorithm: {algorithm}")


def plan(document):
    jobs = []
    root = Path(document["output_root"])
    if not document["seeds"] or len(set(document["seeds"])) != len(document["seeds"]):
        raise ValueError("training seeds must be nonempty and unique")
    names = set()
    for experiment in document["experiments"]:
        name = experiment["name"]
        if name in names or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("experiment names must be unique lowercase safe directory names")
        names.add(name)
        function, extra, suffix = trainer(experiment["algorithm"])
        for seed in document["seeds"]:
            if not isinstance(seed, int) or not 0 <= seed < 100_000:
                raise ValueError("training seeds must be integers in [0, 100000)")
            directory = root / name / f"seed-{seed}"
            kwargs = {**document["common"], **experiment["train"], **extra,
                      "seed": seed, "output": directory / ("model" + suffix)}
            if kwargs.get("resume"):
                raise ValueError("research runs must start independently; resume is unsupported")
            inspect.signature(function).bind(**kwargs)
            jobs.append((name, seed, directory, function, kwargs))
    if not jobs:
        raise ValueError("at least one experiment is required")
    return jobs


def evaluate_run(document, directory, checkpoint, split):
    settings = document["evaluation"]
    games = settings["games"]
    players = document["common"]["players"]
    if games < players * players or games % (players * players):
        raise ValueError("evaluation games must be a positive multiple of players squared")
    # Widely separated negative namespaces cannot overlap these positive training seeds.
    seeds = {"validation": [-10**12], "test": [-2 * 10**12]}[split]
    candidates = {name: make_policy(name) for name in settings["baselines"]}
    metadata = {}
    for mode in ('greedy','sampled'):
        policy = FrozenCheckpointPolicy(checkpoint, mode=mode, epsilon=settings.get('epsilon', .05))
        name = f'learned_{mode}'
        candidates[name] = policy
        metadata[name] = policy.training_metadata
    result = evaluate(candidates, games, seeds, directory / split, players,
                      training_metadata=metadata,
                      matchup_games=settings["matchup_games"], benchmark_suite="baseline")
    return {"policy_rows": result["rows"], "split": split, "games": games}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    document = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = plan(document)
    if args.split == "test" and not args.evaluate_only:
        parser.error("final test requires --evaluate-only after model selection")
    print(json.dumps([{"experiment": n, "seed": s, "directory": str(d),
                       "train": k} for n, s, d, _, k in jobs], indent=2, default=str))
    if not args.execute:
        return
    # Validate every target before beginning any training.
    for _, _, directory, _, kwargs in jobs:
        if args.evaluate_only:
            record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            if record["identity"] != source_identity():
                raise ValueError(f"source changed since training: {directory}")
            if not kwargs["output"].exists() or (directory / args.split).exists():
                raise ValueError(f"missing checkpoint or existing evaluation: {directory}")
            if record.get('checkpoint_sha256') != hashlib.sha256(kwargs['output'].read_bytes()).hexdigest():
                raise ValueError(f"checkpoint digest changed since training: {directory}")
        elif directory.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {directory}")
    for name, seed, directory, function, kwargs in jobs:
        if not args.evaluate_only:
            directory.mkdir(parents=True)
            record = {"experiment": name, "seed": seed, "identity": source_identity(),
                      "config": document, "train": kwargs, "python": platform.python_version(),
                      "status": "running"}
            packages = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                      capture_output=True, text=True, check=True)
            (directory / "requirements.txt").write_text(packages.stdout, encoding="utf-8")
            record_path = directory / "run.json"
            record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            try:
                record["summary"] = function(**kwargs)
                record["status"] = "trained"
                record['checkpoint_sha256'] = hashlib.sha256(kwargs['output'].read_bytes()).hexdigest()
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = repr(exc)
                raise
            finally:
                record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        evaluate_run(document, directory, kwargs["output"], args.split)


if __name__ == "__main__":
    main()
