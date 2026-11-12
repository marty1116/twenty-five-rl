"""Research provenance and independent seat/dealer coverage."""

from __future__ import annotations

import hashlib
from pathlib import Path


def training_positions(episode: int, players: int) -> tuple[int, int]:
    """Cover every (learner, initial dealer) pair once per N² episodes."""
    if episode < 0 or not 2 <= players <= 10:
        raise ValueError("invalid episode index or player count")
    return episode % players, (episode // players) % players


def source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        source.update(path.relative_to(root).as_posix().encode())
        source.update(b"\0")
        source.update(path.read_bytes().replace(b"\r\n", b"\n"))
    rules_digest = hashlib.sha256()
    for name in ("cards.py", "rules.py", "engine.py"):
        rules_digest.update(name.encode())
        rules_digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return {"source_sha256": source.hexdigest(), "rules_sha256": rules_digest.hexdigest()}
