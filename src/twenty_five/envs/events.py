"""Versioned, recipient-specific metadata for private-player interfaces."""
from dataclasses import asdict

from ..engine import Phase, Transition

INFO_SCHEMA_VERSION = 2


def player_event(transition: Transition, observer: int) -> dict[str, object]:
    event = asdict(transition)
    if transition.phase is Phase.ROB and observer != transition.actor:
        event['action'] = None
    return event
