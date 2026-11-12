"""Versioned, player-safe observations derived from the stable game engine."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import rules
from .cards import Card, Suit
from .engine import Phase, TwentyFiveGame
from .valuation import intrinsic_strength

OBSERVATION_SCHEMA_VERSION = 1
MAX_PLAYERS = rules.MAX_PLAYERS
MAX_CURRENT_PLAYS = rules.MAX_PLAYERS
MAX_HISTORY_PLAYS = rules.TRICKS_PER_ROUND * rules.MAX_PLAYERS
CARD_ONE_HOT = 53
SLOT_SIZE = CARD_ONE_HOT + MAX_PLAYERS


@dataclass(frozen=True)
class ObservationLayout:
    """Named slices for the flat neural encoding; never duplicate offsets elsewhere."""

    hand: slice
    public_played: slice
    turned_card: slice
    own_rob_discard: slice
    trump: slice
    phase: slice
    player_mask: slice
    dealer: slice
    leader: slice
    current_player: slice
    robber: slice
    robbed: slice
    scores: slice
    tricks: slice
    current_slots: slice
    history_slots: slice
    size: int


def _build_layout() -> ObservationLayout:
    cursor = 0

    def take(width: int) -> slice:
        nonlocal cursor
        result = slice(cursor, cursor + width)
        cursor += width
        return result

    values = {
        "hand": take(52),
        "public_played": take(52),
        "turned_card": take(52),
        "own_rob_discard": take(52),
        "trump": take(4),
        "phase": take(len(Phase)),
        "player_mask": take(MAX_PLAYERS),
        "dealer": take(MAX_PLAYERS),
        "leader": take(MAX_PLAYERS),
        "current_player": take(MAX_PLAYERS),
        "robber": take(MAX_PLAYERS + 1),
        "robbed": take(1),
        "scores": take(MAX_PLAYERS),
        "tricks": take(MAX_PLAYERS),
        "current_slots": take(MAX_CURRENT_PLAYS * SLOT_SIZE),
        "history_slots": take(MAX_HISTORY_PLAYS * SLOT_SIZE),
    }
    return ObservationLayout(**values, size=cursor)


OBSERVATION_LAYOUT = _build_layout()
NEURAL_OBSERVATION_SIZE = OBSERVATION_LAYOUT.size


@dataclass(frozen=True)
class PlayerView:
    """All and only the information available to one player at a decision point."""

    player: int
    num_players: int
    hand: tuple[Card, ...]
    legal: tuple[Card, ...]
    trump: Suit
    phase: Phase
    round_number: int
    dealer: int
    leader: int
    current_player: int
    turned_card: Card
    scores: tuple[int, ...]
    round_tricks: tuple[int, ...]
    current_trick: tuple[tuple[int, Card], ...]
    trick_history: tuple[tuple[tuple[int, Card], ...], ...]
    robber: int | None
    robbed: bool
    winner: int | None
    own_rob_discard: Card | None

    @classmethod
    def from_game(cls, game: TwentyFiveGame, player: int) -> "PlayerView":
        if not 0 <= player < game.num_players:
            raise ValueError("player outside player range")
        own_discard = game.discards[0] if game.robber == player and game.discards else None
        return cls(
            player=player,
            num_players=game.num_players,
            hand=tuple(game.hands[player]),
            legal=tuple(game.legal_cards(player)),
            trump=game.trump,
            phase=game.phase,
            round_number=game.round_number,
            dealer=game.dealer,
            leader=game.leader,
            current_player=game.current_player,
            turned_card=game.turned_card,
            scores=tuple(game.scores),
            round_tricks=tuple(game.round_tricks),
            current_trick=tuple(game.current_trick),
            trick_history=tuple(tuple(trick) for trick in game.trick_history),
            robber=game.robber,
            robbed=game.rob_action is not None,
            winner=game.winner,
            own_rob_discard=own_discard,
        )

    @property
    def public(self) -> Mapping[str, Any]:
        """Read-only compatibility representation with no private discard."""
        return MappingProxyType(
            {
                "num_players": self.num_players,
                "phase": self.phase.value,
                "round": self.round_number,
                "dealer": self.dealer,
                "leader": self.leader,
                "current_player": self.current_player,
                "trump": int(self.trump),
                "turned_card": self.turned_card.index,
                "scores": list(self.scores),
                "round_tricks": list(self.round_tricks),
                "current_trick": [(seat, card.index) for seat, card in self.current_trick],
                "trick_history": [[(seat, card.index) for seat, card in trick] for trick in self.trick_history],
                "robber": self.robber,
                "robbed": self.robbed,
                "terminal": self.phase is Phase.TERMINAL,
                "winner": self.winner,
            }
        )

    @property
    def action_mask(self) -> np.ndarray:
        mask = np.zeros(52, dtype=np.int8)
        indices = [card.index for card in self.legal]
        if indices:
            mask[indices] = 1
        return mask

    @property
    def neural(self) -> np.ndarray:
        return encode_neural(self)

    @property
    def tabular(self) -> tuple[int, ...]:
        return encode_tabular(self)


def _relative(seat: int, observer: int, players: int) -> int:
    return (seat - observer) % players


def _encode_slot(target: np.ndarray, offset: int, seat: int, card: Card, view: PlayerView) -> None:
    target[offset + card.index + 1] = 1.0
    target[offset + CARD_ONE_HOT + _relative(seat, view.player, view.num_players)] = 1.0


def encode_neural(view: PlayerView) -> np.ndarray:
    """Encode a typed view using the authoritative schema layout."""
    layout = OBSERVATION_LAYOUT
    result = np.zeros(layout.size, dtype=np.float32)
    for card in view.hand:
        result[layout.hand.start + card.index] = 1.0
    for trick in view.trick_history:
        for _, card in trick:
            result[layout.public_played.start + card.index] = 1.0
    for _, card in view.current_trick:
        result[layout.public_played.start + card.index] = 1.0
    result[layout.turned_card.start + view.turned_card.index] = 1.0
    if view.own_rob_discard is not None:
        result[layout.own_rob_discard.start + view.own_rob_discard.index] = 1.0
    result[layout.trump.start + int(view.trump)] = 1.0
    result[layout.phase.start + list(Phase).index(view.phase)] = 1.0
    result[layout.player_mask.start : layout.player_mask.start + view.num_players] = 1.0
    result[layout.dealer.start + _relative(view.dealer, view.player, view.num_players)] = 1.0
    result[layout.leader.start + _relative(view.leader, view.player, view.num_players)] = 1.0
    result[layout.current_player.start + _relative(view.current_player, view.player, view.num_players)] = 1.0
    robber_index = MAX_PLAYERS if view.robber is None else _relative(view.robber, view.player, view.num_players)
    result[layout.robber.start + robber_index] = 1.0
    result[layout.robbed] = float(view.robbed)
    for relative in range(view.num_players):
        absolute = (view.player + relative) % view.num_players
        result[layout.scores.start + relative] = view.scores[absolute] / rules.WIN_SCORE
        result[layout.tricks.start + relative] = view.round_tricks[absolute] / rules.TRICKS_PER_ROUND
    for index, (seat, card) in enumerate(view.current_trick):
        _encode_slot(result, layout.current_slots.start + index * SLOT_SIZE, seat, card, view)
    slot = 0
    for trick in view.trick_history:
        for seat, card in trick:
            _encode_slot(result, layout.history_slots.start + slot * SLOT_SIZE, seat, card, view)
            slot += 1
    return result


def encode_tabular(view: PlayerView) -> tuple[int, ...]:
    """Compact, intentionally lossy state used only by tabular control baselines."""
    strengths = sorted(int(intrinsic_strength(card, view.trump) * 10) for card in view.hand)
    strengths = ([0] * 6 + strengths)[-6:]
    led = view.current_trick[0][1] if view.current_trick else None
    best = max((rules.trick_strength(card, view.trump, led) for _, card in view.current_trick), default=-1)
    rivals = [score for seat, score in enumerate(view.scores) if seat != view.player]
    score_gap = view.scores[view.player] - max(rivals, default=0)
    return (
        list(Phase).index(view.phase), int(view.trump), len(view.current_trick),
        -1 if led is None else (-1 if rules.is_trump(led, view.trump) else int(led.suit)),
        min(10, max(-1, best // 10)), max(-5, min(5, score_gap // 5)),
        view.round_tricks[view.player], *strengths,
    )


def player_view(game: TwentyFiveGame, player: int) -> PlayerView:
    return PlayerView.from_game(game, player)


def neural_observation(game: TwentyFiveGame, player: int) -> np.ndarray:
    return encode_neural(player_view(game, player))


def tabular_state(game: TwentyFiveGame, player: int) -> tuple[int, ...]:
    return encode_tabular(player_view(game, player))


def player_observation(game: TwentyFiveGame, player: int) -> dict[str, object]:
    view = player_view(game, player)
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "player": player,
        "hand": [card.index for card in view.hand],
        "public": dict(view.public),
        "own_rob_discard": view.own_rob_discard.index if view.own_rob_discard else None,
        "action_mask": view.action_mask.tolist(),
        "neural": view.neural,
        "tabular": view.tabular,
    }
