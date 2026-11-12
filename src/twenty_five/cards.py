"""Immutable card types and stable 0..51 action encoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def index(self) -> int:
        return int(self.suit) * 13 + int(self.rank)

    @classmethod
    def from_index(cls, index: int) -> "Card":
        if not 0 <= index < 52:
            raise ValueError(f"card action must be in [0, 51], got {index}")
        return cls(Suit(index // 13), Rank(index % 13))

    def __str__(self) -> str:
        ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
        suits = ("H", "D", "C", "S")
        return f"{ranks[int(self.rank)]}{suits[int(self.suit)]}"


FULL_DECK: tuple[Card, ...] = tuple(Card(s, r) for s in Suit for r in Rank)
