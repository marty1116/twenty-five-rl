"""Experimental heuristic values; these never determine authoritative legality."""

from __future__ import annotations

from collections import Counter

from . import rules
from .cards import Card, Rank, Suit


def intrinsic_strength(card: Card, trump: Suit) -> float:
    """Trump-aware strength on an explainable, non-authoritative 0..1 scale."""
    trump_value = rules.trump_strength(card, trump)
    if trump_value >= 0:
        return 0.70 + 0.003 * trump_value
    return 0.02 + 0.025 * rules.plain_strength(card)


def discard_value(card: Card, hand: list[Card], trump: Suit) -> float:
    """Estimated strategic cost of discarding ``card``; lower is preferred.

    It combines intrinsic strength with suit length, short-suit/void creation,
    and special-card protection. The scale is only meaningful within a hand.
    """
    base = 100.0 * intrinsic_strength(card, trump)
    if card == Card(trump, Rank.FIVE):
        base += 100
    elif card == Card(trump, Rank.JACK):
        base += 80
    elif rules.is_ace_hearts(card):
        base += 70
    ordinary = [c for c in hand if not rules.is_trump(c, trump)]
    counts = Counter(c.suit for c in ordinary)
    if not rules.is_trump(card, trump):
        length = counts[card.suit]
        base += 3.0 * max(0, length - 1)  # long suits retain follow-suit utility
        if length == 1:
            base -= 12.0  # discarding a singleton creates a trumping void
        elif length == 2:
            base -= 3.0
        stronger_same_suit = sum(
            rules.plain_strength(c) > rules.plain_strength(card)
            for c in ordinary
            if c.suit == card.suit and c != card
        )
        base -= 1.5 * stronger_same_suit
    return base


def order_legal_cards(
    legal: list[Card], hand: list[Card], trump: Suit, led_card: Card | None = None
) -> list[Card]:
    """Least-to-most strategically valuable ordering for baseline policies."""
    if len(hand) == 6:  # rob discard decision
        return sorted(legal, key=lambda c: (discard_value(c, hand, trump), c.index))
    if led_card is not None:
        return sorted(legal, key=lambda c: (rules.trick_strength(c, trump, led_card), c.index))
    return sorted(legal, key=lambda c: (intrinsic_strength(c, trump), c.index))
