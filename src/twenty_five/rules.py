"""The single authoritative, non-configurable Twenty-Five rules module."""

from __future__ import annotations

from .cards import Card, Rank, Suit

HAND_SIZE = 5
TRICKS_PER_ROUND = 5
POINTS_PER_TRICK = 5
WIN_SCORE = 25
MIN_PLAYERS = 2
MAX_PLAYERS = 10
RULESET_VERSION = 3


def is_ace_hearts(card: Card) -> bool:
    return card == Card(Suit.HEARTS, Rank.ACE)


def is_trump(card: Card, trump: Suit) -> bool:
    """A♥ belongs to the trump category, even when Hearts is not turned."""
    return card.suit == trump or is_ace_hearts(card)


def led_category(card: Card, trump: Suit) -> Suit | None:
    """Return None for the trump category, otherwise the ordinary led suit."""
    return None if is_trump(card, trump) else card.suit


def protected_when_trump_led(card: Card, trump: Suit, led_card: Card) -> bool:
    five = Card(trump, Rank.FIVE)
    jack = Card(trump, Rank.JACK)
    if card == five:
        return True
    if card == jack:
        return led_card != five
    if is_ace_hearts(card):
        return led_card not in (five, jack)
    return False


def legal_cards(hand: list[Card], trump: Suit, trick: list[tuple[int, Card]]) -> list[Card]:
    if not trick:
        return list(hand)
    led = trick[0][1]
    category = led_category(led, trump)
    if category is None:
        held_trumps = [c for c in hand if is_trump(c, trump)]
        required = [c for c in held_trumps if not protected_when_trump_led(c, trump, led)]
        return held_trumps if required else list(hand)
    followers = [c for c in hand if not is_trump(c, trump) and c.suit == category]
    if not followers:
        return list(hand)
    # When an ordinary suit is led, a player may ruff even while holding that
    # suit. If they do not ruff, they must follow suit. Preserve hand order so
    # deterministic policies and seeded evaluations remain reproducible.
    return [c for c in hand if c in followers or is_trump(c, trump)]


def legal_rob_discards(hand: list[Card], trump: Suit) -> list[Card]:
    """Return legal face-down discards for the mandatory rob action.

    A normal rob may discard any non-trump card. A six-trump hand is possible,
    although extremely rare; in that case the ace enabling the rob is retained
    and only the weakest remaining trump is legal.
    """
    non_trumps = [card for card in hand if not is_trump(card, trump)]
    if non_trumps:
        return non_trumps
    enabling_ace = Card(trump, Rank.ACE)
    fallback = [card for card in hand if card != enabling_ace]
    if not fallback:
        raise ValueError("a rob hand must contain a discardable card")
    weakest = min(fallback, key=lambda card: (trump_strength(card, trump), card.index))
    return [weakest]


_RED_TRUMP = (Rank.TEN, Rank.NINE, Rank.EIGHT, Rank.SEVEN, Rank.SIX, Rank.FOUR, Rank.THREE, Rank.TWO)
_BLACK_TRUMP = (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN)
_RED_PLAIN = (
    Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN, Rank.NINE, Rank.EIGHT, Rank.SEVEN,
    Rank.SIX, Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO, Rank.ACE,
)
_BLACK_PLAIN = (
    Rank.KING, Rank.QUEEN, Rank.JACK, Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR,
    Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
)


def trump_strength(card: Card, trump: Suit) -> int:
    """Intrinsic trump strength; non-trumps return -1 (larger is stronger)."""
    top = (Card(trump, Rank.FIVE), Card(trump, Rank.JACK), Card(Suit.HEARTS, Rank.ACE))
    if card in top:
        return 100 - top.index(card)
    if not is_trump(card, trump):
        return -1
    court = (Rank.ACE, Rank.KING, Rank.QUEEN)
    if card.rank in court:
        return 90 - court.index(card.rank)
    order = _RED_TRUMP if trump.is_red else _BLACK_TRUMP
    return 80 - order.index(card.rank) if card.rank in order else -1


def plain_strength(card: Card) -> int:
    order = _RED_PLAIN if card.suit.is_red else _BLACK_PLAIN
    return len(order) - order.index(card.rank)


def trick_strength(card: Card, trump: Suit, led_card: Card) -> int:
    """Authoritative comparison strength in the specified trick context."""
    if is_trump(card, trump):
        return 1000 + trump_strength(card, trump)
    category = led_category(led_card, trump)
    if category is not None and card.suit == category:
        return plain_strength(card)
    return -1


def trick_winner(trick: list[tuple[int, Card]], trump: Suit) -> int:
    if not trick:
        raise ValueError("cannot rank an empty trick")
    led = trick[0][1]
    return max(trick, key=lambda play: trick_strength(play[1], trump, led))[0]
