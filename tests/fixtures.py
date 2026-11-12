from collections.abc import Sequence

from twenty_five import Card, Rank, Suit
from twenty_five.cards import FULL_DECK


def deck_for_hands(
    hands: Sequence[Sequence[Card]],
    turned_card: Card,
    dealer: int = 0,
) -> tuple[Card, ...]:
    """Build draw order for exact five-card hands under the 3-then-2 deal."""
    players = len(hands)
    if any(len(hand) != 5 for hand in hands):
        raise ValueError("each fixture hand must contain five cards")
    selected = [card for hand in hands for card in hand] + [turned_card]
    if len(set(selected)) != len(selected):
        raise ValueError("fixture cards must be unique")

    order = [(dealer + offset) % players for offset in range(1, players + 1)]
    cursors = [0] * players
    draw_order: list[Card] = []
    for packet in range(2):
        for seat in order:
            count = (2, 3)[packet] if seat == dealer else (3, 2)[packet]
            start = cursors[seat]
            draw_order.extend(hands[seat][start : start + count])
            cursors[seat] += count
    draw_order.append(turned_card)
    draw_order.extend(card for card in FULL_DECK if card not in selected)
    return tuple(draw_order)


def rob_deck(robber: int | None = 2, turned_ace: bool = False) -> tuple[Card, ...]:
    hands = [
        [Card(Suit.HEARTS, rank) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)],
        [
            Card(Suit.HEARTS, rank)
            for rank in (Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK)
        ],
        [
            Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.DIAMONDS, Rank.FOUR), Card(Suit.DIAMONDS, Rank.FIVE),
            Card(Suit.DIAMONDS, Rank.SIX),
        ],
        [
            Card(Suit.HEARTS, Rank.QUEEN), Card(Suit.HEARTS, Rank.KING),
            Card(Suit.CLUBS, Rank.FOUR), Card(Suit.DIAMONDS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.EIGHT),
        ],
    ]
    if robber is not None and not turned_ace:
        hands[robber][0] = Card(Suit.SPADES, Rank.ACE)
    turned = Card(Suit.SPADES, Rank.ACE if turned_ace else Rank.QUEEN)
    return deck_for_hands(hands, turned)
