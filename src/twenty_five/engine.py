"""Deterministic rules engine; callers submit actions and never advance turns."""

from __future__ import annotations

import ast
import copy
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from . import rules
from .cards import Card, FULL_DECK, Rank, Suit


class Phase(str, Enum):
    ROB = "rob"
    PLAY = "play"
    TERMINAL = "terminal"


class InvalidAction(ValueError):
    """Raised before mutation for out-of-turn, absent-card, or illegal actions."""


@dataclass(frozen=True)
class Transition:
    actor: int
    action: int
    phase: Phase
    trick_winner: int | None = None
    deal_completed: bool = False
    game_winner: int | None = None
    points_awarded: tuple[int, ...] = ()

    @property
    def round_completed(self) -> bool:
        """Compatibility alias for the older transition field name."""
        return self.deal_completed


def _card_data(card: Card) -> int:
    return card.index


STATE_FORMAT_VERSION = 2


class TwentyFiveGame:
    """Complete match state for 2..10 players.

    ``num_players`` is the only game-rule parameter. Random seeds and debug
    checks affect reproducibility/validation, never rules.
    """

    def __init__(
        self,
        num_players: int = 4,
        seed: int | None = None,
        debug: bool = False,
        initial_dealer: int = 0,
        *,
        initial_deck: Sequence[Card] | None = None,
    ):
        if not rules.MIN_PLAYERS <= num_players <= rules.MAX_PLAYERS:
            raise ValueError("num_players must be between 2 and 10")
        self.num_players = num_players
        self.debug = debug
        self.rng = random.Random(seed)
        self.initial_seed = seed
        if not 0 <= initial_dealer < num_players:
            raise ValueError("initial_dealer outside player range")
        self.initial_dealer = initial_dealer
        self.dealer = initial_dealer
        self.scores = [0] * num_players
        self.match_history: list[dict[str, Any]] = []
        self.round_number = 0
        self.winner: int | None = None
        self._deal_round(initial_deck)

    def _deal_round(self, ordered_deck: Sequence[Card] | None = None) -> None:
        deck = list(FULL_DECK if ordered_deck is None else ordered_deck)
        if len(deck) != 52 or set(deck) != set(FULL_DECK):
            raise ValueError("ordered deck must contain every physical card exactly once")
        if ordered_deck is None:
            self.rng.shuffle(deck)
        self.round_number += 1
        self.hands = [[] for _ in range(self.num_players)]
        deal_order = [(self.dealer + offset) % self.num_players for offset in range(1, self.num_players + 1)]
        cursor = 0
        for packet in range(2):
            for seat in deal_order:
                count = (2, 3)[packet] if seat == self.dealer else (3, 2)[packet]
                self.hands[seat].extend(deck[cursor : cursor + count])
                cursor += count
        self.turned_card = deck[cursor]
        cursor += 1
        self.trump = self.turned_card.suit
        self.stock = deck[cursor:]
        self.turned_zone = [self.turned_card]
        self.discards: list[Card] = []
        self.current_trick: list[tuple[int, Card]] = []
        self.trick_history: list[list[tuple[int, Card]]] = []
        self.trick_winners: list[int] = []
        self.round_tricks = [0] * self.num_players
        self.round_start_scores = list(self.scores)
        self.leader = (self.dealer + 1) % self.num_players
        self.robber = self._eligible_robber()
        self.rob_action: int | None = None
        if self.robber is not None:
            self.hands[self.robber].append(self.turned_zone.pop())
            self.phase = Phase.ROB
            self.current_player = self.robber
        else:
            self.phase = Phase.PLAY
            self.current_player = self.leader
        if self.debug:
            self.validate()

    def _eligible_robber(self) -> int | None:
        ace = Card(self.trump, Rank.ACE)
        if self.turned_card == ace:
            return self.dealer
        for seat, hand in enumerate(self.hands):
            if ace in hand:
                return seat
        return None

    @property
    def terminal(self) -> bool:
        return self.phase is Phase.TERMINAL

    def legal_cards(self, player: int | None = None) -> list[Card]:
        player = self.current_player if player is None else player
        if self.phase is Phase.TERMINAL or player != self.current_player:
            return []
        if self.phase is Phase.ROB:
            return rules.legal_rob_discards(self.hands[player], self.trump)
        return rules.legal_cards(self.hands[player], self.trump, self.current_trick)

    def legal_action_mask(self, player: int | None = None) -> list[int]:
        mask = [0] * 52
        for card in self.legal_cards(player):
            mask[card.index] = 1
        return mask

    def step(self, player: int, action: int) -> Transition:
        """Apply one legal card action atomically and advance all lifecycle state."""
        if self.terminal:
            raise InvalidAction("the match is terminal")
        if player != self.current_player:
            raise InvalidAction(f"player {player} acted out of turn; expected {self.current_player}")
        try:
            card = Card.from_index(action)
        except ValueError as exc:
            raise InvalidAction(str(exc)) from exc
        if card not in self.hands[player]:
            raise InvalidAction(f"player {player} does not hold {card}")
        if card not in self.legal_cards(player):
            raise InvalidAction(f"{card} is illegal for player {player} in {self.phase.value}")

        phase = self.phase
        if phase is Phase.ROB:
            self.hands[player].remove(card)
            self.discards.append(card)
            self.rob_action = action
            self.phase = Phase.PLAY
            self.current_player = self.leader
            transition = Transition(player, action, phase)
        else:
            transition = self._play(player, card, action)
        if self.debug:
            self.validate()
        return transition

    def _play(self, player: int, card: Card, action: int) -> Transition:
        self.hands[player].remove(card)
        self.current_trick.append((player, card))
        if len(self.current_trick) < self.num_players:
            self.current_player = (player + 1) % self.num_players
            return Transition(player, action, Phase.PLAY)

        won_by = rules.trick_winner(self.current_trick, self.trump)
        self.round_tricks[won_by] += 1
        self.scores[won_by] += rules.POINTS_PER_TRICK
        awarded = tuple(rules.POINTS_PER_TRICK if seat == won_by else 0 for seat in range(self.num_players))
        self.trick_winners.append(won_by)
        self.trick_history.append(list(self.current_trick))
        self.current_trick = []
        self.leader = won_by
        self.current_player = won_by
        if self.scores[won_by] >= rules.WIN_SCORE:
            self.winner = won_by
            self.phase = Phase.TERMINAL
            self.current_player = won_by
            self._archive_round(ended_match=True)
            return Transition(player, action, Phase.PLAY, won_by, True, won_by, awarded)
        if len(self.trick_history) < rules.TRICKS_PER_ROUND:
            return Transition(player, action, Phase.PLAY, trick_winner=won_by, points_awarded=awarded)
        self._archive_round(ended_match=False)
        self.dealer = (self.dealer + 1) % self.num_players
        self._deal_round()
        return Transition(player, action, Phase.PLAY, won_by, True, None, awarded)

    def _archive_round(self, *, ended_match: bool) -> None:
        awarded = tuple(t * rules.POINTS_PER_TRICK for t in self.round_tricks)
        record = {
            "round": self.round_number,
            "dealer": self.dealer,
            "trump": int(self.trump),
            "turned_card": self.turned_card.index,
            "robber": self.robber,
            "rob_action": self.rob_action,
            "discards": [_card_data(c) for c in self.discards],
            "tricks": [[(p, c.index) for p, c in trick] for trick in self.trick_history],
            "trick_winners": list(self.trick_winners),
            "points_awarded": list(awarded),
            "scores": list(self.scores),
            "completed_all_tricks": len(self.trick_history) == rules.TRICKS_PER_ROUND,
            "ended_match": ended_match,
            "winner": self.winner if ended_match else None,
        }
        self.match_history.append(record)

    def public_state(self) -> dict[str, Any]:
        return {
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
            "current_trick": [(p, c.index) for p, c in self.current_trick],
            "trick_history": [[(p, c.index) for p, c in t] for t in self.trick_history],
            "robber": self.robber,
            "robbed": self.rob_action is not None,
            "terminal": self.terminal,
            "winner": self.winner,
        }

    def clone(self) -> "TwentyFiveGame":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.public_state())
        data.update({
            "state_format_version": STATE_FORMAT_VERSION,
            "ruleset_version": rules.RULESET_VERSION,
            "debug": self.debug,
            "initial_seed": self.initial_seed,
            "initial_dealer": self.initial_dealer,
            "rng_state": repr(self.rng.getstate()),
            "hands": [[c.index for c in hand] for hand in self.hands],
            "stock": [c.index for c in self.stock],
            "turned_zone": [c.index for c in self.turned_zone],
            "rob_action": self.rob_action,
            "discards": [c.index for c in self.discards],
            "round_start_scores": list(self.round_start_scores),
            "trick_winners": list(self.trick_winners),
            "match_history": copy.deepcopy(self.match_history),
        })
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwentyFiveGame":
        if data.get("state_format_version") != STATE_FORMAT_VERSION:
            raise ValueError(f"unsupported state format; expected version {STATE_FORMAT_VERSION}")
        if data.get("ruleset_version") != rules.RULESET_VERSION:
            raise ValueError(f"state was created for a different ruleset; expected version {rules.RULESET_VERSION}")
        obj = cls.__new__(cls)
        obj.num_players = data["num_players"]
        obj.debug = data["debug"]
        obj.initial_seed = data["initial_seed"]
        obj.initial_dealer = data["initial_dealer"]
        obj.rng = random.Random()
        obj.rng.setstate(ast.literal_eval(data["rng_state"]))
        obj.phase = Phase(data["phase"])
        obj.round_number = data["round"]
        obj.dealer = data["dealer"]
        obj.leader = data["leader"]
        obj.current_player = data["current_player"]
        obj.trump = Suit(data["trump"])
        obj.turned_card = Card.from_index(data["turned_card"])
        obj.scores = list(data["scores"])
        obj.round_tricks = list(data["round_tricks"])
        obj.current_trick = [(p, Card.from_index(c)) for p, c in data["current_trick"]]
        obj.trick_history = [[(p, Card.from_index(c)) for p, c in t] for t in data["trick_history"]]
        obj.robber = data["robber"]
        obj.rob_action = data["rob_action"]
        obj.discards = [Card.from_index(c) for c in data["discards"]]
        obj.winner = data["winner"]
        obj.hands = [[Card.from_index(c) for c in h] for h in data["hands"]]
        obj.stock = [Card.from_index(c) for c in data["stock"]]
        obj.turned_zone = [Card.from_index(c) for c in data["turned_zone"]]
        obj.round_start_scores = list(data["round_start_scores"])
        obj.trick_winners = list(data["trick_winners"])
        obj.match_history = copy.deepcopy(data["match_history"])
        obj.validate()
        return obj

    @classmethod
    def from_json(cls, value: str) -> "TwentyFiveGame":
        return cls.from_dict(json.loads(value))

    def validate(self) -> None:
        if not rules.MIN_PLAYERS <= self.num_players <= rules.MAX_PLAYERS:
            raise AssertionError("player count outside supported range")
        seats = range(self.num_players)
        if any(value not in seats for value in (self.initial_dealer, self.dealer, self.leader, self.current_player)):
            raise AssertionError("dealer, leader, and current player must be valid seats")
        if len(self.hands) != self.num_players or len(self.scores) != self.num_players or len(self.round_tricks) != self.num_players:
            raise AssertionError("per-player state has the wrong size")
        if any(score < 0 or score % rules.POINTS_PER_TRICK for score in self.scores):
            raise AssertionError("scores must be non-negative multiples of five")

        physical = [c for hand in self.hands for c in hand]
        physical += list(self.stock) + list(self.turned_zone) + list(self.discards)
        physical += [c for trick in self.trick_history for _, c in trick]
        physical += [c for _, c in self.current_trick]
        if len(physical) != 52 or len(set(physical)) != 52:
            raise AssertionError("every physical card must occupy exactly one location")
        if any(len(t) != self.num_players for t in self.trick_history):
            raise AssertionError("completed tricks must contain one play per player")
        if len(self.trick_history) > rules.TRICKS_PER_ROUND:
            raise AssertionError("too many tricks in round")

        expected_leader = (self.dealer + 1) % self.num_players
        computed_winners: list[int] = []
        for trick in self.trick_history:
            expected_players = [(expected_leader + offset) % self.num_players for offset in range(self.num_players)]
            if [player for player, _ in trick] != expected_players:
                raise AssertionError("completed trick has invalid player order")
            won_by = rules.trick_winner(trick, self.trump)
            computed_winners.append(won_by)
            expected_leader = won_by
        if computed_winners != self.trick_winners:
            raise AssertionError("recorded trick winners disagree with played cards")
        expected_round_tricks = [computed_winners.count(seat) for seat in seats]
        if self.round_tricks != expected_round_tricks:
            raise AssertionError("round trick counts disagree with trick history")
        if self.leader != expected_leader:
            raise AssertionError("leader disagrees with trick winners")
        if self.scores != [start + tricks * rules.POINTS_PER_TRICK for start, tricks in zip(self.round_start_scores, self.round_tricks)]:
            raise AssertionError("scores disagree with current-round trick awards")

        expected_current_players = [
            (expected_leader + offset) % self.num_players for offset in range(len(self.current_trick))
        ]
        if [player for player, _ in self.current_trick] != expected_current_players:
            raise AssertionError("current trick has invalid player order")
        if self.current_trick and self.phase is not Phase.PLAY:
            raise AssertionError("only play phase may contain an incomplete trick")

        replay_hands = [list(hand) for hand in self.hands]
        for trick in self.trick_history:
            for player, card in trick:
                replay_hands[player].append(card)
        for player, card in self.current_trick:
            replay_hands[player].append(card)
        for trick in [*self.trick_history, self.current_trick]:
            replay_trick: list[tuple[int, Card]] = []
            for player, card in trick:
                if card not in rules.legal_cards(replay_hands[player], self.trump, replay_trick):
                    raise AssertionError("history contains a play that violated follow rules")
                replay_hands[player].remove(card)
                replay_trick.append((player, card))

        if self.robber is None:
            if self.rob_action is not None or self.discards or self.turned_zone != [self.turned_card]:
                raise AssertionError("non-rob round has inconsistent rob zones")
        else:
            if self.turned_zone:
                raise AssertionError("robbed turned card must leave the turned zone")
            if self.phase is Phase.ROB:
                if self.current_player != self.robber or self.rob_action is not None or self.discards:
                    raise AssertionError("invalid pending rob state")
            elif self.rob_action is None or len(self.discards) != 1 or self.discards[0].index != self.rob_action:
                raise AssertionError("completed rob must retain one private discard")
            else:
                pre_discard_hand = list(replay_hands[self.robber]) + list(self.discards)
                if self.discards[0] not in rules.legal_rob_discards(pre_discard_hand, self.trump):
                    raise AssertionError("rob history contains an illegal discard")

        if self.phase is Phase.ROB:
            expected = [rules.HAND_SIZE] * self.num_players
            expected[self.robber] += 1  # type: ignore[index]
            if [len(h) for h in self.hands] != expected:
                raise AssertionError("invalid rob-phase hand sizes")
        if self.phase is Phase.PLAY:
            complete = len(self.trick_history)
            played_now = {p for p, _ in self.current_trick}
            expected = [rules.HAND_SIZE - complete - (p in played_now) for p in range(self.num_players)]
            if [len(h) for h in self.hands] != expected:
                raise AssertionError("invalid play-phase hand sizes")
            expected_actor = (expected_leader + len(self.current_trick)) % self.num_players
            if self.current_player != expected_actor:
                raise AssertionError("current player does not follow trick order")
        if self.phase is Phase.TERMINAL:
            complete = len(self.trick_history)
            if self.current_trick or [len(hand) for hand in self.hands] != [rules.HAND_SIZE - complete] * self.num_players:
                raise AssertionError("terminal hand sizes disagree with completed tricks")
            if self.winner is None or self.current_player != self.winner or self.scores[self.winner] < rules.WIN_SCORE:
                raise AssertionError("terminal state has no valid winner")
            if any(score >= rules.WIN_SCORE for seat, score in enumerate(self.scores) if seat != self.winner):
                raise AssertionError("a non-winner reached the terminal score")
        elif self.winner is not None or any(score >= rules.WIN_SCORE for score in self.scores):
            raise AssertionError("nonterminal state has a winner or terminal score")

        cumulative_scores = [0] * self.num_players
        expected_dealer = self.initial_dealer
        current_round_start = list(cumulative_scores)
        for index, record in enumerate(self.match_history):
            if record["round"] != index + 1 or record["dealer"] != expected_dealer:
                raise AssertionError("match history has invalid round or dealer order")
            tricks = record["tricks"]
            winners = record["trick_winners"]
            if not 1 <= len(tricks) <= rules.TRICKS_PER_ROUND or len(winners) != len(tricks):
                raise AssertionError("archived round has invalid trick count")
            points = list(record["points_awarded"])
            archived_trump = Suit(record["trump"])
            archived_leader = (expected_dealer + 1) % self.num_players
            computed_archived_winners: list[int] = []
            for raw_trick in tricks:
                trick = [(int(player), Card.from_index(int(card))) for player, card in raw_trick]
                expected_players = [(archived_leader + offset) % self.num_players for offset in seats]
                if [player for player, _ in trick] != expected_players:
                    raise AssertionError("archived trick has invalid player order")
                won_by = rules.trick_winner(trick, archived_trump)
                computed_archived_winners.append(won_by)
                archived_leader = won_by
            if computed_archived_winners != winners:
                raise AssertionError("archived trick winners disagree with cards")
            expected_points = [winners.count(seat) * rules.POINTS_PER_TRICK for seat in seats]
            if points != expected_points:
                raise AssertionError("archived points disagree with trick winners")
            full_round = len(tricks) == rules.TRICKS_PER_ROUND
            ended_match = bool(record["ended_match"])
            if bool(record["completed_all_tricks"]) != full_round or (not full_round and not ended_match):
                raise AssertionError("only the terminal round may end before five tricks")
            if ended_match != (index == len(self.match_history) - 1 and self.phase is Phase.TERMINAL):
                raise AssertionError("terminal history marker is misplaced")
            current_round_start = list(cumulative_scores)
            cumulative_scores = [score + point for score, point in zip(cumulative_scores, points)]
            if list(record["scores"]) != cumulative_scores:
                raise AssertionError("archived cumulative scores are inconsistent")
            if ended_match:
                winner = int(record["winner"])
                if winner != winners[-1] or cumulative_scores[winner] < rules.WIN_SCORE:
                    raise AssertionError("archived terminal winner is inconsistent")
            elif record["winner"] is not None or any(score >= rules.WIN_SCORE for score in cumulative_scores):
                raise AssertionError("nonterminal archive contains a winner")
            expected_dealer = (expected_dealer + 1) % self.num_players

        if self.phase is Phase.TERMINAL:
            if not self.match_history or cumulative_scores != self.scores or self.round_start_scores != current_round_start:
                raise AssertionError("terminal state disagrees with archived final round")
            final = self.match_history[-1]
            current_tricks = [[(player, card.index) for player, card in trick] for trick in self.trick_history]
            archived_tricks = [[(int(player), int(card)) for player, card in trick] for trick in final["tricks"]]
            if archived_tricks != current_tricks or final["trick_winners"] != self.trick_winners:
                raise AssertionError("terminal round archive disagrees with live state")
            if self.round_number != len(self.match_history):
                raise AssertionError("terminal round counter is inconsistent")
        else:
            if self.round_start_scores != cumulative_scores or self.round_number != len(self.match_history) + 1:
                raise AssertionError("active round does not follow match history")
            if self.dealer != expected_dealer:
                raise AssertionError("active dealer does not follow match history")
