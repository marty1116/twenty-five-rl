import random
import unittest

from twenty_five import Card, InvalidAction, Phase, Rank, Suit, TwentyFiveGame
from twenty_five import rules
from twenty_five.cards import FULL_DECK
from tests.fixtures import deck_for_hands, rob_deck


def rob_fixture(robber: int | None = 2, turned_ace: bool = False) -> TwentyFiveGame:
    return TwentyFiveGame(4,initial_dealer=0,initial_deck=rob_deck(robber,turned_ace))


class EngineTests(unittest.TestCase):
    def test_three_then_two_deal_starts_left_of_dealer_and_inverts_dealer_packets(self):
        for dealer in range(4):
            game=TwentyFiveGame(4,initial_dealer=dealer,initial_deck=FULL_DECK)
            order=[(dealer+offset)%4 for offset in range(1,5)]
            expected=[[] for _ in range(4)]
            cursor=0
            for packet in range(2):
                for seat in order:
                    count=(2,3)[packet] if seat==dealer else (3,2)[packet]
                    expected[seat].extend(FULL_DECK[cursor:cursor+count])
                    cursor+=count
            self.assertEqual(game.hands,expected)
            self.assertEqual(game.turned_card,FULL_DECK[20])

    def test_turn_card_follows_five_cards_per_supported_player(self):
        for players in range(2,11):
            game=TwentyFiveGame(players,initial_dealer=players-1,initial_deck=FULL_DECK,debug=True)
            self.assertEqual(game.turned_card,FULL_DECK[5*players])

    def test_seeded_deal_and_serialization_clone(self):
        a=TwentyFiveGame(4,123,debug=True); b=TwentyFiveGame(4,123,debug=True)
        self.assertEqual(a.to_json(),b.to_json())
        c=TwentyFiveGame.from_json(a.to_json()); d=a.clone(); action=a.legal_cards()[0].index
        ta=a.step(a.current_player,action); tc=c.step(c.current_player,action); td=d.step(d.current_player,action)
        self.assertEqual(ta,tc); self.assertEqual(a.to_json(),c.to_json()); self.assertEqual(a.to_json(),d.to_json())

    def test_out_of_turn_and_illegal_are_atomic(self):
        game=TwentyFiveGame(4,2); before=game.to_json(); wrong=(game.current_player+1)%4
        with self.assertRaises(InvalidAction): game.step(wrong,game.hands[wrong][0].index)
        self.assertEqual(before,game.to_json())
        absent=next(c for c in [Card.from_index(i) for i in range(52)] if c not in game.hands[game.current_player])
        with self.assertRaises(InvalidAction): game.step(game.current_player,absent.index)
        self.assertEqual(before,game.to_json())

    def test_non_dealer_robber_and_discard(self):
        game=rob_fixture(2)
        self.assertEqual(game.phase,Phase.ROB); self.assertEqual(len(game.hands[game.robber]),6)
        self.assertTrue(all(not rules.is_trump(card,game.trump) for card in game.legal_cards()))
        action=game.legal_cards()[0].index; robber=game.robber; game.step(robber,action)
        self.assertEqual(len(game.hands[robber]),5)
        self.assertEqual(game.phase,Phase.PLAY)
        self.assertEqual(game.current_player,(game.dealer+1)%4)
        self.assertNotIn("rob_action",game.public_state()); self.assertNotIn("discards",game.public_state())

    def test_dealer_robs_when_dealt_ace(self):
        game=rob_fixture(0)
        self.assertEqual(game.current_player,game.dealer)

    def test_turned_ace_makes_dealer_robber(self):
        game=rob_fixture(None,turned_ace=True)
        self.assertEqual(game.robber,game.dealer); self.assertEqual(game.phase,Phase.ROB)

    def test_all_trump_rob_fallback_completes_without_deadlock(self):
        hands=[
            [Card(Suit.HEARTS,rank) for rank in (Rank.TWO,Rank.THREE,Rank.FOUR,Rank.FIVE,Rank.SIX)],
            [Card(Suit.SPADES,rank) for rank in (Rank.ACE,Rank.TWO,Rank.THREE,Rank.FOUR,Rank.SIX)],
        ]
        turned=Card(Suit.SPADES,Rank.SEVEN)
        game=TwentyFiveGame(2,initial_deck=deck_for_hands(hands,turned),debug=True)
        self.assertEqual(game.legal_cards(),[turned])
        game.step(game.current_player,turned.index)
        self.assertEqual(game.phase,Phase.PLAY)

    def test_mask_exactly_matches_legality(self):
        game=TwentyFiveGame(6,44,debug=True); rng=random.Random(1)
        for _ in range(40):
            legal={c.index for c in game.legal_cards()}
            mask={i for i,v in enumerate(game.legal_action_mask()) if v}
            self.assertEqual(legal,mask)
            game.step(game.current_player,rng.choice(list(legal)))
            if game.terminal: break

    def test_complete_matches_all_player_counts(self):
        for players in range(2,11):
            game=TwentyFiveGame(players,100+players,debug=True); rng=random.Random(players)
            while not game.terminal: game.step(game.current_player,rng.choice(game.legal_cards()).index)
            self.assertGreaterEqual(game.scores[game.winner],25)
            self.assertTrue(all(len(r["tricks"])==5 and sum(r["points_awarded"])==25 for r in game.match_history[:-1]))
            self.assertTrue(1<=len(game.match_history[-1]["tricks"])<=5)
            self.assertTrue(game.match_history[-1]["ended_match"])
            self.assertEqual([r["dealer"] for r in game.match_history],
                             [i%players for i in range(len(game.match_history))])

    def test_match_ends_on_the_trick_that_reaches_twenty_five(self):
        game=TwentyFiveGame(4,12)
        if game.phase is Phase.ROB:
            game.step(game.current_player,game.legal_cards()[0].index)
        game.scores=[20]*4; game.round_start_scores=[20]*4
        rng=random.Random(12)
        while not game.trick_history:
            transition=game.step(game.current_player,rng.choice(game.legal_cards()).index)
        winner=game.trick_winners[0]
        self.assertTrue(game.terminal); self.assertEqual(game.winner,winner)
        self.assertEqual(game.scores[winner],25)
        self.assertEqual(len(game.match_history),1)
        self.assertFalse(game.match_history[0]["completed_all_tricks"])
        self.assertEqual(sum(transition.points_awarded),5)
        self.assertTrue(transition.deal_completed); self.assertTrue(transition.round_completed)

    def test_state_format_is_versioned_and_loaded_state_is_validated(self):
        game=TwentyFiveGame(4,123,debug=False)
        data=game.to_dict(); self.assertEqual(data["state_format_version"],2)
        data["hands"][0].append(data["hands"][1][0])
        with self.assertRaises(AssertionError): TwentyFiveGame.from_dict(data)
        legacy=game.to_dict(); legacy.pop("state_format_version")
        with self.assertRaises(ValueError): TwentyFiveGame.from_dict(legacy)


if __name__=="__main__": unittest.main()
