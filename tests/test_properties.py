import random
import unittest

import numpy as np

from twenty_five import rules
from twenty_five.engine import Phase, TwentyFiveGame
from twenty_five.observations import OBSERVATION_LAYOUT, neural_observation, player_observation
from tests.fixtures import rob_deck


class PropertyStyleTests(unittest.TestCase):
    def test_random_legal_matches_preserve_all_invariants(self):
        for players in range(2, 11):
            for seed in range(10):
                game = TwentyFiveGame(players, seed=10_000 * players + seed, debug=True)
                rng = random.Random(seed)
                while not game.terminal:
                    legal = game.legal_cards()
                    game.step(game.current_player, rng.choice(legal).index)
                self.assertEqual(sum(game.scores), sum(sum(record["points_awarded"]) for record in game.match_history))

    def test_private_observation_does_not_reveal_opponent_card_ownership(self):
        game = next(game for seed in range(100) if (game := TwentyFiveGame(4, seed)).phase is Phase.PLAY)
        before = neural_observation(game, 0)
        game.hands[1][0], game.hands[2][0] = game.hands[2][0], game.hands[1][0]
        game.validate()
        after = neural_observation(game, 0)
        np.testing.assert_array_equal(before, after)

    def test_face_down_rob_discard_is_known_only_to_robber(self):
        game = TwentyFiveGame(4, initial_deck=rob_deck(2))
        robber = game.robber
        assert robber is not None
        discard = next(card for card in game.legal_cards() if not rules.is_trump(card, game.trump))
        game.step(robber, discard.index)
        self.assertNotIn("rob_action", game.public_state())
        self.assertNotIn("discards", game.public_state())
        self.assertEqual(player_observation(game, robber)["own_rob_discard"], discard.index)
        for player in range(game.num_players):
            discard_offset = OBSERVATION_LAYOUT.own_rob_discard.start + discard.index
            if player == robber:
                self.assertEqual(neural_observation(game, player)[discard_offset], 1)
            else:
                self.assertEqual(player_observation(game, player)["own_rob_discard"], None)
                self.assertEqual(neural_observation(game, player)[discard_offset], 0)

    def test_observation_survives_serialization_exactly(self):
        game = TwentyFiveGame(6, seed=44, debug=True)
        restored = TwentyFiveGame.from_json(game.to_json())
        for player in range(game.num_players):
            np.testing.assert_array_equal(neural_observation(game, player), neural_observation(restored, player))

    def test_random_intermediate_states_round_trip_and_remain_playable(self):
        for players in (2,4,10):
            for seed in range(5):
                game=TwentyFiveGame(players,seed=1000*players+seed,debug=True)
                rng=random.Random(seed)
                for _ in range(20):
                    restored=TwentyFiveGame.from_json(game.to_json())
                    self.assertEqual(restored.to_json(),game.to_json())
                    if game.terminal:
                        break
                    action=rng.choice(game.legal_cards()).index
                    original_transition=game.step(game.current_player,action)
                    restored_transition=restored.step(restored.current_player,action)
                    self.assertEqual(original_transition,restored_transition)
                    self.assertEqual(game.to_json(),restored.to_json())


if __name__ == "__main__":
    unittest.main()
