import tempfile
import unittest
from pathlib import Path

import numpy as np

from twenty_five.agents.tabular import TabularControl
from twenty_five.engine import TwentyFiveGame
from twenty_five.evaluation import evaluate, play_match
from twenty_five.observations import (
    NEURAL_OBSERVATION_SIZE,
    OBSERVATION_LAYOUT,
    OBSERVATION_SCHEMA_VERSION,
    PlayerView,
)
from twenty_five.policies import make_policy
from twenty_five.training.replay import ReplayBuffer


class AgentContractTests(unittest.TestCase):
    def test_observation_layout_is_contiguous_bounded_and_versioned(self):
        slices = [
            OBSERVATION_LAYOUT.hand,
            OBSERVATION_LAYOUT.public_played,
            OBSERVATION_LAYOUT.turned_card,
            OBSERVATION_LAYOUT.own_rob_discard,
            OBSERVATION_LAYOUT.trump,
            OBSERVATION_LAYOUT.phase,
            OBSERVATION_LAYOUT.player_mask,
            OBSERVATION_LAYOUT.dealer,
            OBSERVATION_LAYOUT.leader,
            OBSERVATION_LAYOUT.current_player,
            OBSERVATION_LAYOUT.robber,
            OBSERVATION_LAYOUT.robbed,
            OBSERVATION_LAYOUT.scores,
            OBSERVATION_LAYOUT.tricks,
            OBSERVATION_LAYOUT.current_slots,
            OBSERVATION_LAYOUT.history_slots,
        ]
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, 1)
        self.assertEqual(slices[0].start, 0)
        self.assertTrue(all(first.stop == second.start for first, second in zip(slices, slices[1:])))
        self.assertEqual(slices[-1].stop, NEURAL_OBSERVATION_SIZE)
        for players in range(2, 11):
            game = TwentyFiveGame(players, seed=players)
            for seat in range(players):
                encoded = PlayerView.from_game(game, seat).neural
                self.assertEqual(encoded.shape, (NEURAL_OBSERVATION_SIZE,))
                self.assertGreaterEqual(float(encoded.min()), 0.0)
                self.assertLessEqual(float(encoded.max()), 1.0)

    def test_expected_sarsa_target_does_not_consume_action_rng(self):
        agent = TabularControl(alpha=1.0, gamma=1.0, seed=17)
        agent.q[(1,)][3] = 4.0
        agent.q[(1,)][4] = 4.0
        state = agent.rng.getstate()
        agent.td_update((0,), 0, 0.0, (1,), [3, 4], "expected_sarsa", epsilon=0.2)
        self.assertEqual(agent.rng.getstate(), state)
        self.assertEqual(agent.value((0,), 0), 4.0)

    def test_tabular_checkpoint_restores_counts_and_rng(self):
        agent = TabularControl(seed=9)
        agent.monte_carlo_update([(1,)], [2], [1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            agent.save(path, episodes=12)
            restored = TabularControl.load(path)
        self.assertEqual(restored.counts[((1,), 2)], 1)
        self.assertEqual(restored.rng.getstate(), agent.rng.getstate())
        self.assertEqual(restored.episodes, 12)

    def test_compressed_replay_round_trip_preserves_quantized_entries(self):
        replay = ReplayBuffer(8, 4)
        state = np.asarray([0.0, 0.2, 0.8, 1.0], dtype=np.float32)
        replay.add(state, 3, 0.5, state[::-1], False, np.ones(52, dtype=np.int8), 0.99)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path, metadata={"episodes": 2, "environment_steps": 9})
            restored, metadata = ReplayBuffer.load(path)
        batch = restored.sample(1, np.random.default_rng(1), alpha=0.0, beta=0.0)
        np.testing.assert_allclose(batch.states[0], state, atol=1 / 255)
        self.assertEqual(metadata, {"episodes": 2, "environment_steps": 9})

    def test_main_benchmark_is_independent_of_candidate_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = evaluate({"random": make_policy("random")}, 8, [8123], root / "one", 4)
            second = evaluate(
                {"random": make_policy("random"), "lowest": make_policy("lowest")},
                8,
                [8123],
                root / "two",
                4,
                matchup_games=1,
            )
        first_row = next(row for row in first["rows"] if row["policy"] == "random")
        second_row = next(row for row in second["rows"] if row["policy"] == "random")
        for metric in ("wins", "win_rate", "average_final_score", "average_score_differential"):
            self.assertEqual(first_row[metric], second_row[metric])

    def test_every_baseline_completes_every_supported_player_count(self):
        names = ("random", "lowest", "highest", "greedy", "conservative", "counting")
        for players in range(2, 11):
            for index, name in enumerate(names):
                game, timing = play_match(
                    [make_policy(name) for _ in range(players)],
                    seed=50_000 + players * 100 + index,
                    dealer=index % players,
                    validate=True,
                )
                self.assertTrue(game.terminal)
                self.assertEqual(timing["mask_disagreements"], 0)


if __name__ == "__main__":
    unittest.main()
