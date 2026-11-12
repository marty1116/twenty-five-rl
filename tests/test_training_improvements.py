import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError as exc:  # pragma: no cover - engine-only verification environment
    raise unittest.SkipTest("PyTorch is not installed") from exc

from twenty_five.agents.neural import CardAwareActorCritic, LinearActorCritic, load_actor_critic
from twenty_five.engine import Phase, Transition, TwentyFiveGame
from twenty_five.evaluation import evaluate, evaluation_schedule
from twenty_five.observations import NEURAL_OBSERVATION_SIZE, OBSERVATION_LAYOUT
from twenty_five.policies import make_policy
from twenty_five.rewards import reward_spec
from twenty_five.training.dqn import train as dqn_train
from twenty_five.training.experiment import TerminalProgress
from twenty_five.training.ppo import train as ppo_train
from twenty_five.training.policy_gradient import train as policy_gradient_train
from twenty_five.training.play import replay_game
from twenty_five.training.replay import ReplayBuffer


class TrainingImprovementTests(unittest.TestCase):
    def test_terminal_progress_reports_completion_eta_and_metrics(self):
        stream = io.StringIO()
        stream.isatty = lambda: True  # type: ignore[method-assign]
        progress = TerminalProgress(10, label="PPO", stream=stream)
        progress.update(4, episodes=1234, speed=2500.0, evaluation=0.25)
        progress.close()
        rendered = stream.getvalue()
        self.assertIn("PPO 4/10", rendered)
        self.assertIn("40.0%", rendered)
        self.assertIn("ETA", rendered)
        self.assertIn("episodes 1,234", rendered)
        self.assertIn("2,500 trans/s", rendered)
        self.assertIn("eval 25.0%", rendered)

    def test_evaluation_schedule_honors_exact_game_count(self):
        schedule = evaluation_schedule(19, [9001, 9002], 4)
        self.assertEqual(len(schedule), 19)
        self.assertEqual(len({(seat, dealer) for _, seat, dealer in schedule[:16]}), 16)
        self.assertEqual(schedule[16][0], 9002)

    def test_evaluation_reports_exact_games_and_multiplayer_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(
                {"random": make_policy("random"), "lowest": make_policy("lowest")},
                7,
                [9911, 9912],
                Path(directory),
                4,
                matchup_games=2,
            )
            self.assertEqual(result["actual_games_per_policy"], {"random": 7, "lowest": 7})
            self.assertEqual(result["multiplayer_matchup_games_per_cell"], 2)
            self.assertIn("two_player_head_to_head_matrix", result)

    def test_shaped_rewards_remain_zero_sum(self):
        scored = Transition(0, 1, Phase.PLAY, trick_winner=1, points_awarded=(0, 5, 0, 0))
        terminal = Transition(0, 1, Phase.PLAY, trick_winner=1, game_winner=1, points_awarded=(0, 5, 0, 0))
        for scheme, transition in (("score_difference", scored), ("terminal", terminal), ("terminal_plus_score", terminal)):
            self.assertAlmostEqual(sum(reward_spec(scheme).event_rewards(transition, 4)), 0.0)

    def test_separate_rob_head_roundtrip(self):
        model = LinearActorCritic(NEURAL_OBSERVATION_SIZE, phase_heads=True)
        self.assertIsNotNone(model.rob_actor)
        observations = torch.zeros((2, NEURAL_OBSERVATION_SIZE))
        observations[0, OBSERVATION_LAYOUT.phase.start] = 1.0
        observations[1, OBSERVATION_LAYOUT.phase.start + 1] = 1.0
        masks = torch.ones((2, 52), dtype=torch.bool)
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.zero_()
            model.rob_actor.weight.zero_()
            model.rob_actor.bias.fill_(1.0)
            logits, values = model.policy_and_value(observations, masks)
        self.assertTrue(torch.all(logits[0] > logits[1]))
        self.assertEqual(values.shape, (2,))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model.save(checkpoint)
            self.assertTrue(LinearActorCritic.load(checkpoint).phase_heads)

    def test_prioritized_replay_shapes_and_priority_update(self):
        replay = ReplayBuffer(16, 4)
        for index in range(8):
            replay.add(
                np.full(4, index, dtype=np.float32),
                index,
                float(index),
                np.full(4, index + 1, dtype=np.float32),
                False,
                np.ones(52, dtype=np.int8),
                0.99,
            )
        batch = replay.sample(4, np.random.default_rng(4))
        self.assertEqual(batch.states.shape, (4, 4))
        replay.update_priorities(batch.indices, np.full(4, 7.0))
        np.testing.assert_allclose(replay.priorities[batch.indices], 7.00001, rtol=1e-5)

    def test_card_aware_actor_uses_structured_history_and_roundtrips(self):
        model = CardAwareActorCritic(NEURAL_OBSERVATION_SIZE)
        observations = torch.zeros((2, NEURAL_OBSERVATION_SIZE))
        observations[:, :5] = 1.0
        observations[1, OBSERVATION_LAYOUT.history_slots.start + 1] = 1.0
        observations[1, OBSERVATION_LAYOUT.history_slots.start + 53] = 1.0
        masks = torch.ones((2, 52), dtype=torch.bool)
        logits, values = model.policy_and_value(observations, masks)
        self.assertEqual(logits.shape, (2, 52))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.isfinite(logits).all())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "card-aware.pt"
            model.save(checkpoint)
            self.assertIsInstance(load_actor_critic(checkpoint), CardAwareActorCritic)

    def test_batched_trainers_write_resumable_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dqn_path = root / "dqn.pt"
            dqn = dqn_train(
                2,
                5,
                dqn_path,
                players=2,
                device="cpu",
                batch_size=4,
                num_envs=2,
                replay_capacity=32,
                warmup_steps=100,
                collect_only=True,
                eval_games=0,
                checkpoint_interval=0,
                log_interval=10,
            )
            self.assertEqual(dqn["episodes_this_run"], 2)
            self.assertIn("optimizer_state_dict", torch.load(dqn_path, weights_only=True)["training_state"])
            self.assertTrue(dqn_path.with_suffix(".replay.npz").exists())
            resumed_dqn = dqn_train(
                1,
                5,
                dqn_path,
                players=2,
                device="cpu",
                batch_size=4,
                num_envs=1,
                replay_capacity=32,
                warmup_steps=100,
                collect_only=True,
                resume=dqn_path,
                eval_games=0,
                checkpoint_interval=0,
                log_interval=10,
            )
            self.assertEqual(resumed_dqn["replay_restored"], 1)

            ppo_path = root / "ppo.pt"
            ppo = ppo_train(
                2,
                7,
                ppo_path,
                players=2,
                num_envs=1,
                horizon=1,
                self_play=True,
                self_play_snapshot_interval=1,
                device="cpu",
                batch_size=2,
                eval_games=0,
                checkpoint_interval=0,
                log_interval=10,
            )
            self.assertGreaterEqual(ppo["environment_steps"], 2)
            self.assertGreaterEqual(ppo["environment_resets"], 1)
            self.assertTrue(ppo["shared_policy_all_seats"])
            self.assertIn("optimizer_state_dict", torch.load(ppo_path, weights_only=True)["training_state"])

            resumed = ppo_train(
                1,
                7,
                ppo_path,
                players=2,
                num_envs=1,
                horizon=1,
                self_play=True,
                resume=ppo_path,
                device="cpu",
                batch_size=2,
                self_play_snapshot_interval=1,
                eval_games=0,
                checkpoint_interval=0,
                log_interval=10,
            )
            payload = torch.load(ppo_path, weights_only=True)
            self.assertEqual(resumed["updates"], 3)
            self.assertIn("chooser_rng_state", payload["training_state"])
            self.assertIn("historical_state_dicts", payload["training_state"])

    def test_n_step_actor_critic_is_a_distinct_trainable_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = policy_gradient_train(
                "n_step_actor_critic",
                1,
                13,
                Path(directory) / "actor.pt",
                players=2,
                device="cpu",
                n_steps=2,
            )
        self.assertEqual(summary["algorithm"], "n_step_actor_critic")

    def test_saved_game_replays_to_identical_terminal_state(self):
        game = TwentyFiveGame(2, seed=88)
        initial = game.to_dict()
        rng = random.Random(88)
        actions = []
        while not game.terminal:
            seat = game.current_player
            action = rng.choice(game.legal_cards()).index
            actions.append({"seat": seat, "action": action})
            game.step(seat, action)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps({"initial": initial, "actions": actions, "human_seat": 0}))
            with contextlib.redirect_stdout(io.StringIO()):
                replayed = replay_game(path)
        self.assertEqual(replayed.to_json(), game.to_json())


if __name__ == "__main__":
    unittest.main()
