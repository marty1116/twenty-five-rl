import unittest
from pathlib import Path

from twenty_five.agents.tabular import TabularControl
from twenty_five.training.protocol import training_positions
from twenty_five.training.rules_lab import check_cases


class ResearchProtocolTests(unittest.TestCase):
    def test_positions_cover_cartesian_product(self):
        for players in range(2, 11):
            positions = [training_positions(i, players) for i in range(players ** 2)]
            self.assertEqual(set(positions), {(s, d) for s in range(players) for d in range(players)})
            self.assertEqual(len(positions), len(set(positions)))

    def test_reviewable_rules_cases(self):
        path = Path(__file__).resolve().parents[1] / "configs/research/rules_cases.json"
        for result in check_cases(path):
            with self.subTest(case=result["name"]):
                self.assertTrue(result["passed"], result)

    def test_td_targets_by_hand(self):
        for method, expected in [("sarsa", 2.0), ("q_learning", 3.0), ("expected_sarsa", 2.9)]:
            agent = TabularControl(alpha=1.0, gamma=0.5)
            agent.q[(1,)] = {0: 2.0, 1: 4.0}
            agent.td_update((0,), 0, 1.0, (1,), [0, 1], method, 0.2, 0)
            self.assertAlmostEqual(agent.value((0,), 0), expected)

    def test_tabular_algorithms_learn_known_terminal_choice(self):
        # Two-action, one-step MDP: action 1 yields +1, action 0 yields -1.
        # This verifies an actual learned preference, not file creation.
        for method in ("mc", "sarsa", "q_learning", "expected_sarsa"):
            for seed in range(3):
                agent = TabularControl(seed=seed)
                for _ in range(200):
                    action = agent.act((0,), [0, 1], epsilon=0.3)
                    reward = 1.0 if action == 1 else -1.0
                    if method == "mc":
                        agent.monte_carlo_update([(0,)], [action], [reward])
                    else:
                        agent.td_update((0,), action, reward, None, [], method)
                self.assertEqual(agent.act((0,), [0, 1]), 1)
