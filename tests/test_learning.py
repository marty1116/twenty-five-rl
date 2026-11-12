import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - engine-only verification environment
    raise unittest.SkipTest("PyTorch is not installed") from exc

from twenty_five.agents.neural import LinearActorCritic, MLPQNetwork
from twenty_five.agents.tabular import TabularControl, discounted_returns
from twenty_five.evaluation import evaluate
from twenty_five.observations import NEURAL_OBSERVATION_SIZE
from twenty_five.policies import make_policy
from twenty_five.training.gridworld_mc import train as grid_train
from twenty_five.training.tabular_twentyfive import train as tabular_train


class LearningTests(unittest.TestCase):
    def test_returns_and_tabular_save_load(self):
        self.assertEqual(discounted_returns([1,2],.5),[2,2]); a=TabularControl(); a.q[(1,2)][3]=4
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"q.json"; a.save(p); b=TabularControl.load(p); self.assertEqual(b.q[(1,2)][3],4)

    def test_neural_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            q=MLPQNetwork(NEURAL_OBSERVATION_SIZE,8,dueling=True); p=Path(d)/"q.pt"; q.save(p); self.assertEqual(MLPQNetwork.load(p)(torch.zeros((1,NEURAL_OBSERVATION_SIZE))).shape,(1,52))
            ac=LinearActorCritic(NEURAL_OBSERVATION_SIZE); p2=Path(d)/"ac.pt"; ac.save(p2); self.assertEqual(LinearActorCritic.load(p2).actor.out_features,52)

    def test_smoke_training_and_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.assertEqual(grid_train(5,1,root/"g.json")["episodes"],5); self.assertEqual(tabular_train("q_learning",1,1,root/"t.json",2)["episodes"],1)
            result=evaluate({"random":make_policy("random"),"lowest":make_policy("lowest")},1,[9991],root/"eval",2)
            self.assertTrue((root/"eval/leaderboard.md").exists()); self.assertTrue((root/"eval/leaderboard.csv").exists()); self.assertEqual(len(result["rows"]),2)


if __name__=="__main__": unittest.main()
