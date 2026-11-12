import io
import json
import tempfile
import unittest
import weakref
import gc
from contextlib import redirect_stdout
from pathlib import Path

try:
    from gymnasium.utils.env_checker import check_env
except ImportError:
    check_env = None

try:
    from pettingzoo.test import api_test
except ImportError:
    api_test = None

from twenty_five.envs import TwentyFiveAECEnv, TwentyFiveEnv
from twenty_five.training import sandbox, run


class ExperimentalContractTests(unittest.TestCase):
    @unittest.skipIf(check_env is None, "Gymnasium is not installed")
    def test_gymnasium_checker(self):
        # Gymnasium's generic checker samples the full Discrete(52) space and
        # does not consume action masks. Keep production/training behavior
        # strict while allowing the checker to test deterministic transitions.
        check_env(
            TwentyFiveEnv(4, controlled_seat=0, illegal_action_mode="penalize"),
            skip_render_check=True,
        )

    @unittest.skipIf(api_test is None, "PettingZoo is not installed")
    def test_pettingzoo_aec_checker(self):
        api_test(TwentyFiveAECEnv(4), num_cycles=1_000, verbose_progress=False)

    def test_aec_env_does_not_leak_memory_on_reset(self):
        env = TwentyFiveAECEnv(4)
        env.reset()
        ref = weakref.ref(env)
        del env
        gc.collect()
        self.assertIsNone(ref(), "TwentyFiveAECEnv instance was not garbage-collected after reset")

    def test_sandbox_run_save_and_dict_replay(self):
        with tempfile.TemporaryDirectory() as td:
            replay_path = Path(td) / "replay.json"
            out = io.StringIO()
            with redirect_stdout(out):
                sandbox.run(4, ["random"], seed=1, save=replay_path)
            self.assertTrue(replay_path.exists())
            # Test replaying tuple actions
            with redirect_stdout(out):
                sandbox.run(4, ["random"], seed=1, replay=replay_path)
            # Convert actions to dict format as produced by play.py / gui.py
            data = json.loads(replay_path.read_text(encoding="utf-8"))
            data["actions"] = [{"seat": p, "action": a, "actor": "test"} for p, a in data["actions"]]
            replay_path.write_text(json.dumps(data), encoding="utf-8")
            # Verify dict-based replay executes cleanly without ValueError
            with redirect_stdout(out):
                sandbox.run(4, ["random"], seed=1, replay=replay_path)

    def test_run_config_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            config_data = {
                "algorithm": "ppo",
                "train": {
                    "updates": 1,
                    "seed": 0,
                    "output": str(Path(td) / "ppo.pt"),
                    "players": 2,
                    "num_envs": 1,
                    "horizon": 2,
                    "device": "cpu",
                    "batch_size": 2,
                    "progress": False,
                    "log_interval": 0,
                    "checkpoint_interval": 0,
                    "eval_games": 0,
                },
            }
            config_path.write_text(json.dumps(config_data), encoding="utf-8")
            summary = run.run_config(config_path)
            self.assertEqual(summary["algorithm"], "ppo")


if __name__ == "__main__":
    unittest.main()
