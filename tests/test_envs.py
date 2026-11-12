import random
import unittest
import numpy as np

from twenty_five import Phase, TwentyFiveGame
from twenty_five.envs import TwentyFiveAECEnv, TwentyFiveEnv
from twenty_five.observations import NEURAL_OBSERVATION_SIZE
from tests.fixtures import rob_deck


class EnvironmentTests(unittest.TestCase):
    def test_gym_shapes_seed_and_no_invalid_terminal(self):
        env=TwentyFiveEnv(4,controlled_seat=0,illegal_action_mode="raise"); obs,info=env.reset(seed=8); obs2,_=env.reset(seed=8)
        self.assertEqual(obs["observation"].shape,(NEURAL_OBSERVATION_SIZE,))
        np.testing.assert_array_equal(obs["observation"],obs2["observation"])
        illegal=next((i for i,v in enumerate(obs2["action_mask"]) if not v),None)
        if illegal is not None:
            before=env.game.to_json()
            with self.assertRaises(ValueError): env.step(illegal)
            self.assertEqual(before,env.game.to_json()); self.assertFalse(env.game.terminal)

    def test_gym_rob_action_is_restricted(self):
        env=TwentyFiveEnv(4,controlled_seat=0); obs,info=env.reset(seed=1,options={"initial_deck":rob_deck(0)})
        self.assertEqual(env.game.phase,Phase.ROB)
        self.assertTrue(0<int(obs["action_mask"].sum())<6)
        env.step(int(np.flatnonzero(obs["action_mask"])[0])); self.assertEqual(len(env.game.hands[0]),5)

    def test_default_gym_contract_truncates_illegal_sample_without_mutating_engine(self):
        env=TwentyFiveEnv(4,controlled_seat=0); obs,_=env.reset(seed=8)
        illegal=next(i for i,value in enumerate(obs["action_mask"]) if not value)
        before=env.game.to_json()
        _,reward,terminated,truncated,info=env.step(illegal)
        self.assertFalse(terminated); self.assertTrue(truncated); self.assertEqual(reward,-1.0)
        self.assertEqual(before,env.game.to_json()); self.assertEqual(info["invalid_action_count"],1)

    def test_gym_checker_mode_penalizes_illegal_action_without_ending_episode(self):
        env=TwentyFiveEnv(4,controlled_seat=0,illegal_action_mode="penalize"); obs,_=env.reset(seed=8)
        illegal=next(i for i,value in enumerate(obs["action_mask"]) if not value)
        before=env.game.to_json()
        _,reward,terminated,truncated,info=env.step(illegal)
        self.assertFalse(terminated); self.assertFalse(truncated); self.assertEqual(reward,-1.0)
        self.assertEqual(before,env.game.to_json()); self.assertEqual(info["invalid_action_count"],1)

    def test_aec_full_match_terminal_rewards_every_player(self):
        env=TwentyFiveAECEnv(3,reward="terminal"); env.reset(seed=12); rng=random.Random(12); final_nonzero=set()
        while env.agents:
            obs,reward,terminated,truncated,_=env.last()
            if terminated or truncated:
                final_nonzero.update(a for a,v in env.rewards.items() if v!=0)
                env.step(None)
            else:
                action=rng.choice([i for i,v in enumerate(obs["action_mask"]) if v])
                env.step(action)
        self.assertEqual(final_nonzero,set(env.possible_agents)); self.assertTrue(all(env.terminations.values()))

    def test_aec_last_returns_accumulated_reward_since_agents_previous_turn(self):
        env=TwentyFiveAECEnv(3,reward="score_difference"); env.reset(seed=22); rng=random.Random(22)
        for _ in range(100):
            obs,reward,terminated,truncated,_=env.last()
            if terminated or truncated:
                break
            agent=env.agent_selection
            action=rng.choice([i for i,v in enumerate(obs["action_mask"]) if v])
            env.step(action)
            self.assertEqual(env._cumulative_rewards[agent],env.rewards[agent])

    def test_aec_rob_action(self):
        env=TwentyFiveAECEnv(4); env.reset(seed=1,options={"initial_deck":rob_deck(2)})
        if env.game.phase is Phase.ROB:
            obs=env.observe(env.agent_selection)
            self.assertTrue(0<sum(obs["action_mask"])<6)


if __name__=="__main__": unittest.main()
