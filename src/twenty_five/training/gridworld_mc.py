"""Small generic Monte Carlo validation task."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..agents.tabular import TabularControl
from .curves import write_curve


class GridWorld:
    def __init__(self, size: int = 4, seed: int = 0):
        self.size, self.rng = size, random.Random(seed)
        self.state = (0, 0)
    def reset(self): self.state = (0, 0); return self.state
    def legal(self): return [0, 1, 2, 3]
    def step(self, action: int):
        y, x = self.state
        dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
        self.state = (min(self.size - 1, max(0, y + dy)), min(self.size - 1, max(0, x + dx)))
        done = self.state == (self.size - 1, self.size - 1)
        return self.state, (1.0 if done else -0.02), done


def train(episodes: int, seed: int, output: Path, first_visit: bool = True, max_steps: int = 200) -> dict[str, float | int | str]:
    if episodes < 1 or max_steps < 1:
        raise ValueError('episodes and max_steps must be positive')
    env, agent = GridWorld(seed=seed), TabularControl(gamma=0.98, seed=seed)
    lengths = []
    truncated_episodes = 0
    for episode in range(episodes):
        state = env.reset(); states = []; actions = []; rewards = []
        epsilon = max(0.05, 1.0 - episode / max(1, episodes * 0.8))
        for _ in range(max_steps):
            action = agent.act(state, env.legal(), epsilon)
            states.append(state); actions.append(action)
            state, reward, done = env.step(action); rewards.append(reward)
            if done: break
        bootstrap = 0.0
        if not done:
            truncated_episodes += 1
            values = [agent.value(state, a) for a in env.legal()]
            best = max(values)
            tied = values.count(best)
            bootstrap = sum((epsilon / 4 + ((1-epsilon)/tied if q == best else 0.0)) * q for q in values)
        agent.monte_carlo_update(states, actions, rewards, first_visit, bootstrap=bootstrap)
        lengths.append(len(actions))
    output.parent.mkdir(parents=True, exist_ok=True); agent.save(output)
    write_curve(output,[float(x) for x in lengths],"episode_length")
    eval_lengths = []
    for _ in range(30):
        state = env.reset()
        for step in range(1, 201):
            state, _, done = env.step(agent.act(state, env.legal(), 0.0))
            if done: break
        eval_lengths.append(step)
    return {"episodes": episodes, "mean_train_length": sum(lengths) / len(lengths), "mean_eval_length": sum(eval_lengths) / len(eval_lengths),
            'truncated_episodes': truncated_episodes, 'cutoff_semantics': 'nonterminal expected-policy tail bootstrap'}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--every-visit", action="store_true")
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/gridworld_mc.json")); args = parser.parse_args()
    print(json.dumps(train(args.episodes, args.seed, args.output, not args.every_visit, args.max_steps), indent=2))


if __name__ == "__main__": main()
