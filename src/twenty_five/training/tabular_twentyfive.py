"""Twenty-Five Monte Carlo, SARSA, Q-learning, and Expected SARSA CLI."""

from __future__ import annotations

import argparse
import ast
import json
import random
import time
from pathlib import Path

from .. import rules
from ..agents.tabular import TabularControl
from ..envs.gym_env import TwentyFiveEnv
from ..observations import OBSERVATION_SCHEMA_VERSION, tabular_state
from ..policies import make_policy
from ..rewards import REWARD_SCHEMES
from .common_eval import checkpoint_win_rate
from .curves import write_curve
from .experiment import write_manifest
from .protocol import training_positions
from .state import resolved_arguments, continuation_contract, effective_config


def _decision(env: TwentyFiveEnv) -> tuple[tuple[int, ...], list[int]]:
    assert env.game is not None and env.controlled_seat is not None
    state = tabular_state(env.game, env.controlled_seat)
    legal = [index for index, value in enumerate(env.game.legal_action_mask(env.controlled_seat)) if value]
    return state, legal


def train(
    algorithm: str,
    episodes: int,
    seed: int,
    output: Path,
    players: int = 4,
    *,
    reward: str = "terminal",
    opponent_names: list[str] | None = None,
    resume: Path | None = None,
    gamma: float = 1.0,
    alpha: float = 0.15,
    epsilon_decay_episodes: int = 20_000,
    resume_mode: str = 'continue',
) -> dict[str, float | int]:
    requested = resolved_arguments(train, locals())
    if algorithm not in {"mc", "sarsa", "q_learning", "expected_sarsa"}:
        raise ValueError("unknown tabular algorithm")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 <= gamma <= 1 or epsilon_decay_episodes < 1 or resume_mode not in {'continue', 'warm_start'}:
        raise ValueError('invalid gamma, exploration horizon or resume mode')
    started_at = time.time()
    opponent_rng = random.Random(seed)
    pool = [make_policy(name) for name in (opponent_names or [])]
    agent = TabularControl.load(resume) if resume else TabularControl(alpha=alpha, gamma=gamma, seed=seed)
    contract = continuation_contract(requested, 'cpu')
    completed_before = 0
    if resume and resume_mode == 'continue':
        if agent.metadata.get('continuation_contract') != contract or 'opponent_rng_state' not in agent.training_state:
            raise ValueError('exact tabular continuation requires matching source/runtime/settings; use resume_mode="warm_start"')
        opponent_rng.setstate(ast.literal_eval(agent.training_state['opponent_rng_state']))
        completed_before = agent.episodes
    elif resume:
        agent.counts.clear()
        agent.rng.seed(seed)
        agent.alpha, agent.gamma = alpha, gamma
    returns: list[float] = []

    for local_episode in range(episodes):
        episode = completed_before + local_episode
        learner_seat, dealer = training_positions(episode, players)
        opponents = (
            {seat: opponent_rng.choice(pool) for seat in range(players) if seat != learner_seat}
            if pool
            else None
        )
        env = TwentyFiveEnv(
            players,
            controlled_seat=learner_seat,
            opponents=opponents,
            reward=reward,
            illegal_action_mode="raise",
        )
        env.reset(seed=seed + episode, options={"dealer": dealer})
        states: list[tuple[int, ...]] = []
        actions: list[int] = []
        rewards: list[float] = []
        total = 0.0
        state, legal = _decision(env)
        epsilon = max(0.05, 0.8 * (1 - episode / epsilon_decay_episodes))
        action = agent.act(state, legal, epsilon)
        while True:
            _, step_reward, done, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError('strict card training unexpectedly truncated')
            total += step_reward
            states.append(state)
            actions.append(action)
            rewards.append(step_reward)
            if done:
                if algorithm != "mc":
                    agent.td_update(state, action, step_reward, None, [], algorithm, epsilon)
                break
            next_state, next_legal = _decision(env)
            next_action = agent.act(next_state, next_legal, epsilon)
            if algorithm != "mc":
                agent.td_update(
                    state,
                    action,
                    step_reward,
                    next_state,
                    next_legal,
                    algorithm,
                    epsilon,
                    next_action,
                )
            state, legal, action = next_state, next_legal, next_action
        if algorithm == "mc":
            agent.monte_carlo_update(states, actions, rewards, first_visit=True)
        returns.append(total)

    completed_total = completed_before + episodes
    config = effective_config(requested, continuation_contract=contract, gamma=agent.gamma, alpha=agent.alpha,
                              resume_semantics='exact completed-match continuation; fixed global episode exploration horizon')
    config.update({
        "ruleset_version": rules.RULESET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "episodes": completed_total,
        "seed": seed,
        "players": players,
        "reward": reward,
        "opponent_names": opponent_names or [],
        "resume": str(resume) if resume else None,
    })
    agent.save(output, episodes=completed_total, metadata=config,
               training_state={'opponent_rng_state': repr(opponent_rng.getstate())})
    write_curve(output, returns, "episodic_return")
    summary: dict[str, float | int | str] = {
        "algorithm": algorithm,
        "episodes": completed_total,
        "episodes_this_run": episodes,
        "mean_return": sum(returns) / len(returns),
        "states": len(agent.q),
        "smoke_eval_win_rate": checkpoint_win_rate(output, players, seed),
    }
    write_manifest(output, algorithm, config, started_at, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["mc", "sarsa", "q_learning", "expected_sarsa"],
        default="mc",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward", choices=REWARD_SCHEMES, default="terminal")
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        choices=["random", "lowest", "highest", "greedy", "conservative", "counting"],
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=.15)
    parser.add_argument('--epsilon-decay-episodes', type=int, default=20_000)
    parser.add_argument('--resume-mode', choices=['continue','warm_start'], default='continue')
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/tabular.json"))
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.algorithm,
                args.episodes,
                args.seed,
                args.output,
                args.players,
                reward=args.reward,
                opponent_names=args.opponent,
                resume=args.resume, gamma=args.gamma, alpha=args.alpha,
                epsilon_decay_episodes=args.epsilon_decay_episodes, resume_mode=args.resume_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
