"""Masked REINFORCE and a correctly named n-step actor-critic baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from .protocol import training_positions

import numpy as np
import torch
from torch.nn import functional as F

from .. import rules
from ..agents.neural import LinearActorCritic, resolve_device, seed_torch
from ..agents.tabular import discounted_returns
from ..envs.gym_env import TwentyFiveEnv
from ..observations import NEURAL_OBSERVATION_SIZE, OBSERVATION_SCHEMA_VERSION
from ..policies import make_policy
from ..rewards import REWARD_SCHEMES
from .common_eval import checkpoint_win_rate
from .curves import write_curve
from .experiment import MetricLogger, write_manifest
from .state import (capture_rng, restore_rng, resolved_arguments, continuation_contract,
                    validate_continuation, effective_config)

VARIANTS = ("reinforce", "reinforce_baseline", "n_step_actor_critic")


def _update(
    model: LinearActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: list[np.ndarray],
    masks: list[np.ndarray],
    actions: list[int],
    returns: list[float],
    use_baseline: bool,
    device: torch.device,
    policy_weights: list[float] | None = None,
    entropy_coefficient: float = 0.0,
) -> float:
    states = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
    legal_masks = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)
    chosen_actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    logits, values = model.policy_and_value(states, legal_masks)
    distribution = torch.distributions.Categorical(logits=logits)
    advantages = return_tensor - values.detach() if use_baseline else return_tensor
    weights = torch.ones_like(advantages) if policy_weights is None else torch.as_tensor(policy_weights, device=device)
    # One episode is one sample: sum its score-function contributions, without
    # an action-dependent trajectory baseline or random inverse-length weight.
    policy_loss = -(distribution.log_prob(chosen_actions) * advantages * weights).sum()
    policy_loss -= entropy_coefficient * distribution.entropy().mean()
    value_loss = F.mse_loss(values, return_tensor) if use_baseline else torch.zeros((), device=device)
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return float(loss.detach().cpu())


def train(
    variant: str,
    episodes: int,
    seed: int,
    output: Path,
    players: int = 4,
    device: str = "auto",
    *,
    reward: str = "terminal",
    opponent_names: list[str] | None = None,
    tensorboard_dir: Path | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 32,
    resume: Path | None = None,
    gamma: float = 1.0,
    entropy_coefficient: float = 0.0,
    resume_mode: str = 'continue',
    override_learning_rate: bool = False,
) -> dict[str, float | int | str]:
    requested = resolved_arguments(train, locals())
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if episodes < 1 or n_steps < 1:
        raise ValueError("episodes and n_steps must be positive")
    if not 0 <= gamma <= 1 or resume_mode not in {'continue', 'warm_start'}:
        raise ValueError('invalid gamma or resume mode')
    started_at = time.time()
    compute = resolve_device(device)
    seed_torch(seed)
    rng = random.Random(seed)
    pool = [make_policy(name) for name in (opponent_names or [])]
    model = LinearActorCritic.load(resume, compute) if resume else LinearActorCritic(
        NEURAL_OBSERVATION_SIZE, seed, phase_heads=True
    ).to(compute)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    completed_before = 0
    contract = continuation_contract(requested, compute)
    if resume and resume_mode == 'continue':
        payload = torch.load(resume, map_location=compute, weights_only=True)
        validate_continuation(payload, contract)
        state = payload.get("training_state", {})
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        completed_before = int(state.get("episodes", 0))
        rng.setstate(state['opponent_rng_state'])
        restore_rng(state, compute)
        if override_learning_rate:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
    else:
        seed_torch(seed)

    totals: list[float] = []
    losses: list[float] = []
    logger = MetricLogger(output, tensorboard_dir, append=bool(resume and resume_mode == 'continue'))
    try:
        for local_episode in range(episodes):
            episode = completed_before + local_episode
            learner_seat, dealer = training_positions(episode, players)
            opponents = {seat: rng.choice(pool) for seat in range(players) if seat != learner_seat} if pool else None
            env = TwentyFiveEnv(
                players, controlled_seat=learner_seat, reward=reward,
                opponents=opponents, illegal_action_mode="raise",
            )
            observation, _ = env.reset(seed=seed + episode, options={"dealer": dealer})
            total = 0.0
            episode_observations: list[np.ndarray] = []
            episode_masks: list[np.ndarray] = []
            episode_actions: list[int] = []
            episode_rewards: list[float] = []
            done = False
            decision_index = 0
            while not done:
                chunk_observations: list[np.ndarray] = []
                chunk_masks: list[np.ndarray] = []
                chunk_actions: list[int] = []
                chunk_rewards: list[float] = []
                chunk_limit = n_steps if variant == "n_step_actor_critic" else None
                while True:
                    state = torch.as_tensor(observation["observation"], dtype=torch.float32, device=compute).unsqueeze(0)
                    mask = torch.as_tensor(observation["action_mask"], dtype=torch.bool, device=compute).unsqueeze(0)
                    with torch.no_grad():
                        action = int(model.distribution(state, mask).sample().item())
                    chunk_observations.append(observation["observation"])
                    chunk_masks.append(observation["action_mask"])
                    chunk_actions.append(action)
                    observation, step_reward, done, truncated, _ = env.step(action)
                    if truncated:
                        raise RuntimeError('strict card training unexpectedly truncated')
                    chunk_rewards.append(step_reward)
                    total += step_reward
                    if done or (chunk_limit is not None and len(chunk_rewards) >= chunk_limit):
                        break
                if variant == "n_step_actor_critic":
                    bootstrap = 0.0
                    if not done:
                        with torch.no_grad():
                            next_state = torch.as_tensor(
                                observation["observation"], dtype=torch.float32, device=compute
                            ).unsqueeze(0)
                            bootstrap = float(model.values(next_state).item())
                    chunk_returns = discounted_returns(chunk_rewards, gamma, bootstrap=bootstrap)
                    losses.append(
                        _update(model, optimizer, chunk_observations, chunk_masks, chunk_actions, chunk_returns, True, compute,
                                [gamma ** (decision_index + i) for i in range(len(chunk_rewards))], entropy_coefficient)
                    )
                    decision_index += len(chunk_rewards)
                else:
                    episode_observations.extend(chunk_observations)
                    episode_masks.extend(chunk_masks)
                    episode_actions.extend(chunk_actions)
                    episode_rewards.extend(chunk_rewards)
            if variant != "n_step_actor_critic":
                losses.append(
                    _update(
                        model,
                        optimizer,
                        episode_observations,
                        episode_masks,
                        episode_actions,
                        discounted_returns(episode_rewards, gamma),
                        variant == "reinforce_baseline",
                        compute,
                        [gamma ** i for i in range(len(episode_rewards))],
                        entropy_coefficient,
                    )
                )
            totals.append(total)
            logger.log(
                episode + 1,
                **{
                    "train/loss": losses[-1],
                    "rollout/episodic_return": totals[-1],
                    "rollout/learner_seat": learner_seat,
                },
            )
    finally:
        logger.close()

    completed_total = completed_before + episodes
    config = {
        "ruleset_version": rules.RULESET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "variant": variant,
        "episodes": completed_total,
        "seed": seed,
        "players": players,
        "device": str(compute),
        "reward": reward,
        "opponent_names": opponent_names or [],
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "resume_semantics": "weights_optimizer_and_torch_rng_at_episode_boundary",
    }
    config = effective_config(requested, optimizer, **{**config, 'gamma': gamma, 'entropy_coefficient': entropy_coefficient,
                              'continuation_contract': contract, 'resume_semantics': 'exact completed-episode continuation; warm_start resets state'})
    model.save(
        output,
        training_state={
            "optimizer_state_dict": optimizer.state_dict(),
            "episodes": completed_total,
            **capture_rng(),
            'opponent_rng_state': rng.getstate(),
        },
        metadata=config,
    )
    write_curve(output, totals, "episodic_return")
    summary: dict[str, float | int | str] = {
        "algorithm": variant,
        "device": str(compute),
        "episodes": completed_total,
        "episodes_this_run": episodes,
        "mean_return": float(np.mean(totals)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "smoke_eval_win_rate": checkpoint_win_rate(output, players, seed),
    }
    write_manifest(output, variant, config, started_at, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="reinforce_baseline")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reward", choices=REWARD_SCHEMES, default="terminal")
    parser.add_argument(
        "--opponent", action="append", default=[],
        choices=["random", "lowest", "highest", "greedy", "conservative", "counting"],
    )
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--entropy-coefficient', type=float, default=0.0)
    parser.add_argument('--resume-mode', choices=['continue', 'warm_start'], default='continue')
    parser.add_argument('--override-learning-rate', action='store_true')
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/policy_gradient.pt"))
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.variant, args.episodes, args.seed, args.output, args.players, args.device,
                reward=args.reward, opponent_names=args.opponent, tensorboard_dir=args.tensorboard_dir,
                learning_rate=args.learning_rate, n_steps=args.n_steps, resume=args.resume, gamma=args.gamma,
                entropy_coefficient=args.entropy_coefficient, resume_mode=args.resume_mode,
                override_learning_rate=args.override_learning_rate,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
