"""Batched multi-environment Double DQN with efficient prioritized replay."""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import statistics
import time
from collections import deque
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from torch.nn import functional as F

from .. import rules
from ..agents.neural import FrozenQPolicy, MLPQNetwork, resolve_device, seed_torch
from ..envs.gym_env import TwentyFiveEnv
from ..observations import NEURAL_OBSERVATION_SIZE, OBSERVATION_SCHEMA_VERSION
from ..policies import Policy, make_policy
from ..rewards import REWARD_SCHEMES
from .common_eval import policy_win_rate
from .curves import write_curve
from .experiment import MetricLogger, prune_snapshots, write_manifest
from .replay import ReplayBuffer
from .protocol import training_positions
from .protocol import source_identity
from .state import capture_rng, restore_rng, resolved_arguments, effective_config

RawTransition: TypeAlias = tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


def _make_env(players: int, learner_seat: int, pool: list[Policy], rng: random.Random, reward: str) -> TwentyFiveEnv:
    opponents = {seat: rng.choice(pool) for seat in range(players) if seat != learner_seat} if pool else None
    return TwentyFiveEnv(
        players, controlled_seat=learner_seat, opponents=opponents, reward=reward,
        validate=False, illegal_action_mode="raise",
    )


def _flush_n_step(
    pending: deque[RawTransition],
    replay: ReplayBuffer,
    n_step: int,
    gamma: float,
    force: bool,
) -> None:
    while pending and (force or len(pending) >= n_step):
        state, action = pending[0][0], pending[0][1]
        total_reward = 0.0
        used = 0
        next_state = pending[0][3]
        done = False
        next_mask = pending[0][5]
        for index, transition in enumerate(pending):
            if index >= n_step:
                break
            total_reward += (gamma**index) * transition[2]
            next_state, done, next_mask = transition[3], transition[4], transition[5]
            used += 1
            if done:
                break
        replay.add(state, action, total_reward, next_state, done, next_mask, gamma**used)
        pending.popleft()
        if not force and len(pending) < n_step:
            break


def _training_state(
    optimizer: torch.optim.Optimizer,
    episodes: int,
    environment_steps: int,
    optimizer_steps: int,
    best_win_rate: float,
    *, target=None, action_rng=None, opponent_rng=None, replay_rng=None,
) -> dict[str, object]:
    return {
        "optimizer_state_dict": optimizer.state_dict(),
        "episodes": episodes,
        "environment_steps": environment_steps,
        "optimizer_steps": optimizer_steps,
        "best_win_rate": best_win_rate,
        **capture_rng(),
        'target_state_dict': target.state_dict() if target is not None else None,
        'action_rng_state': action_rng.getstate() if action_rng is not None else None,
        'opponent_rng_state': opponent_rng.getstate() if opponent_rng is not None else None,
        'replay_rng_state': replay_rng.bit_generator.state if replay_rng is not None else None,
    }


def train(
    episodes: int,
    seed: int,
    output: Path,
    players: int = 4,
    dueling: bool = False,
    device: str = "auto",
    batch_size: int = 64,
    *,
    num_envs: int = 4,
    replay_capacity: int = 10_000,
    warmup_steps: int = 1_000,
    train_frequency: int = 4,
    gradient_steps: int = 1,
    target_interval: int = 500,
    n_step: int = 1,
    prioritized_replay: bool = True,
    opponent_names: list[str] | None = None,
    reward: str = "terminal",
    learning_rate: float = 5e-4,
    resume: Path | None = None,
    eval_interval: int = 500,
    eval_games: int = 32,
    eval_opponent: str = "conservative",
    checkpoint_interval: int = 1_000,
    keep_snapshots: int = 5,
    tensorboard_dir: Path | None = None,
    log_interval: int = 100,
    phase_heads: bool = True,
    save_replay: bool = True,
    gamma: float = 1.0,
    collect_only: bool = False,
    resume_mode: str = 'approximate',
    override_learning_rate: bool = False,
    per_beta_steps: int = 200_000,
) -> dict[str, float | int | str]:
    requested = resolved_arguments(train, locals())
    if episodes < 1 or num_envs < 1 or batch_size < 1 or n_step < 1 or train_frequency < 1 or target_interval < 1:
        raise ValueError("episodes, environments, batch size, n-step, train frequency, and target interval must be positive")
    if checkpoint_interval > 0 and keep_snapshots < 1:
        raise ValueError("keep-snapshots must be positive when periodic checkpoints are enabled")
    if gradient_steps < 0 or (not collect_only and gradient_steps < 1):
        raise ValueError('positive gradient_steps required unless collect_only=True')
    if not 0 <= gamma <= 1 or per_beta_steps < 1 or warmup_steps < 0:
        raise ValueError('invalid gamma, warmup or PER beta horizon')
    if resume_mode not in {'approximate', 'warm_start'}:
        raise ValueError('DQN supports approximate continuation or warm_start; live collectors are not serialized')
    started_at = time.time()
    compute = resolve_device(device)
    seed_torch(seed)
    action_rng = random.Random(seed)
    opponent_rng = random.Random(seed ^ 0x5A5A)
    replay_rng = np.random.default_rng(seed)

    online = (
        MLPQNetwork.load(resume, compute)
        if resume
        else MLPQNetwork(NEURAL_OBSERVATION_SIZE, 128, seed, dueling, phase_heads).to(compute)
    )
    target = MLPQNetwork(
        NEURAL_OBSERVATION_SIZE,
        online.hidden,
        seed + 1,
        online.dueling,
        online.phase_heads,
    ).to(compute)
    target.copy_from(online)
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    completed_total = 0
    environment_steps = 0
    optimizer_steps = 0
    best_win_rate = -1.0
    state = {}
    replay_contract = {**source_identity(), 'gamma': gamma, 'reward': reward, 'players': players, 'n_step': n_step}
    if resume and resume_mode == 'approximate':
        payload = torch.load(resume, map_location=compute, weights_only=True)
        if payload.get("metadata", {}).get("ruleset_version") != rules.RULESET_VERSION:
            raise ValueError("resume checkpoint was trained under an incompatible or unversioned ruleset")
        state = payload.get("training_state", {})
        if payload.get('metadata', {}).get('trainer_version') != 2:
            raise ValueError('legacy DQN training state requires resume_mode="warm_start"')
        if payload.get('metadata', {}).get('data_contract') != replay_contract:
            raise ValueError('DQN continuation requires matching source/reward/discount/player/n-step contract; use warm_start')
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        completed_total = int(state.get("episodes", 0))
        environment_steps = int(state.get("environment_steps", 0))
        optimizer_steps = int(state.get("optimizer_steps", 0))
        best_win_rate = float(state.get("best_win_rate", -1.0))
        if state.get('target_state_dict') is not None:
            target.load_state_dict(state['target_state_dict'])
        if state.get('action_rng_state') is not None:
            action_rng.setstate(state['action_rng_state'])
            opponent_rng.setstate(state['opponent_rng_state'])
            replay_rng.bit_generator.state = state['replay_rng_state']
        restore_rng(state, compute)
        if override_learning_rate:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

    pool = [make_policy(name) for name in (opponent_names or [])]
    envs = [_make_env(players, env_id % players, pool, opponent_rng, reward) for env_id in range(num_envs)]
    per_env_episodes = [0 for _ in envs]
    episode_returns = [0.0 for _ in envs]
    pending = [deque() for _ in envs]
    next_episode_id = completed_total

    def reset_env(env_id: int) -> dict[str, np.ndarray]:
        nonlocal next_episode_id
        episode_number = next_episode_id
        next_episode_id += 1
        learner_seat, dealer = training_positions(episode_number, players)
        envs[env_id] = _make_env(players, learner_seat, pool, opponent_rng, reward)
        observation, _ = envs[env_id].reset(
            seed=seed + episode_number,
            options={"dealer": dealer},
        )
        return observation

    observations = [reset_env(env_id) for env_id in range(num_envs)]
    replay = ReplayBuffer(replay_capacity, NEURAL_OBSERVATION_SIZE)
    replay_restored = False
    if resume and resume_mode == 'approximate':
        replay_path = resume.with_suffix(".replay.npz")
        if replay_path.exists():
            replay, replay_metadata = ReplayBuffer.load(replay_path)
            if replay.states.shape[1] != NEURAL_OBSERVATION_SIZE:
                raise ValueError("replay sidecar uses an incompatible observation schema")
            expected = {"episodes": completed_total, "environment_steps": environment_steps}
            if any(replay_metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("replay sidecar does not match the resumed checkpoint counters")
            if replay_metadata.get('checkpoint_sha256') != hashlib.sha256(resume.read_bytes()).hexdigest() or replay_metadata.get('contract') != replay_contract:
                raise ValueError('replay source/configuration/digest mismatch; use an explicit warm start')
            replay_restored = True
    if not collect_only and replay.capacity < max(batch_size, warmup_steps):
        raise ValueError('replay capacity cannot reach batch/warmup readiness; use collect_only=True if intentional')
    losses: list[float] = []
    returns: list[float] = []
    config = {
        "ruleset_version": rules.RULESET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "episodes": episodes,
        "seed": seed,
        "players": players,
        "dueling": online.dueling,
        "phase_heads": online.phase_heads,
        "device": str(compute),
        "batch_size": batch_size,
        "num_envs": num_envs,
        "replay_capacity": replay.capacity,
        "warmup_steps": warmup_steps,
        "train_frequency": train_frequency,
        "gradient_steps": gradient_steps,
        "target_interval": target_interval,
        "n_step": n_step,
        "prioritized_replay": prioritized_replay,
        "opponent_names": opponent_names or [],
        "reward": reward,
        "learning_rate": learning_rate,
        "save_replay": save_replay,
        "resume_semantics": "main checkpoint restores compressed replay when present; live environments restart",
    }
    config = effective_config(requested, optimizer, **{**config,
        'data_contract': replay_contract,
        'multi_step_semantics': 'one-step Double DQN' if n_step == 1 else 'uncorrected behavior-tail approximation',
        'resume_semantics': 'approximate: target/optimizer/RNG/replay retained; live environments and pending prefixes restart'})
    def training_state():
        return _training_state(optimizer, completed_total, environment_steps, optimizer_steps, best_win_rate,
                               target=target, action_rng=action_rng, opponent_rng=opponent_rng, replay_rng=replay_rng)
    logger = MetricLogger(output, tensorboard_dir, append=bool(resume and resume_mode == 'approximate'))
    next_log_episode = completed_total + max(1, log_interval)
    next_eval_episode = completed_total + max(1, eval_interval)
    next_checkpoint_episode = completed_total + max(1, checkpoint_interval)
    previous_train_boundary = environment_steps // max(1, train_frequency)

    try:
        completed_this_run = 0
        while completed_this_run < episodes:
            state_arrays = np.stack([observation["observation"] for observation in observations])
            mask_arrays = np.stack([observation["action_mask"] for observation in observations])
            states = torch.as_tensor(state_arrays, dtype=torch.float32, device=compute)
            masks = torch.as_tensor(mask_arrays, dtype=torch.bool, device=compute)
            with torch.no_grad():
                greedy_actions = online(states).masked_fill(~masks, torch.finfo(torch.float32).min).argmax(dim=-1).cpu().tolist()

            actions: list[int] = []
            epsilon = max(0.05, 1.0 - environment_steps / 50_000)
            for env_id in range(num_envs):
                legal = np.flatnonzero(mask_arrays[env_id])
                action = int(action_rng.choice(list(legal))) if action_rng.random() < epsilon else greedy_actions[env_id]
                actions.append(action)

            for env_id, env in enumerate(envs):
                if completed_this_run >= episodes:
                    break
                nxt, step_reward, done, truncated, _ = env.step(actions[env_id])
                if truncated:
                    raise RuntimeError('strict card training unexpectedly truncated')
                transition: RawTransition = (
                    state_arrays[env_id],
                    actions[env_id],
                    step_reward,
                    nxt["observation"],
                    done,
                    nxt["action_mask"],
                )
                pending[env_id].append(transition)
                _flush_n_step(pending[env_id], replay, n_step, gamma, force=done)
                episode_returns[env_id] += step_reward
                environment_steps += 1
                observations[env_id] = nxt
                if done:
                    returns.append(episode_returns[env_id])
                    episode_returns[env_id] = 0.0
                    per_env_episodes[env_id] += 1
                    completed_this_run += 1
                    completed_total += 1
                    if completed_this_run < episodes:
                        observations[env_id] = reset_env(env_id)

            train_boundary = environment_steps // max(1, train_frequency)
            if not collect_only and len(replay) >= max(batch_size, warmup_steps) and train_boundary > previous_train_boundary:
                for _ in range(gradient_steps * (train_boundary - previous_train_boundary)):
                    fraction = min(1.0, optimizer_steps / per_beta_steps)
                    beta = 0.4 + 0.6 * fraction if prioritized_replay else 0.0
                    alpha = 0.6 if prioritized_replay else 0.0
                    batch = replay.sample(batch_size, replay_rng, alpha=alpha, beta=beta)
                    batch_states = torch.as_tensor(batch.states, dtype=torch.float32, device=compute)
                    batch_actions = torch.as_tensor(batch.actions, dtype=torch.long, device=compute)
                    batch_rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=compute)
                    next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=compute)
                    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=compute)
                    next_masks = torch.as_tensor(batch.next_masks, dtype=torch.bool, device=compute)
                    discounts = torch.as_tensor(batch.discounts, dtype=torch.float32, device=compute)
                    weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=compute)

                    predicted = online(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_online = online(next_states).masked_fill(~next_masks, torch.finfo(torch.float32).min)
                        next_actions = next_online.argmax(dim=1, keepdim=True)
                        next_values = target(next_states).gather(1, next_actions).squeeze(1)
                        expected = batch_rewards + discounts * (1.0 - dones) * next_values
                    temporal_errors = expected - predicted
                    loss = (F.smooth_l1_loss(predicted, expected, reduction="none") * weights).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    optimizer.step()
                    optimizer_steps += 1
                    losses.append(float(loss.detach().cpu()))
                    if prioritized_replay:
                        replay.update_priorities(batch.indices, temporal_errors.detach().cpu().numpy())
                    if optimizer_steps % target_interval == 0:
                        target.copy_from(online)
            previous_train_boundary = train_boundary

            metrics: dict[str, float | int | str | bool] = {
                "train/epsilon": epsilon,
                "train/mean_loss_100": statistics.fmean(losses[-100:]) if losses else 0.0,
                "rollout/mean_return_100": statistics.fmean(returns[-100:]) if returns else 0.0,
                "rollout/completed_episodes": completed_total,
                "performance/environment_steps": environment_steps,
                "performance/optimizer_steps": optimizer_steps,
                "replay/size": len(replay),
            }

            if eval_games > 0 and (completed_total >= next_eval_episode or completed_this_run >= episodes):
                win_rate = policy_win_rate(FrozenQPolicy(online), players, seed, eval_games, eval_opponent)
                metrics[f"evaluation/win_rate_vs_{eval_opponent}"] = win_rate
                while next_eval_episode <= completed_total:
                    next_eval_episode += max(1, eval_interval)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_path = output.with_name(f"{output.stem}.best{output.suffix}")
                    online.save(
                        best_path,
                        training_state=training_state(),
                        metadata=config,
                    )
            if checkpoint_interval > 0 and completed_total >= next_checkpoint_episode:
                snapshot = output.with_name(f"{output.stem}.step-{completed_total:08d}{output.suffix}")
                online.save(
                    snapshot,
                    training_state=training_state(),
                    metadata=config,
                )
                prune_snapshots(output.parent, output.stem, keep_snapshots)
                while next_checkpoint_episode <= completed_total:
                    next_checkpoint_episode += max(1, checkpoint_interval)
            if completed_total >= next_log_episode or completed_this_run >= episodes:
                logger.log(completed_total, **metrics)
                print(json.dumps({"episode": completed_total, **metrics}), flush=True)
                while next_log_episode <= completed_total:
                    next_log_episode += max(1, log_interval)
    finally:
        logger.close()

    online.save(
        output,
        training_state=training_state(),
        metadata=config,
    )
    if save_replay:
        replay.save(
            output.with_suffix(".replay.npz"),
            metadata={"episodes": completed_total, "environment_steps": environment_steps,
                      'checkpoint_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'contract': replay_contract},
        )
    write_curve(output, returns, "episodic_return")
    smoke_win_rate = policy_win_rate(FrozenQPolicy(online), players, seed, 16, "random")
    summary: dict[str, float | int | str] = {
        "algorithm": "double_dqn",
        "device": str(compute),
        "episodes": completed_total,
        "episodes_this_run": len(returns),
        "environment_steps": environment_steps,
        "optimizer_steps": optimizer_steps,
        "mean_return": float(np.mean(returns)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "replay_size": len(replay),
        "replay_restored": int(replay_restored),
        "parameters": sum(parameter.numel() for parameter in online.parameters()),
        "smoke_eval_win_rate": smoke_win_rate,
        "best_eval_win_rate": best_win_rate,
    }
    write_manifest(output, "double_dqn", config, started_at, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--shared-phase-head", action="store_false", dest="phase_heads", help="use one Q head for ROB and PLAY")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-interval", type=int, default=500)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--uniform-replay", action="store_false", dest="prioritized_replay")
    parser.add_argument("--opponent", action="append", default=[], choices=["random", "lowest", "highest", "greedy", "conservative", "counting"])
    parser.add_argument("--reward", choices=REWARD_SCHEMES, default="terminal")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=32)
    parser.add_argument("--eval-opponent", choices=["random", "lowest", "highest", "greedy", "conservative", "counting"], default="conservative")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--keep-snapshots", type=int, default=5)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-save-replay", action="store_false", dest="save_replay")
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--resume-mode', choices=['approximate','warm_start'], default='approximate')
    parser.add_argument('--override-learning-rate', action='store_true')
    parser.add_argument('--per-beta-steps', type=int, default=200_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/dqn.pt"))
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.episodes,
                args.seed,
                args.output,
                args.players,
                args.dueling,
                args.device,
                args.batch_size,
                num_envs=args.num_envs,
                replay_capacity=args.replay_capacity,
                warmup_steps=args.warmup_steps,
                train_frequency=args.train_frequency,
                gradient_steps=args.gradient_steps,
                target_interval=args.target_interval,
                n_step=args.n_step,
                prioritized_replay=args.prioritized_replay,
                opponent_names=args.opponent,
                reward=args.reward,
                learning_rate=args.learning_rate,
                resume=args.resume,
                eval_interval=args.eval_interval,
                eval_games=args.eval_games,
                eval_opponent=args.eval_opponent,
                checkpoint_interval=args.checkpoint_interval,
                keep_snapshots=args.keep_snapshots,
                tensorboard_dir=args.tensorboard_dir,
                log_interval=args.log_interval,
                phase_heads=args.phase_heads,
                save_replay=args.save_replay,
                gamma=args.gamma, collect_only=args.collect_only, resume_mode=args.resume_mode,
                override_learning_rate=args.override_learning_rate, per_beta_steps=args.per_beta_steps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
