"""Masked PPO with complete-episode, shared-policy multiplayer collection."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import rules
from ..agents.neural import (
    ActorCriticModel,
    CardAwareActorCritic,
    FrozenActorPolicy,
    LinearActorCritic,
    load_actor_critic,
    resolve_device,
    seed_torch,
)
from ..envs.aec_env import TwentyFiveAECEnv
from ..observations import NEURAL_OBSERVATION_SIZE, OBSERVATION_SCHEMA_VERSION
from ..policies import FrozenCheckpointPolicy, Policy, make_policy
from ..rewards import REWARD_SCHEMES
from .common_eval import policy_win_rate
from .curves import write_curve
from .experiment import MetricLogger, TerminalProgress, prune_snapshots, write_manifest
from .protocol import training_positions
from .losses import compute_gae, ppo_objective
from .state import (capture_rng, restore_rng, resolved_arguments, continuation_contract,
                    validate_continuation, effective_config)


@dataclass
class PendingDecision:
    state: np.ndarray
    mask: np.ndarray
    action: int
    value: float
    log_probability: float
    reward: float = 0.0


@dataclass(frozen=True)
class RolloutRow:
    env_id: int
    episode_id: int
    seat: int
    state: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    done: bool
    value: float
    next_value: float
    log_probability: float


@dataclass
class Collector:
    env: TwentyFiveAECEnv
    episode_id: int
    learner_seats: set[int]
    opponents: dict[int, Policy]
    pending: dict[int, PendingDecision]
    returns: list[float]
    length: int = 0


def _historical_states(pool: list[FrozenActorPolicy]) -> list[dict[str, torch.Tensor]]:
    return [
        {name: value.detach().cpu() for name, value in policy.model.state_dict().items()}
        for policy in pool
    ]


def _training_state(
    optimizer: torch.optim.Optimizer,
    update: int,
    environment_steps: int,
    environment_resets: int,
    completed_episodes: int,
    best_win_rate: float,
    chooser: random.Random,
    historical_pool: list[FrozenActorPolicy],
    optimizer_steps: int = 0,
) -> dict[str, object]:
    return {
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "environment_steps": environment_steps,
        "environment_resets": environment_resets,
        "completed_episodes": completed_episodes,
        "best_win_rate": best_win_rate,
        **capture_rng(),
        "chooser_rng_state": chooser.getstate(),
        "historical_state_dicts": _historical_states(historical_pool),
        'optimizer_steps': optimizer_steps,
    }


def _restore_historical(
    model: ActorCriticModel,
    states: list[dict[str, torch.Tensor]],
) -> list[FrozenActorPolicy]:
    restored: list[FrozenActorPolicy] = []
    for state in states:
        snapshot = copy.deepcopy(model).to("cpu")
        snapshot.load_state_dict(state)
        restored.append(FrozenActorPolicy(snapshot))
    return restored


def train(
    updates: int,
    seed: int,
    output: Path,
    players: int = 4,
    num_envs: int = 2,
    horizon: int = 32,
    self_play: bool = False,
    resume: Path | None = None,
    opponent_checkpoints: list[Path] | None = None,
    opponent_weights: list[float] | None = None,
    device: str = "auto",
    batch_size: int = 128,
    *,
    opponent_names: list[str] | None = None,
    reward: str = "terminal",
    learning_rate: float = 2.5e-4,
    epochs: int = 3,
    phase_heads: bool = True,
    architecture: str = "mlp",
    self_play_snapshot_interval: int = 25,
    self_play_pool_size: int = 8,
    league_opponent_probability: float = 0.25,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    entropy_coefficient: float = 0.0,
    value_coefficient: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = 0.03,
    eval_interval: int = 50,
    eval_games: int = 32,
    eval_opponent: str = "conservative",
    checkpoint_interval: int = 100,
    keep_snapshots: int = 5,
    tensorboard_dir: Path | None = None,
    log_interval: int = 10,
    progress: bool = True,
    resume_mode: str = 'continue',
    override_learning_rate: bool = False,
) -> dict[str, float | int | str | bool]:
    requested = resolved_arguments(train, locals())
    if resume_mode not in {'continue', 'warm_start'}:
        raise ValueError('resume_mode must be continue or warm_start')
    if updates < 1 or num_envs < 1 or horizon < 1 or batch_size < 1 or epochs < 1:
        raise ValueError("updates, environments, horizon, batch size, and epochs must be positive")
    if self_play_pool_size < 1 or self_play_snapshot_interval < 1:
        raise ValueError("self-play pool size and snapshot interval must be positive")
    if not 0.0 <= league_opponent_probability <= 1.0:
        raise ValueError("league opponent probability must be in [0, 1]")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and GAE lambda must be in [0, 1]")
    if clip_range <= 0 or max_grad_norm <= 0:
        raise ValueError("clip range and max gradient norm must be positive")
    if eval_games > 0 and eval_interval < 1:
        raise ValueError("eval interval must be positive when evaluation is enabled")
    if checkpoint_interval > 0 and keep_snapshots < 1:
        raise ValueError("keep-snapshots must be positive when periodic checkpoints are enabled")
    if self_play and opponent_weights is not None:
        raise ValueError("fixed opponent weights are unsupported when the historical self-play pool changes")
    if opponent_weights is not None and (any(w < 0 or not np.isfinite(w) for w in opponent_weights) or sum(opponent_weights) <= 0):
        raise ValueError('opponent weights must be finite, nonnegative and have positive total')
    if target_kl is not None and target_kl <= 0:
        raise ValueError('target_kl must be positive or None')

    started_at = time.time()
    compute = resolve_device(device)
    seed_torch(seed)
    chooser = random.Random(seed)
    if architecture not in {"mlp", "card_aware"}:
        raise ValueError("architecture must be 'mlp' or 'card_aware'")
    if resume:
        model = load_actor_critic(resume, compute)
        architecture = "card_aware" if isinstance(model, CardAwareActorCritic) else "mlp"
    elif architecture == "card_aware":
        model = CardAwareActorCritic(NEURAL_OBSERVATION_SIZE, seed, phase_heads=phase_heads).to(compute)
    else:
        model = LinearActorCritic(NEURAL_OBSERVATION_SIZE, seed, phase_heads=phase_heads).to(compute)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_update = 0
    environment_steps = 0
    environment_resets = 0
    completed_episodes_total = 0
    best_win_rate = -1.0
    historical_pool: list[FrozenActorPolicy] = []
    state = {}
    contract = continuation_contract(requested, compute)
    if resume and resume_mode == 'continue':
        payload = torch.load(resume, map_location=compute, weights_only=True)
        validate_continuation(payload, contract)
        state = payload.get("training_state", {})
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state.get("update", 0))
        environment_steps = int(state.get("environment_steps", 0))
        environment_resets = int(state.get("environment_resets", 0))
        completed_episodes_total = int(state.get("completed_episodes", 0))
        best_win_rate = float(state.get("best_win_rate", -1.0))
        if "chooser_rng_state" in state:
            chooser.setstate(state["chooser_rng_state"])
        historical_pool = _restore_historical(model, list(state.get("historical_state_dicts", [])))

    fixed_pool: list[Policy] = [make_policy(name) for name in (opponent_names or [])]
    fixed_pool.extend(FrozenCheckpointPolicy(path) for path in (opponent_checkpoints or []))
    if opponent_weights is not None and len(opponent_weights) != len(fixed_pool):
        raise ValueError("--opponent-weights must provide one value for every fixed opponent")
    # Loading a fixed neural opponent constructs a model and can reseed torch.
    # Restore all generator streams only after every model/history construction.
    if state:
        restore_rng(state, compute)
        if override_learning_rate:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
    else:
        seed_torch(seed)

    def choose_opponent(candidates: list[Policy]) -> Policy:
        if not candidates:
            return make_policy("random")
        if opponent_weights is not None and not historical_pool:
            return chooser.choices(candidates, weights=opponent_weights, k=1)[0]
        return chooser.choice(candidates)

    def reset_collector(env_id: int) -> Collector:
        nonlocal environment_resets
        reset_index = environment_resets
        environment_resets += 1
        env = TwentyFiveAECEnv(players, reward=reward, validate=False, illegal_action_mode="raise")
        learner_seat, dealer = training_positions(reset_index, players)
        env.reset(seed=seed + reset_index, options={"dealer": dealer})
        if self_play:
            candidates = [*fixed_pool, *historical_pool]
            learner_seats = {
                seat for seat in range(players)
                if not candidates or chooser.random() >= league_opponent_probability
            }
            if not learner_seats:
                learner_seats.add(reset_index % players)
        else:
            learner_seats = {learner_seat}
            candidates = fixed_pool
        opponents = {
            seat: choose_opponent(candidates)
            for seat in range(players)
            if seat not in learner_seats
        }
        return Collector(env, reset_index, learner_seats, opponents, {}, [0.0] * players)

    collectors = [reset_collector(env_id) for env_id in range(num_envs)]
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    all_value_losses: list[float] = []
    config = {
        "ruleset_version": rules.RULESET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "updates": updates,
        "seed": seed,
        "players": players,
        "num_envs": num_envs,
        "target_transitions_per_update": num_envs * horizon,
        "complete_episode_batches": True,
        "self_play": self_play,
        "shared_policy_all_seats": self_play,
        "opponent_names": opponent_names or [],
        "opponent_checkpoints": [str(path) for path in (opponent_checkpoints or [])],
        "reward": reward,
        "device": str(compute),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "entropy_coefficient": entropy_coefficient,
        "value_coefficient": value_coefficient,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "phase_heads": model.phase_heads,
        "architecture": architecture,
        "league_opponent_probability": league_opponent_probability,
        "resume_semantics": "exact at completed PPO update boundaries, including historical pool and RNG",
    }
    config = effective_config(requested, optimizer, **{**config, 'continuation_contract': contract,
        'advantage_normalization': 'none', 'policy_weighting': 'episode sums averaged over learner trajectories',
        'resume_semantics': 'same-contract completed PPO boundaries; warm_start resets training state'})
    logger = MetricLogger(output, tensorboard_dir, append=bool(resume and resume_mode == 'continue'))
    progress_bar = TerminalProgress(
        start_update + updates,
        initial=start_update,
        label="PPO",
        enabled=progress,
    )
    latest_eval: float | None = None
    optimizer_steps = int(state.get('optimizer_steps', 0))

    try:
        for local_update in range(1, updates + 1):
            update = start_update + local_update
            update_started = time.perf_counter()
            for env_id, collector in enumerate(collectors):
                game = collector.env.game
                if game is not None and game.terminal:
                    collectors[env_id] = reset_collector(env_id)
            target_rows = num_envs * horizon
            rows: list[RolloutRow] = []
            paused: set[int] = set()

            while len(paused) < num_envs:
                learner_turns: list[tuple[int, int, np.ndarray, np.ndarray]] = []
                actions_by_env: dict[int, int] = {}
                for env_id, collector in enumerate(collectors):
                    if env_id in paused:
                        continue
                    game = collector.env.game
                    assert game is not None
                    seat = game.current_player
                    if seat in collector.learner_seats:
                        view = collector.env.raw_view(collector.env.possible_agents[seat])
                        learner_turns.append((env_id, seat, view.neural, view.action_mask))
                    else:
                        view = collector.env.raw_view(collector.env.possible_agents[seat])
                        actions_by_env[env_id] = collector.opponents[seat].act(view, chooser)

                if learner_turns:
                    state_arrays = np.stack([item[2] for item in learner_turns])
                    mask_arrays = np.stack([item[3] for item in learner_turns])
                    states = torch.as_tensor(state_arrays, dtype=torch.float32, device=compute)
                    masks = torch.as_tensor(mask_arrays, dtype=torch.bool, device=compute)
                    with torch.no_grad():
                        logits, values = model.policy_and_value(states, masks)
                        distribution = torch.distributions.Categorical(logits=logits)
                        sampled = distribution.sample()
                        log_probabilities = distribution.log_prob(sampled)
                    sampled_actions = sampled.cpu().tolist()
                    sampled_values = values.cpu().tolist()
                    sampled_log_probabilities = log_probabilities.cpu().tolist()
                    for index, (env_id, seat, state_array, mask_array) in enumerate(learner_turns):
                        collector = collectors[env_id]
                        if seat in collector.pending:
                            previous = collector.pending.pop(seat)
                            rows.append(
                                RolloutRow(
                                    env_id, collector.episode_id, seat, previous.state, previous.mask, previous.action,
                                    previous.reward, False, previous.value, sampled_values[index],
                                    previous.log_probability,
                                )
                            )
                        action = sampled_actions[index]
                        actions_by_env[env_id] = action
                        collector.pending[seat] = PendingDecision(
                            state_array, mask_array, action, sampled_values[index],
                            sampled_log_probabilities[index],
                        )

                for env_id, collector in enumerate(collectors):
                    if env_id in paused:
                        continue
                    collector.env.step(actions_by_env[env_id])
                    environment_steps += 1
                    collector.length += 1
                    for seat, pending in collector.pending.items():
                        delivered = collector.env.rewards[collector.env.possible_agents[seat]]
                        pending.reward += delivered
                    for seat in range(players):
                        collector.returns[seat] += collector.env.rewards[collector.env.possible_agents[seat]]

                    game = collector.env.game
                    assert game is not None
                    if game.terminal:
                        for seat, pending in list(collector.pending.items()):
                            rows.append(
                                RolloutRow(
                                    env_id, collector.episode_id, seat, pending.state, pending.mask, pending.action,
                                    pending.reward, True, pending.value, 0.0, pending.log_probability,
                                )
                            )
                        collector.pending.clear()
                        completed_episodes_total += 1
                        completed_lengths.append(collector.length)
                        completed_returns.extend(collector.returns[seat] for seat in collector.learner_seats)
                        if len(rows) >= target_rows:
                            paused.add(env_id)
                        else:
                            collectors[env_id] = reset_collector(env_id)

            advantages, return_targets = compute_gae(rows, gamma, gae_lambda)
            raw_advantages = advantages.copy()
            trajectories = {(r.env_id, r.episode_id, r.seat) for r in rows}
            row_times = {}
            policy_weights = []
            for row in rows:
                key = (row.env_id, row.episode_id, row.seat)
                k = row_times.get(key, 0)
                policy_weights.append(gamma**k * len(rows) / len(trajectories))
                row_times[key] = k + 1

            states = torch.as_tensor(np.stack([row.state for row in rows]), dtype=torch.float32, device=compute)
            legal_masks = torch.as_tensor(np.stack([row.mask for row in rows]), dtype=torch.bool, device=compute)
            actions = torch.as_tensor([row.action for row in rows], dtype=torch.long, device=compute)
            old_log_probabilities = torch.as_tensor(
                [row.log_probability for row in rows], dtype=torch.float32, device=compute
            )
            advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=compute)
            return_tensor = torch.as_tensor(return_targets, dtype=torch.float32, device=compute)
            policy_weight_tensor = torch.as_tensor(policy_weights, dtype=torch.float32, device=compute)

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropy_values: list[float] = []
            kl_values: list[float] = []
            clip_fractions: list[float] = []
            stop_for_kl = False
            steps_before_update = optimizer_steps
            for _ in range(epochs):
                order = torch.randperm(len(rows), device=compute)
                for start in range(0, len(rows), batch_size):
                    indices = order[start : start + batch_size]
                    logits, values = model.policy_and_value(states[indices], legal_masks[indices])
                    loss, policy_loss, value_loss, entropy, log_ratio, ratio = ppo_objective(
                        logits, values, actions[indices], old_log_probabilities[indices],
                        advantage_tensor[indices], return_tensor[indices], clip_range, value_coefficient,
                        entropy_coefficient, policy_weight_tensor[indices])
                    approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach().cpu())
                    kl_values.append(approximate_kl)
                    if target_kl is not None and approximate_kl > target_kl:
                        stop_for_kl = True
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer_steps += 1
                    policy_losses.append(float(policy_loss.detach().cpu()))
                    value_losses.append(float(value_loss.detach().cpu()))
                    entropy_values.append(float(entropy.detach().cpu()))
                    clip_fractions.append(
                        float((torch.abs(ratio - 1.0) > clip_range).float().mean().detach().cpu())
                    )
                if stop_for_kl:
                    break
            all_value_losses.extend(value_losses)

            if self_play and update % self_play_snapshot_interval == 0:
                historical_pool.append(FrozenActorPolicy(model))
                historical_pool[:] = historical_pool[-self_play_pool_size:]

            return_variance = float(np.var(return_targets))
            explained_variance = (
                1.0 - float(np.var(raw_advantages)) / return_variance if return_variance > 1e-8 else 0.0
            )
            duration = max(time.perf_counter() - update_started, 1e-9)
            metrics: dict[str, float | int | str | bool] = {
                "train/policy_loss": statistics.fmean(policy_losses) if policy_losses else 0.0,
                "train/value_loss": statistics.fmean(value_losses) if value_losses else 0.0,
                "train/entropy": statistics.fmean(entropy_values) if entropy_values else 0.0,
                "train/approx_kl": statistics.fmean(kl_values),
                "train/clip_fraction": statistics.fmean(clip_fractions) if clip_fractions else 0.0,
                'train/optimizer_steps': optimizer_steps,
                'train/optimizer_steps_this_update': optimizer_steps - steps_before_update,
                "train/explained_variance": explained_variance,
                "train/stopped_for_target_kl": stop_for_kl,
                "rollout/transitions": len(rows),
                "rollout/mean_return_100": statistics.fmean(completed_returns[-100:]) if completed_returns else 0.0,
                "rollout/mean_length_100": statistics.fmean(completed_lengths[-100:]) if completed_lengths else 0.0,
                "rollout/completed_episodes_total": completed_episodes_total,
                "performance/training_transitions_per_second": len(rows) / duration,
                "performance/environment_steps": environment_steps,
                "self_play/history_pool_size": len(historical_pool),
            }

            if eval_games > 0 and (update % eval_interval == 0 or local_update == updates):
                win_rate = policy_win_rate(FrozenActorPolicy(model), players, seed, eval_games, eval_opponent)
                latest_eval = win_rate
                metrics[f"evaluation/win_rate_vs_{eval_opponent}"] = win_rate
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_path = output.with_name(f"{output.stem}.best{output.suffix}")
                    model.save(
                        best_path,
                        training_state=_training_state(
                            optimizer, update, environment_steps, environment_resets,
                            completed_episodes_total, best_win_rate, chooser, historical_pool, optimizer_steps,
                        ),
                        metadata=config,
                    )
            logger.log(update, **metrics)

            if checkpoint_interval > 0 and update % checkpoint_interval == 0:
                snapshot = output.with_name(f"{output.stem}.step-{update:08d}{output.suffix}")
                model.save(
                    snapshot,
                    training_state=_training_state(
                        optimizer, update, environment_steps, environment_resets,
                        completed_episodes_total, best_win_rate, chooser, historical_pool, optimizer_steps,
                    ),
                    metadata=config,
                )
                prune_snapshots(output.parent, output.stem, keep_snapshots)
            if log_interval > 0 and (update % log_interval == 0 or local_update == updates):
                print(json.dumps({"update": update, **metrics}), flush=True)
            progress_bar.update(
                update,
                episodes=completed_episodes_total,
                speed=float(metrics["performance/training_transitions_per_second"]),
                evaluation=latest_eval,
            )
    finally:
        progress_bar.close()
        logger.close()

    final_update = start_update + updates
    model.save(
        output,
        training_state=_training_state(
            optimizer, final_update, environment_steps, environment_resets,
            completed_episodes_total, best_win_rate, chooser, historical_pool, optimizer_steps,
        ),
        metadata=config,
    )
    write_curve(output, completed_returns, "episodic_return")
    smoke_win_rate = policy_win_rate(FrozenActorPolicy(model), players, seed, 16, "random")
    summary: dict[str, float | int | str | bool] = {
        "algorithm": "ppo",
        "device": str(compute),
        "updates": final_update,
        "updates_this_run": updates,
        'optimizer_steps': optimizer_steps,
        "vector_envs": num_envs,
        "environment_steps": environment_steps,
        "environment_resets": environment_resets,
        "completed_episodes": completed_episodes_total,
        "mean_terminal_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "mean_value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
        "self_play": self_play,
        "shared_policy_all_seats": self_play,
        "opponent_pool_size": len(fixed_pool) + len(historical_pool),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "smoke_eval_win_rate": smoke_win_rate,
        "best_eval_win_rate": best_win_rate,
    }
    write_manifest(output, "ppo", config, started_at, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--opponent", action="append", default=[],
        choices=["random", "lowest", "highest", "greedy", "conservative", "counting"],
    )
    parser.add_argument("--opponent-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--opponent-weights", type=float, nargs="+")
    parser.add_argument("--reward", choices=REWARD_SCHEMES, default="terminal")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--shared-phase-head", action="store_false", dest="phase_heads")
    parser.add_argument("--architecture", choices=["mlp", "card_aware"], default="mlp")
    parser.add_argument("--self-play-snapshot-interval", type=int, default=25)
    parser.add_argument("--self-play-pool-size", type=int, default=8)
    parser.add_argument("--league-opponent-probability", type=float, default=0.25)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=32)
    parser.add_argument(
        "--eval-opponent",
        choices=["random", "lowest", "highest", "greedy", "conservative", "counting"],
        default="conservative",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--keep-snapshots", type=int, default=5)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument('--resume-mode', choices=['continue', 'warm_start'], default='continue')
    parser.add_argument('--override-learning-rate', action='store_true')
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/ppo.pt"))
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.updates, args.seed, args.output, args.players, args.num_envs, args.horizon,
                args.self_play, args.resume, args.opponent_checkpoint, args.opponent_weights,
                args.device, args.batch_size, opponent_names=args.opponent, reward=args.reward,
                learning_rate=args.learning_rate, epochs=args.epochs, phase_heads=args.phase_heads,
                architecture=args.architecture,
                self_play_snapshot_interval=args.self_play_snapshot_interval,
                self_play_pool_size=args.self_play_pool_size,
                league_opponent_probability=args.league_opponent_probability,
                gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
                entropy_coefficient=args.entropy_coefficient, value_coefficient=args.value_coefficient,
                max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                eval_interval=args.eval_interval, eval_games=args.eval_games,
                eval_opponent=args.eval_opponent, checkpoint_interval=args.checkpoint_interval,
                keep_snapshots=args.keep_snapshots, tensorboard_dir=args.tensorboard_dir,
                log_interval=args.log_interval, progress=args.progress, resume_mode=args.resume_mode,
                override_learning_rate=args.override_learning_rate,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
