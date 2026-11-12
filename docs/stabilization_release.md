# Stabilization release 0.3.0

This release rebuilds the complete layer between ruleset version 3 and learned agents.

## Breaking changes

- Observation schema version 1 replaces the previous anonymous 935-value vector.
- Old `.npz`, unversioned JSON, and pre-schema PyTorch checkpoints no longer load.
- Rewards are now `terminal`, `score_difference`, and `terminal_plus_score`.
- Policy-gradient variants are `reinforce`, `reinforce_baseline`, and `n_step_actor_critic`.
- Evaluation uses a fixed candidate-independent suite selected by `--suite`.
- PettingZoo uses standard cumulative rewards and dead-agent steps.

## Training behavior

- Tabular, policy-gradient, and DQN learners rotate seats.
- PPO self-play learns from every current-policy seat and updates only after complete matches.
- PPO persists its historical pool and random state at update boundaries.
- DQN stores compressed replay next to the main checkpoint by default.
- The card-aware PPO configuration is isolated under `configs/experimental/`.

## Acceptance commands

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[tracking]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m twenty_five.training.run configs\ppo_smoke.json
.\.venv\Scripts\python.exe -m twenty_five.training.dqn --episodes 4 --players 2 --num-envs 2 --batch-size 4 --replay-capacity 32 --warmup-steps 4 --eval-games 0 --checkpoint-interval 0 --device cuda --output artifacts\checkpoints\dqn_acceptance.pt
.\.venv\Scripts\python.exe -m twenty_five.training.dqn --episodes 1 --players 2 --num-envs 1 --batch-size 4 --replay-capacity 32 --warmup-steps 4 --eval-games 0 --checkpoint-interval 0 --device cuda --resume artifacts\checkpoints\dqn_acceptance.pt --output artifacts\checkpoints\dqn_acceptance.pt
```

Expected: all tests pass, Gymnasium and PettingZoo checker tests are not skipped, PPO reports `shared_policy_all_seats: true` when self-play is enabled, and the resumed DQN reports `replay_restored: 1`.
