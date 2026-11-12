# Deferred research roadmap

These features are intentionally postponed until the versioned observation, environment, reward, baseline, and evaluation contracts pass locally.

## Search

- Information-set Monte Carlo tree search with hidden-card determinizations consistent with public play.
- Search-generated demonstrations for behavioral cloning.
- Search diagnostics for rob-discard and end-game decisions.

Ordinary perfect-information MCTS must not be used because it would inspect hidden hands.

## Learning

- Recurrent PPO with per-player memory across deals.
- Centralized critic during training with a private actor at inference.
- Recurrent or distributional DQN.
- Auxiliary prediction of void suits, trick winner, and remaining trumps.
- Behavioral cloning from counting/search policies.

## Scale and leagues

- Multiprocess environment collectors.
- Batched inference for frozen neural opponents.
- Persistent multi-population league with adaptive matchup sampling.
- Multiple-seed sweep orchestration.

## Product and observability

- Richer dashboards after metric definitions stop changing.
- Spectator analysis, calibrated action explanations, and web play.
- Long-term checkpoint registry and artifact retention policies.

The next acceptance gate is a clean full test run with PyTorch, Gymnasium, and PettingZoo, followed by fresh smoke training under the new schema.

