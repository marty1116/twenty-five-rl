# PPO

PPO uses masked categorical actions, clipped policy ratios, GAE, a value loss, entropy regularization, gradient clipping, and optional target-KL early stopping. Gamma, GAE lambda, clipping range, loss coefficients, and maximum gradient norm are command-line parameters.

Defaults use terminal reward, gamma=1, GAE lambda=.95 and entropy coefficient=0.
Advantages are raw and detached, with no batch/trajectory centering. The actor surrogate
averages episode sums across learner trajectories; discounted experiments also use
gamma^k. Forced decisions contribute zero actor gradient and retain their value targets.
GAE with a learned critic remains an approximation under the reactive observation.
The target-KL monitor checks the pre-step estimate before applying that minibatch.
It does not bound the KL after a step or roll back earlier steps. Metrics count actual steps.

## Multiplayer collection

The collector batches neural decisions across games. Under `--self-play`, the current policy controls and learns from every current-policy seat, each using its own private `PlayerView`. Rewards are accumulated from one seat decision until that seat's next decision.

Updates happen only after all active matches finish. No pending action crosses a policy update, which keeps the rollout genuinely on-policy. Historical/fixed league opponents may occupy a configured fraction of seats.

Checkpoints include model, optimizer, counters, CPU/CUDA Torch RNG, Python/NumPy RNG,
opponent-selection RNG and historical model states. All model construction finishes
before generator restoration. `continue` enforces the same source, runtime, settings
and opponent content; CPU split-run tests include fixed neural opponents/history.
`warm_start` resets training state. Saved LR wins unless explicitly overridden.
CUDA generator persistence does not establish deterministic CUDA kernels.

Shared-policy self-play performs independent seat-local updates, not an equilibrium
method or the derivative of a tied joint-policy return.

## Architectures

`mlp` is the supported baseline. It uses named observation slices and separate ROB/PLAY heads.

`card_aware` is experimental. It uses explicit one-hot card slots, card/seat embeddings, attention over public play order, and candidate-card scoring. Its configuration is isolated under `configs/experimental`; do not use it to establish the first baseline.

Slot zero remains an available attention key, including when padded; this is the
retained sentinel convention. Current-deal history does not make observations Markov
or recover persistent hidden opponent types.

`configs/ppo_curriculum.json` is a fixed mixture despite its historical filename.
Select settings on validation matches; reserve held-out final tests after choices freeze.

