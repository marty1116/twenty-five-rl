# Deep Q network

The DQN learner uses masked Double DQN, optional dueling heads, separate ROB/PLAY heads,
and optional prioritized replay. Defaults are terminal reward, gamma=1 and one-step
backups. Configuring n>1 gives an uncorrected exploratory behavior-tail approximation,
not exact multi-step greedy-policy evaluation.

Collection rotates the learner through every seat and batches action selection across environments. Replay observations are quantized from the schema's known `[0,1]` range to `uint8`, keeping the 4,067-value safe observation affordable. Values are restored to float32 when sampled.

Prioritized records are drawn with replacement. Weights are `(M*p_i)^(-beta)`, normalized
by the maximum over the entire buffer, and beta approaches 1 on a fixed optimizer-step
schedule. At beta=1 this preserves uniform-record expected update direction up to a
buffer-dependent constant. Duplicate updates use maximum absolute TD error. Replay
weighting does not correct intermediate behavior actions for n>1.

The main checkpoint writes a compressed `.replay.npz` sidecar by default. Restoration
checks source/reward/discount/player/n-step identity, counters and checkpoint digest.
Target lag, optimizer and generator states also restore, but live games and pending
prefixes restart: `--resume-mode approximate` is explicit. `warm_start` starts a new run
from weights; use it for legacy compatible checkpoints and changed contracts. Saved
optimizer LR is preserved unless `--override-learning-rate` is requested. Numbered/best
snapshots have no replay sidecars. Missing replay is reported, not treated as exact continuation.

Replay capacity must reach batch/warmup readiness and gradient steps must be positive.
Use `--collect-only` when deliberately collecting without optimization; a saved
checkpoint alone is not evidence that training occurred.

```powershell
.\.venv\Scripts\python.exe -m twenty_five.training.run configs\dqn_baseline.json
```

DQN is an off-policy comparison. Partial observability and changing opponents are difficult for feed-forward Q-learning, so recurrent/distributional DQN is deferred until this baseline shows repeatable improvement.
