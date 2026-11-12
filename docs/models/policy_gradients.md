# Policy-gradient controls

Three correctly named controls are available:

- `reinforce`: episodic Monte Carlo policy gradient.
- `reinforce_baseline`: episodic REINFORCE with a learned value baseline.
- `n_step_actor_critic`: bootstraps after configurable `--n-steps` chunks and updates during the episode.

The previous `a2c` label was removed because the implementation performed one episodic REINFORCE-style update and was not synchronous Advantage Actor-Critic.

All variants use masked categorical actions, entropy regularization, gradient clipping, rotating learner seats, and schema-versioned checkpoints.

Defaults use terminal rewards, gamma=1 and zero entropy coefficient. Episodic actor
losses sum raw detached advantages, without same-trajectory centering or division by
random trajectory length. REINFORCE uses raw returns. With gamma below 1, contributions
carry gamma^k in same-seat decision time. Critic targets stay unnormalized. Chunked
actor–critic uses its estimated nonterminal tail and updates within a match, so it is
an approximate bootstrapped control.

Complete-match continuation requires matching source/runtime/settings and restores
optimizer and generator states after opponent construction. Explicit warm starts
reset counters/optimizer. Requested and effective LR are recorded separately;
`--override-learning-rate` deliberately changes restored optimization.

```powershell
.\.venv\Scripts\python.exe -m twenty_five.training.policy_gradient --variant reinforce_baseline --episodes 100 --device cuda
.\.venv\Scripts\python.exe -m twenty_five.training.policy_gradient --variant n_step_actor_critic --n-steps 32 --episodes 100 --device cuda
```

These are diagnostic stepping stones. PPO remains the primary policy-gradient learner.

