# GridWorld Monte Carlo

This four-by-four task validates generic episode generation before card-game state aliasing is introduced. An epsilon-greedy policy reaches the opposite corner; terminal reward is 1 and each other move is −0.02. Returns are `G_t=Σ_k γ^k r_{t+k+1}`. First-visit updates use `Q←Q+(G−Q)/N`; `--every-visit` updates repeated pairs too.

State is the grid coordinate; all four actions are legal. Epsilon decays to 0.05; evaluation uses zero exploration. The generic tabular JSON format stores alpha, gamma, Q rows, visit counts, and RNG state. Smoke: `python -m twenty_five.training.gridworld_mc --episodes 20`. Longer: `--episodes 5000 --seed 7`. Strengths are transparent convergence diagnostics; failures are short training, unlucky exploration, or too-small horizons. It is only a utility validation and is not compared numerically with Twenty-Five learners.

The training cap (`--max-steps`, default 200) is nonterminal truncation, not a timed
goal condition. Its tail bootstraps from the current epsilon-greedy expected Q value;
true goal termination has zero tail. Coordinate state needs no remaining-time feature
under this continuing-task interpretation. This is a truncated-return approximation,
not pure complete-episode Monte Carlo when the cutoff is reached. This diagnostic
retains gamma=.98 independently of the card-game gamma=1 default.
