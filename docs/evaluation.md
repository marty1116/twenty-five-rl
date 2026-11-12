# Evaluation protocol

`python -m twenty_five.training.evaluate` writes JSON, CSV, and Markdown reports.

## Main leaderboard

The main leaderboard uses a named, candidate-independent opponent suite. Adding or removing a candidate therefore cannot change another candidate's opponents.

- `--suite smoke`: random and lowest-card opponents.
- `--suite baseline`: random, greedy, conservative, and public-counting opponents.

`--games N` means exactly N attempted trials per policy. Every target receives the
same deal, target-seat, and initial-dealer schedule. Failures/invalid actions are
recorded and count as non-wins in the attempted-trial rate; they invalidate comparison
eligibility. Score/return metrics use completed matches only, and are null when none
complete. Inspect attempted/completed/failed counts before ranking policies.

Deterministic additional seeds are derived when needed. Use complete N² seat/dealer
blocks for balanced evidence. Intervals bootstrap whole duplicate-deal clusters when
at least two exist; reports explicitly label the Wilson fallback for smaller samples.
Neither interval substitutes for uncertainty across training seeds.

`--policy-mode both` (default) produces separate greedy and sampled checkpoint rows.
Actor sampling uses masked softmax and evaluation-owned randomness. Q/tabular sampling
uses `--evaluation-epsilon` (default .05); tabular greedy ties are uniform in sampled
mode. Greedy mode uses the first maximizing legal action. These are different policies.

## Separate diagnostics

Candidate-vs-candidate results do not feed the main leaderboard:

- The multiplayer matrix places one focal policy against a homogeneous table of another candidate.
- The two-player head-to-head matrix is explicitly a different player-count experiment.
- Bradley-Terry ratings are derived only from that two-player matrix and are labeled accordingly.

Failed diagnostic games are recorded and excluded from those matchup denominators;
undefined cells are null. Never use them to repair a failed main leaderboard row.

Match, deal, and trick units are kept distinct. Trump strata report average trick share in deals with each trump, rather than repeating the final match result. Rob statistics report the number of decisions and match win rate among matches containing a rob.

Use fixed validation games during training and a larger untouched benchmark seed file for final reporting. Compare at least three independently trained seeds.

