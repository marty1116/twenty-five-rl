# Research workflow

Run commands from the repository root using `.venv/Scripts/python.exe` on Windows.
The primary benchmark is four-player Twenty-Five under the current agreed rules.
Two-player matchups are separate diagnostics, not substitutes for that benchmark.

The [approved decisions](../verification/repairs/decisions.md) now freeze the repair-pass
rules and win-probability objective. Read [current acceptance evidence](../verification/repairs/README.md)
before launching experiments. Historical outputs and the original audit/baseline are
preserved; current research configurations write to new v2 roots.

## Gate 1: human rules acceptance

Run `python -m twenty_five.training.rules_lab`. Review each hand, lead, expected
legal set, and actual legal set in `configs/research/rules_cases.json`.
The expected sets are hand-authored; passing means agreement with those expectations,
not human approval. Add cases when a rule is clarified.

Run `python -m twenty_five.gui` for private manual play, or add `--seed 11` for an
explicitly revealing debug session. Review dealing order,
mandatory rob (including turned ace), hidden discards, trick winners, dealer changes,
and immediate termination at 25. Save completed replays and annotate the decisions
you checked. The user accepted the six-trump fallback and existing ruff/ROB conventions
for this repair pass on 2026-10-26. Additional manual acceptance cases remain useful,
but do not block these agreed correctness repairs.

Run `python -m unittest discover -s tests -v`. This includes known numerical TD
targets, a terminal-choice learning diagnostic for all tabular algorithms, privacy,
serialization, environment contracts, and seat/dealer coverage. Neural smoke runs
are not convergence tests. The repair harness adds independent gradient/estimator checks,
three-seed known-optimum neural diagnostics and exact CPU continuation comparisons.
Those bounded tasks do not establish card-playing strength.

## Gate 2: smoke experiments

`python -m twenty_five.training.research configs/research/smoke.json`

The default prints the complete plan without training or writing outputs.
Add `--execute` to train MC, SARSA, Q-learning, Expected SARSA, DQN, and PPO and
evaluate each final checkpoint. Existing run directories are rejected, never overwritten.
Choose a new output_root for a new trial; interrupted runs remain inspectable.

Every run stores its exact configuration, package versions, source/rules SHA-256,
checkpoint, training summary, and validation JSON/CSV/Markdown tables. Source hashes
distinguish the recent rule correction from earlier ruleset-3 artifacts even without Git.
Old checkpoints are historical; the existing generic loaders still accept version-3
files. The research runner starts fresh and refuses later evaluation if source changed.

## Gate 3: pilot study

After rules acceptance and smoke verification:

`python -m twenty_five.training.research configs/research/pilot.json --execute`

Three independent seeds use the same fixed opponent mixture, terminal reward, and
independently balanced learner/dealer positions. Gamma is 1; default actor entropy
coefficient is zero. These are feasibility budgets, not
equal-compute comparisons. Tabular and DQN use 2,000 matches; PPO uses 100 updates.
Record runtime and learner decisions before choosing equal sample budgets for the paper.
PPO counts all card actions while DQN counts learner decisions: do not compare their
environment_steps columns directly. Pilot terminal evaluations supplement existing
trainer curves; periodic common-suite evaluation for tabular models remains future work.
Evaluation reports greedy and sampled learned policies separately. Any failed/invalid
trial invalidates comparison eligibility and remains in the attempted-trial denominator.

Validation uses a negative seed namespace starting at -10^12; final test uses -2*10^12.
Training starts from small positive seeds. Evaluation expands seeds using the common
schedule; for these bounded configs the namespaces are disjoint. All candidates within
a run use the same evaluation schedule and fixed baseline suite. Rotations are dependent:
report deal-cluster uncertainty and training-seed variation separately. Internal trainer
evaluations are tuning diagnostics; they do not touch the final-test namespace.

## Gate 4: targeted PPO study

Freeze baseline results, then test terminal versus shaped rewards and one representation
change independently. Use validation to select settings. Keep all seed results, including
failures. Do not interpret shaped return as win probability. Check deterministic versus
sampled inference before deciding which policy is deployed. Self-play comes afterward.

Copy the pilot config to define the selected final experiments, increase evaluation games
to a multiple of 16 based on pilot uncertainty, and use at least five training seeds when
feasible. Predeclare budgets and the selection rule. All research-runner evaluations use
the final checkpoint; internal best snapshots are not silently substituted.

## Gate 5: untouched final test and paper

After freezing choices, evaluate already trained runs explicitly:

`python -m twenty_five.training.research configs/research/pilot.json --evaluate-only --split test --execute`

Use the selected final config in place of pilot.json for the actual paper. Do not tune
against this output. Report per-seed results, paired comparisons against heuristics,
runtime, sample counts, and limitations of the lossy tabular state. Existing historical
results informed development and are not an untouched test set for future experiments.

Deliver a short paper with question, rules and observation contract, methods, experimental
protocol, results, ablations, failure analysis, and limitations. Add reproduction commands,
small representative artifacts and a playable demo. Improvements over random, evidence
about representations, and explained negative results are all valuable outcomes.
