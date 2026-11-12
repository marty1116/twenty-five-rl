# Twenty-Five Monte Carlo

Monte Carlo estimates action value from complete match returns and updates first visits by the sample mean. It uses the compact, lossy tabular key and legal masking, terminal zero-sum reward, a decaying epsilon-greedy behavior policy, rotating learner seats, and evaluation without exploration. Versioned JSON checkpoints store Q values, visit counts, RNG state, episode count, and hyperparameters.

Smoke: `python -m twenty_five.training.tabular_twentyfive --algorithm mc --episodes 5`. Longer: use `--episodes 100000 --seed 7`. Strengths are unbiased complete-return targets and simple debugging; weaknesses are high variance, long credit delay, and severe state aliasing. Failures include sparse state revisits and evaluating with exploration. Compare held-out win rate—not training return—with TD and neural approaches after real runs exist.
