# Temporal-difference control

Q-learning uses `Q←Q+α[r+γ max_a'Q(s',a')−Q]`; SARSA replaces the maximum with the sampled next action; Expected SARSA uses the epsilon-greedy expectation. Bootstrapping always occurs at the learner’s next actual decision after opponent roll-forward, and maxima/expectations use only masked legal actions.

State, rewards, epsilon schedule, and JSON checkpoint match Monte Carlo. Smoke: `python -m twenty_five.training.tabular_twentyfive --algorithm q_learning --episodes 5`; substitute `sarsa` or `expected_sarsa`. Longer: `--episodes 100000`. TD learns online with lower target delay but inherits abstraction aliasing; off-policy maximization can overestimate, while SARSA reflects exploratory risk. Compare all variants on held-out duplicate deals.

