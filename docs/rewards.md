# Training rewards

The approved card-game objective uses `terminal` and gamma=1 by default. For N players,
expected terminal utility is `(N*p(win)-1)/(N-1)`, so maximizing it maximizes match win
probability. A lower gamma in same-seat decision time introduces a duration preference.
Dense alternatives below remain different objectives; zero-sum event arithmetic alone
does not establish policy-invariant shaping.

Rewards are environment configuration, never game rules. Every scheme is zero-sum across players.

- `terminal`: the winner receives `+1`; every loser receives `-1/(N-1)`.
- `score_difference`: each five-point trick award is scaled by 25 and centered by the table mean.
- `terminal_plus_score`: adds the terminal outcome to the dense score-difference signal.

The former `trick`, `score`, and `potential` names were removed. Immediate engine scoring made trick and score rewards scalar versions of the same event, and the old “potential” reward was not a policy-invariant `gamma * Phi(s') - Phi(s)` construction.

PettingZoo accumulates each player's event rewards until that player acts again. The Gym adapter sums the controlled player's rewards while opponents advance. The engine always rejects illegal actions atomically. Framework adapters truncate an externally supplied illegal-action episode with a penalty so official API checkers may sample the full discrete action space; built-in trainers use strict mode and raise immediately because their masks should make an illegal action impossible.
