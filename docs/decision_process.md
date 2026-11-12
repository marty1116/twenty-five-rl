# Decision-process formalization

## Full simulator state

At decision time, the Markov state is

\[
s=(D,r,\phi,p,\ell,T,c,H,K,U,X,z,\rho,a_\rho,q,W,RNG),
\]

where `D` is dealer, `r` deal number, `φ` phase, `p` actor, `ℓ` leader, `T` trump/turned card, `c` scores and deal trick counts, `H` all private hands, `K` ordered current trick, `U` ordered completed-deal tricks, `X` stock/turned/private-discard locations, `z` match history, `ρ` robber, `aρ` private rob discard, `q` terminal flag/winner, and `RNG` the shuffle generator state. This is sufficient to sample the next state without earlier state.

## Player observation and information set

Player `i` sees its own hand; public turned card and trump; phase; ordered `(relative seat, card)` plays in the current trick and completed tricks; dealer, leader and acting seat relative to `i`; padded player mask; deal tricks and match scores; whether and by whom the turned card was robbed; and the legal mask. Only the robber observes the face-down card they discarded. Opponent hands, opponent rob-discard identity, and undealt stock identities are absent.

## Formal game

- State space `S`: every valid serialized engine state.
- Agent set: `I={0,…,N−1}` and acting function `ι:S→I` outside terminal states.
- Action space: `A={0,…,51}`; `A(s)` is the phase-specific mask.
- Transition `P(s'|s,a)`: deterministic within a dealt round; shuffle transitions use the serialized RNG and initial seed distribution.
- Observation `O_i:S→Ω_i`: the private-player mapping above.
- Reward: selectable ML signal; primary terminal vector is `+1` to the winner and `−1/(N−1)` to each loser.
- Initial distribution: seeded shuffled deck plus selected initial dealer/seat assignment.
- Terminal set: states immediately following a trick that raises its winner to at least 25.
- Discount: configurable `γ`; card-game defaults use `1` and terminal utility to maximize match win probability.
- Opponent policies `π_{−i}` map their own observations and legal masks to actions.

The complete simulator is a finite, turn-based stochastic game. An individual cannot observe the Markov state because opponents’ hands and stock are hidden, so its problem is a partially observable stochastic game—not an ordinary fully observed MDP. A belief over information-set states or a recurrent policy may summarize action-observation history.

Fixing opponent policies and seat assignment in the single-agent wrapper induces an MDP-like controlled process over the learner’s decision epochs: opponent moves are folded into one environment transition and the returned successor is the learner’s next real decision. The observation remains lossy, so recurrence may still help. Shared-policy self-play changes opponents during learning and remains a multi-agent learning problem.

Opponent identities can persist within a match without being observed, and encoded
history resets at each deal. Neither the neural vector nor the tabular key is assumed
Markov. Rewards inside a same-seat interval are summed without discount; configurable
gamma applies between those intervals. The default gamma=1 removes duration preference.
Shared weights receive independent seat-local improvements, not a joint-policy or
equilibrium gradient. See the [approved decision record](../verification/repairs/decisions.md).
