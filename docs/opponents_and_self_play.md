# Opponents and self-play

Baselines consume only `PlayerView` and form a deliberate ladder:

1. `random`: uniform legal action.
2. `lowest`: weakest contextual legal card.
3. `highest`: strongest contextual legal card.
4. `greedy`: wins the current trick as cheaply as possible, but leads strength.
5. `conservative`: preserves strength and wins cheaply.
6. `counting`: uses public played cards, inferred voids, remaining trump threats, score, and trick position.

Rob decisions use the separate hand-relative discard heuristic. It protects special trumps, values singleton void creation, and may discard a card other than the nominally weakest rank.

With PPO `--self-play`, every current-policy seat contributes training transitions. Fixed and historical opponents can replace a configurable fraction of seats through `--league-opponent-probability`. Complete matches finish before an update, preventing unfinished decisions from crossing policy versions. Historical snapshots and sampling RNG state are stored in PPO checkpoints.

DQN and the simpler policy-gradient controls rotate one controlled learner seat per episode. They do not claim all-seat self-play.

A persistent multi-population league and adaptive opponent matchmaking are deferred.

The repaired counting policy permits trump ruffing while followers remain: a trump
play does not establish an ordinary-suit void. A robbed turned trump remains a possible
opponent holding until otherwise accounted for. Historical results against the earlier
counting implementation are a different opponent version and must not be overwritten.

Mixtures are fixed unless explicitly configured otherwise; `ppo_curriculum.json` is
a historical filename for a fixed mixture. Seat-local shared-policy learning is not
an equilibrium/exploitability method. The retained feed-forward state does not expose
hidden opponent identity or preserve prior-deal evidence about persistent types.

