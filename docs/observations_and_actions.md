# Observations and actions

Every action is a card index `suit * 13 + rank` in `[0, 51]`. In `PLAY` it plays that card. In `ROB` it chooses the mandatory face-down discard from the legal non-trumps, with the documented all-trump fallback.

## Player-safe source view

`PlayerView` is the authoritative agent information boundary. It contains the observing player's hand and legal cards plus public dealer, leader, actor, trump, turned card, scores, completed plays, current trick, robber identity, and terminal state. Only the robber receives `own_rob_discard`; no other view or public state exposes that card.

Policies never receive a `TwentyFiveGame` object.

## Observation schema version 1

The neural encoding is generated only by `ObservationLayout` and its named slices. It has 4,067 values in `[0, 1]`:

- own hand, public played cards, turned card, and own private rob discard;
- trump, phase, active-player mask, and rob state;
- relative dealer, leader, current actor, and robber;
- relative scores and deal trick counts;
- ten current-trick slots and fifty completed-play slots.

A slot contains a 53-position card encoding (padding plus 52 physical cards) and a ten-position relative-seat encoding. Card identity is never reconstructed by rounding a normalized scalar. The legal-action mask is returned separately and is not duplicated inside the state vector.

All neural and tabular checkpoints record the observation schema version. Unversioned or incompatible checkpoints are rejected.

The compact tabular key remains deliberately lossy: phase, trump, trick context, score gap, own tricks, and quantized hand strengths. Tabular results are controls, not expected state-of-the-art agents.

## Framework contracts

`TwentyFiveAECEnv` is the canonical multiplayer adapter and follows PettingZoo AEC cumulative-reward and dead-agent semantics. `TwentyFiveEnv` is a Gymnasium single-controlled-seat adapter over that AEC environment; it advances configured opponents until the learner's next decision.

Player-facing info schema **2** uses recipient-specific event dictionaries. A ROB
action is null for observers other than the actor; phase, actor and public outcome
remain available. Own-discard knowledge stays in the robber's view. The neural
observation schema remains 1. Consumers of the former Transition-valued info field
must use the new dictionary contract.

Normal GUI sessions omit live seeds; explicit seeds select revealing debug mode and
are labeled accordingly. Completed replay/state diagnostics intentionally reveal
the full game. Live CLI output hides AI ROB discards. Invalid opponent actions propagate
wrapper truncation and a zero action mask; reset is required before another step.

Run both official checkers after installing the project:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_experimental_contracts -v
```

