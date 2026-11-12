# Game engine

`TwentyFiveGame` is an atomic transition engine. The caller supplies `(current_player, card_index)` and the engine validates ownership and legality before mutation. It advances the seat, closes a complete trick, immediately awards five points, and either continues play, rotates dealer after five tricks, or enters `TERMINAL` as soon as the trick winner reaches 25.

Phases are `ROB`, `PLAY`, and `TERMINAL`. The same stable 0–51 card index is used in both actionable phases. `legal_cards()` and `legal_action_mask()` are generated from the same rule function. Invalid actions raise `InvalidAction` and leave serialization byte-for-byte unchanged.

Full state includes hands, stock, physical turned zone, private discard, current ordered trick, completed ordered tricks, trick winners, scores, dealer/leader/actor, turned card/trump, rob metadata, histories, winner, and RNG state. Public state reveals that a rob occurred but never the face-down card. The robber’s private observation retains that knowledge. Match history archives every naturally completed deal and the terminal partial deal. `clone()`, `to_dict()/from_dict()`, and JSON equivalents preserve behavior.

Serialized states carry explicit state-format and ruleset versions and are validated whenever loaded. Validation covers physical card ownership, player and trick order, legal follow behavior, computed winners, trick counts, score accounting, rob zones and discard legality, phase-dependent hand sizes, dealer rotation, history totals, and terminal consistency.

`initial_deck` is a deterministic setup hook for conformance tests. It is draw order, must contain every card exactly once, and does not alter the rules.

## Prototype discrepancies corrected

- Prototype robbing was dealer-only, silently exchanged away the ace, and offered no discard action.
- Turn advancement and round reset were caller-managed; the new engine owns both.
- Invalid moves ended RL episodes and could be exploited; they now fail atomically.
- A♥ was sometimes treated as ordinary Hearts for follow-suit logic; category is now authoritative.
- One-at-a-time dealing was replaced by the dealer-relative three-then-two packet deal.
- Rob discards are restricted to non-trumps and hidden from opponents.
- Scores and terminal checks occur on every trick; no post-hoc crossing tie rule remains.
- Deferred rewards could be lost at round/game boundaries; wrappers use immediate owner-attributed events.
- Trick/player history lacked full ordered identities and boundaries; both are preserved.
- Observations assumed a four-card trick and leaked structure for variable players; padding now supports ten.
- Global shuffling was unseeded and RNG state was absent from serialization.
- State loading is versioned and always validated, independent of the runtime debug setting.
