# Stability and experimental boundary

The stable game surface is `cards.py`, `rules.py`, and `engine.py`, currently ruleset version 3.

The stabilized integration surface consists of:

- player-safe `PlayerView`;
- observation schema version 1;
- the PettingZoo AEC adapter;
- the Gymnasium controlled-seat adapter;
- explicit zero-sum reward definitions;
- fixed evaluation-suite semantics.

Checkpoint formats are versioned against ruleset and observation schema. Incompatible
or unversioned files are rejected; compatible historical weights remain inspectable
or usable as explicit warm starts. Exact continuation additionally enforces the current
source/runtime/settings contract. Player-facing infos use schema 2 independently of
observation schema 1.

Learning algorithms, learned parameters, dashboard presentation, search agents, card-aware attention, large-scale collectors, and league policy remain experimental. Their commands are retained for research, but their quality is not part of the engine contract. The card-aware configuration lives under `configs/experimental`.

Core verification:

```bash
python -m unittest tests.test_rules tests.test_engine tests.test_properties tests.test_agent_contracts -v
```

Full local verification, including PyTorch and official Gymnasium/PettingZoo checks:

```bash
python -B verification/repairs/run.py
```

See `docs/deferred_roadmap.md` for intentionally postponed work.
