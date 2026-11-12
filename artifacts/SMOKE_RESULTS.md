# Stabilization smoke results

These checks validate infrastructure and baseline behavior; they are not model-quality claims.

## Dependency-light verification

```text
PYTHONPATH=src python -m unittest tests.test_engine tests.test_rules tests.test_properties tests.test_envs tests.test_agent_contracts -q
Ran 41 tests - OK
```

Coverage includes every player count, packet dealing, rob privacy, scoring and termination, serialization, observation-schema layout, hidden-hand invariance, AEC cumulative rewards, Gym opponent roll-forward, Expected SARSA RNG behavior, tabular checkpoint continuation, and candidate-independent evaluation.

## Fresh baseline board

The current board contains no learned checkpoints. Each baseline completes exactly 80 matches against the fixed `baseline` suite. Candidate multiplayer and two-player matrices are separate. All old neural/tabular artifacts were moved under `artifacts/legacy/`.

Full discovery in the dependency-light workspace reports 45 tests with 41 passing and four dependency-gated skips. The PyTorch trainer smoke tests and official Gymnasium/PettingZoo checker tests must be rerun in an environment containing the declared project dependencies.
