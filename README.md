# Twenty-Five RL

An RL agent for [Twenty-Five](https://en.wikipedia.org/wiki/Twenty-five_(card_game)), a four-player Irish trick-taking card game modeled as a POMDP — each player observes only their own hand, the trump card, and the trick history while choosing from a phase-dependent legal action set.
The agent uses PPO with logit-level action masking, GAE (γ=1, terminal rewards), and a curriculum of rule-based opponents.

## Dependencies

Python ≥ 3.10, PyTorch ≥ 2.7, NumPy ≥ 1.26, Gymnasium ≥ 1.0, PettingZoo ≥ 1.24.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# Train (PPO, 100 updates, 8 parallel envs)
python -m twenty_five.training.ppo --updates 100 --num-envs 8 --device auto

# Evaluate a checkpoint
python -m twenty_five.training.evaluate --checkpoint ppo=artifacts/checkpoints/ppo.best.pt --games 100
```
