"""Experimental opponent policies consuming player-safe engine views."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Protocol

from . import rules
from .cards import Card, Suit
from .engine import Phase
from .observations import OBSERVATION_SCHEMA_VERSION, PlayerView
from .valuation import discard_value, intrinsic_strength, order_legal_cards


PolicyView = PlayerView


class Policy(Protocol):
    name: str

    def act(self, view: PolicyView, rng: random.Random) -> int: ...


class RandomPolicy:
    name = "random"
    requires_neural = False
    requires_tabular = False
    def act(self, view: PolicyView, rng: random.Random) -> int:
        return rng.choice(view.legal).index


class LowestPolicy:
    name = "lowest"
    requires_neural = False
    requires_tabular = False
    def act(self, view: PolicyView, rng: random.Random) -> int:
        led = view.current_trick[0][1] if view.current_trick else None
        return order_legal_cards(list(view.legal), list(view.hand), view.trump, led)[0].index


class HighestPolicy:
    name = "highest"
    requires_neural = False
    requires_tabular = False
    def act(self, view: PolicyView, rng: random.Random) -> int:
        led = view.current_trick[0][1] if view.current_trick else None
        return order_legal_cards(list(view.legal), list(view.hand), view.trump, led)[-1].index


class GreedyPolicy:
    """Win the current trick as cheaply as possible; otherwise shed weakness."""

    name = "greedy"
    requires_neural = False
    requires_tabular = False

    def act(self, view: PolicyView, rng: random.Random) -> int:
        legal = list(view.legal)
        if view.phase is Phase.ROB:
            return min(legal, key=lambda card: (discard_value(card, list(view.hand), view.trump), card.index)).index
        if not view.current_trick:
            return max(legal, key=lambda card: (intrinsic_strength(card, view.trump), -card.index)).index
        led = view.current_trick[0][1]
        current_best = max(rules.trick_strength(card, view.trump, led) for _, card in view.current_trick)
        ordered = sorted(legal, key=lambda card: (rules.trick_strength(card, view.trump, led), card.index))
        winners = [card for card in ordered if rules.trick_strength(card, view.trump, led) > current_best]
        return (winners[0] if winners else ordered[0]).index


class ConservativePolicy:
    name = "conservative"
    requires_neural = False
    requires_tabular = False
    def act(self, view: PolicyView, rng: random.Random) -> int:
        legal = list(view.legal)
        if view.phase is Phase.ROB:
            return min(legal, key=lambda c: discard_value(c, list(view.hand), view.trump)).index
        led = view.current_trick[0][1] if view.current_trick else None
        ordered = order_legal_cards(legal, list(view.hand), view.trump, led)
        if led is None:
            non_trumps = [c for c in ordered if not rules.is_trump(c, view.trump)]
            return (non_trumps[0] if non_trumps else ordered[0]).index
        current_best = max(rules.trick_strength(c, view.trump, led) for _, c in view.current_trick)
        winners = [c for c in ordered if rules.trick_strength(c, view.trump, led) > current_best]
        return (winners[0] if winners else ordered[0]).index


class CountingPolicy(ConservativePolicy):
    """Public-information counter with score, position, void, and winner awareness."""
    name = "counting"
    def act(self, view: PolicyView, rng: random.Random) -> int:
        if view.phase is Phase.ROB:
            return super().act(view, rng)
        legal = list(view.legal)
        trick = view.current_trick
        led = trick[0][1] if trick else None
        ordered = order_legal_cards(legal, list(view.hand), view.trump, led)
        if led is None:
            played = {card for historical in view.trick_history for _, card in historical}
            unseen_trumps = [
                card for card in (Card.from_index(index) for index in range(52))
                if rules.is_trump(card, view.trump)
                and card not in played and card not in view.hand
                and (view.robbed or card != view.turned_card) and card != view.own_rob_discard
            ]
            trumps = [card for card in legal if rules.is_trump(card, view.trump)]
            strongest = max(trumps, key=lambda card: rules.trump_strength(card, view.trump), default=None)
            unbeatable = strongest is not None and all(
                rules.trump_strength(strongest, view.trump) > rules.trump_strength(card, view.trump)
                for card in unseen_trumps
            )
            if unbeatable or (view.scores[view.player] >= 20 and strongest is not None):
                return strongest.index
            voids: dict[int, set[Suit]] = {seat: set() for seat in range(view.num_players)}
            for historical in view.trick_history:
                category = rules.led_category(historical[0][1], view.trump)
                if category is None:
                    continue
                for seat, card in historical[1:]:
                    if not rules.is_trump(card, view.trump) and card.suit != category:
                        voids[seat].add(category)
            ordinary = [card for card in ordered if not rules.is_trump(card, view.trump)]
            if ordinary:
                opponents = [(view.player + offset) % view.num_players for offset in range(1, view.num_players)]
                safe = [card for card in ordinary if not any(card.suit in voids[seat] for seat in opponents)]
                return (safe[0] if safe else ordinary[0]).index
            return ordered[0].index
        best = max(rules.trick_strength(card, view.trump, led) for _, card in trick)
        winners = [c for c in ordered if rules.trick_strength(c, view.trump, led) > best]
        last_to_play = len(trick) == view.num_players - 1
        return (winners[0] if winners and (last_to_play or view.scores[view.player] >= 20) else ordered[0]).index


class FrozenCheckpointPolicy:
    """Loads a copied checkpoint; it never references a live learner object."""
    def __init__(self, path: str | Path, mode: str = 'greedy', epsilon: float = 0.05):
        if mode not in {'greedy', 'sampled'} or not 0 <= epsilon <= 1:
            raise ValueError('invalid evaluation mode or epsilon')
        self.mode, self.epsilon = mode, epsilon
        self.path = Path(path)
        self.name = f"checkpoint:{self.path.stem}"
        self.training_metadata: dict[str, object] = {"checkpoint_identifier": self.path.name,
            'evaluation_mode': mode, 'evaluation_epsilon': epsilon if mode == 'sampled' else 0.0}
        if self.path.suffix == ".json":
            self.data = json.loads(self.path.read_text())
            if self.data.get("ruleset_version") != rules.RULESET_VERSION:
                raise ValueError("tabular checkpoint was trained under an incompatible or unversioned ruleset")
            if self.data.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
                raise ValueError("tabular checkpoint was trained under an incompatible or unversioned observation schema")
            self.kind = "tabular"
            self.tabular_q = {
                tuple(json.loads(k)): {int(a): float(v) for a, v in row.items()}
                for k, row in self.data.get("q", {}).items()
            }
            self.training_metadata.update(training_steps=self.data.get('episodes'), ruleset_version=rules.RULESET_VERSION)
        elif self.path.suffix in {".pt", ".pth"}:
            import torch

            from .agents.neural import load_torch_checkpoint

            payload = torch.load(self.path, map_location="cpu", weights_only=True)
            checkpoint_ruleset = payload.get("metadata", {}).get("ruleset_version")
            if checkpoint_ruleset != rules.RULESET_VERSION:
                raise ValueError("neural checkpoint was trained under an incompatible or unversioned ruleset")
            checkpoint_observation = payload.get("metadata", {}).get("observation_schema_version")
            if checkpoint_observation != OBSERVATION_SCHEMA_VERSION:
                raise ValueError("neural checkpoint was trained under an incompatible or unversioned observation schema")
            checkpoint_kind, self.model = load_torch_checkpoint(self.path, payload)
            self.model.eval()
            self.data = None
            self.kind = f"torch_{checkpoint_kind}"
            state = payload.get("training_state", {})
            config = payload.get("metadata", {})
            self.training_metadata.update(
                {
                    "training_steps": state.get("environment_steps", state.get("update", state.get("episodes"))),
                    "model_parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
                    "training_seed": config.get("seed"),
                    "ruleset_version": checkpoint_ruleset,
                }
            )
            manifest_path = self.path.with_suffix(".manifest.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                self.training_metadata["training_wall_clock_seconds"] = manifest.get("duration_seconds")
        else:
            raise ValueError("checkpoint must be a schema-versioned .json, .pt, or .pth file; legacy .npz models are unsupported")
        self.requires_neural = self.kind != "tabular"
        self.requires_tabular = self.kind == "tabular"

    def act(self, view: PolicyView, rng: random.Random) -> int:
        legal = [c.index for c in view.legal]
        if self.kind.startswith("torch_"):
            import torch

            obs = torch.as_tensor(view.neural, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if self.kind == "torch_dqn":
                    scores = self.model(obs)[0]
                else:
                    mask = torch.zeros((1, 52), dtype=torch.bool)
                    mask[0, legal] = True
                    scores = self.model.policy_logits(obs, mask)[0]
            if self.mode == 'sampled':
                if self.kind != 'torch_dqn':
                    probabilities = torch.softmax(scores[legal], dim=-1).tolist()
                    return rng.choices(legal, weights=probabilities, k=1)[0]
                if rng.random() < self.epsilon:
                    return rng.choice(legal)
            return max(legal, key=lambda action: float(scores[action]))
        row = self.tabular_q.get(view.tabular, {})
        if self.mode == 'sampled':
            if rng.random() < self.epsilon:
                return rng.choice(legal)
            best = max(row.get(a, 0.0) for a in legal)
            return rng.choice([a for a in legal if row.get(a, 0.0) == best])
        return max(legal, key=lambda a: row.get(a, 0.0))


class MixedPolicy:
    name = "mixed"
    def __init__(self, policies: list[Policy], weights: list[float] | None = None):
        self.policies = policies
        self.weights = weights
        self.requires_neural = any(getattr(policy, "requires_neural", True) for policy in policies)
        self.requires_tabular = any(getattr(policy, "requires_tabular", True) for policy in policies)
    def act(self, view: PolicyView, rng: random.Random) -> int:
        return rng.choices(self.policies, weights=self.weights, k=1)[0].act(view, rng)


POLICIES = {
    "random": RandomPolicy,
    "lowest": LowestPolicy,
    "highest": HighestPolicy,
    "greedy": GreedyPolicy,
    "conservative": ConservativePolicy,
    "counting": CountingPolicy,
}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose {sorted(POLICIES)}") from exc
