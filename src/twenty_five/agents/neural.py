"""PyTorch models and device helpers for the neural training examples."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import torch
from torch import nn

from .. import rules
from ..observations import (
    CARD_ONE_HOT,
    NEURAL_OBSERVATION_SIZE,
    OBSERVATION_LAYOUT,
    OBSERVATION_SCHEMA_VERSION,
    SLOT_SIZE,
)

if TYPE_CHECKING:
    from ..policies import PolicyView


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a CLI device choice and fail clearly for unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access it. Run the README GPU "
            "verification command and check the installed wheel and driver compatibility."
        )
    return torch.device(requested)


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks.bool(), torch.finfo(logits.dtype).min)


def _metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    return {
        **(metadata or {}),
        "ruleset_version": rules.RULESET_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
    }


def _validate_checkpoint(checkpoint: dict[str, object]) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported neural checkpoint format")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("ruleset_version") != rules.RULESET_VERSION:
        raise ValueError("checkpoint uses an incompatible or unversioned ruleset")
    if metadata.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("checkpoint uses an incompatible or unversioned observation schema")
    if int(checkpoint.get("input_dim", -1)) != NEURAL_OBSERVATION_SIZE:
        raise ValueError("checkpoint input dimension does not match the current observation schema")


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        seed: int = 0,
        dueling: bool = False,
        phase_heads: bool = False,
    ):
        super().__init__()
        seed_torch(seed)
        self.input_dim = input_dim
        self.hidden = hidden
        self.dueling = dueling
        self.phase_heads = phase_heads
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
        self.advantage = nn.Linear(hidden, 52)
        self.rob_advantage = nn.Linear(hidden, 52) if phase_heads else None
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        advantage = self.advantage(h)
        if self.rob_advantage is not None:
            rob_phase = x[..., OBSERVATION_LAYOUT.phase.start : OBSERVATION_LAYOUT.phase.start + 1]
            advantage = advantage * (1.0 - rob_phase) + self.rob_advantage(h) * rob_phase
        if self.dueling:
            return self.value(h) + advantage - advantage.mean(dim=-1, keepdim=True)
        return advantage

    def copy_from(self, other: "MLPQNetwork") -> None:
        self.load_state_dict(other.state_dict())

    def save(
        self,
        path: Path,
        *,
        training_state: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        payload: dict[str, object] = {
            "format_version": 1,
            "kind": "dqn",
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "dueling": self.dueling,
            "phase_heads": self.phase_heads,
            "state_dict": state,
        }
        if training_state is not None:
            payload["training_state"] = training_state
        payload["metadata"] = _metadata(metadata)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> "MLPQNetwork":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        _validate_checkpoint(checkpoint)
        model = cls(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden"]),
            dueling=bool(checkpoint["dueling"]),
            phase_heads=bool(checkpoint.get("phase_heads", False)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)


class LinearActorCritic(nn.Module):
    """Actor-critic kept under its old public name for checkpoint API stability."""

    def __init__(self, input_dim: int, seed: int = 0, hidden: int = 128, phase_heads: bool = False):
        super().__init__()
        seed_torch(seed)
        self.input_dim = input_dim
        self.hidden = hidden
        self.phase_heads = phase_heads
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 52)
        self.rob_actor = nn.Linear(hidden, 52) if phase_heads else None
        self.critic = nn.Linear(hidden, 1)

    def policy_and_value(self, obs: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        logits = self.actor(encoded)
        if self.rob_actor is not None:
            rob_phase = obs[..., OBSERVATION_LAYOUT.phase.start : OBSERVATION_LAYOUT.phase.start + 1]
            logits = logits * (1.0 - rob_phase) + self.rob_actor(encoded) * rob_phase
        return masked_logits(logits, masks), self.critic(encoded).squeeze(-1)

    def policy_logits(self, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.policy_and_value(obs, masks)[0]

    def distribution(self, obs: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.policy_logits(obs, masks))

    def values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(obs)).squeeze(-1)

    def save(
        self,
        path: Path,
        *,
        training_state: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        payload: dict[str, object] = {
            "format_version": 1,
            "kind": "actor_critic",
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "phase_heads": self.phase_heads,
            "state_dict": state,
        }
        if training_state is not None:
            payload["training_state"] = training_state
        payload["metadata"] = _metadata(metadata)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> "LinearActorCritic":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        _validate_checkpoint(checkpoint)
        model = cls(
            int(checkpoint["input_dim"]),
            hidden=int(checkpoint["hidden"]),
            phase_heads=bool(checkpoint.get("phase_heads", False)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)


class CardAwareActorCritic(nn.Module):
    """Actor-critic with card embeddings and attention over ordered public plays."""

    def __init__(self, input_dim: int, seed: int = 0, hidden: int = 128, phase_heads: bool = True):
        super().__init__()
        seed_torch(seed)
        self.input_dim = input_dim
        self.hidden = hidden
        self.phase_heads = phase_heads
        self.card_embedding = nn.Embedding(53, 32, padding_idx=0)
        self.seat_projection = nn.Linear(10, 32, bias=False)
        self.position_embedding = nn.Embedding(60, 64)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.sequence_encoder = nn.TransformerEncoder(sequence_layer, num_layers=1, enable_nested_tensor=False)
        self.flat_encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(hidden + 32 * 3 + 64, hidden), nn.Tanh())
        self.action_query = nn.Linear(hidden, 32)
        self.rob_action_query = nn.Linear(hidden, 32) if phase_heads else None
        self.action_bias = nn.Parameter(torch.zeros(52))
        self.rob_action_bias = nn.Parameter(torch.zeros(52)) if phase_heads else None
        self.critic = nn.Linear(hidden, 1)

    def _mean_card_embedding(self, card_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.card_embedding.weight[1:]
        count = card_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return card_mask @ embeddings / count

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        hand = self._mean_card_embedding(obs[..., :52])
        known = self._mean_card_embedding(obs[..., 52:104])
        turned = obs[..., 104:156] @ self.card_embedding.weight[1:]

        current = obs[..., OBSERVATION_LAYOUT.current_slots].reshape(*obs.shape[:-1], 10, SLOT_SIZE)
        history = obs[..., OBSERVATION_LAYOUT.history_slots].reshape(*obs.shape[:-1], 50, SLOT_SIZE)
        slots = torch.cat([history, current], dim=-2)
        card_one_hot = slots[..., :CARD_ONE_HOT]
        card_ids = card_one_hot.argmax(dim=-1)
        padding = card_one_hot.sum(dim=-1) == 0
        safe_padding = padding.clone()
        safe_padding[..., 0] = False
        slot_features = torch.cat(
            [self.card_embedding(card_ids), self.seat_projection(slots[..., CARD_ONE_HOT:])],
            dim=-1,
        )
        positions = torch.arange(60, device=obs.device)
        slot_features = slot_features + self.position_embedding(positions)
        encoded_slots = self.sequence_encoder(slot_features, src_key_padding_mask=safe_padding)
        valid = (~padding).unsqueeze(-1).to(encoded_slots.dtype)
        sequence = (encoded_slots * valid).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)
        flat = self.flat_encoder(obs)
        return self.context(torch.cat([flat, hand, known, turned, sequence], dim=-1))

    def policy_and_value(self, obs: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._encode(obs)
        action_cards = self.card_embedding.weight[1:]
        logits = self.action_query(context) @ action_cards.T / math.sqrt(action_cards.shape[-1]) + self.action_bias
        if self.rob_action_query is not None and self.rob_action_bias is not None:
            rob_logits = self.rob_action_query(context) @ action_cards.T / math.sqrt(action_cards.shape[-1]) + self.rob_action_bias
            rob_phase = obs[..., OBSERVATION_LAYOUT.phase.start : OBSERVATION_LAYOUT.phase.start + 1]
            logits = logits * (1.0 - rob_phase) + rob_logits * rob_phase
        return masked_logits(logits, masks), self.critic(context).squeeze(-1)

    def policy_logits(self, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.policy_and_value(obs, masks)[0]

    def distribution(self, obs: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.policy_logits(obs, masks))

    def values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._encode(obs)).squeeze(-1)

    def save(
        self,
        path: Path,
        *,
        training_state: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "format_version": 1,
            "kind": "card_aware_actor_critic",
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "phase_heads": self.phase_heads,
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
        }
        if training_state is not None:
            payload["training_state"] = training_state
        payload["metadata"] = _metadata(metadata)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> "CardAwareActorCritic":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        _validate_checkpoint(checkpoint)
        model = cls(
            int(checkpoint["input_dim"]),
            hidden=int(checkpoint["hidden"]),
            phase_heads=bool(checkpoint.get("phase_heads", True)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)


ActorCriticModel: TypeAlias = LinearActorCritic | CardAwareActorCritic


def load_actor_critic(path: Path, device: str | torch.device = "cpu") -> ActorCriticModel:
    kind, model = load_torch_checkpoint(path)
    if kind not in {"actor_critic", "card_aware_actor_critic"}:
        raise ValueError(f"checkpoint is not an actor-critic model: {path}")
    return model.to(device)


class FrozenActorPolicy:
    """CPU policy snapshot that cannot alias a live learner's parameters."""

    name = "frozen_actor"
    requires_neural = True
    requires_tabular = False

    def __init__(self, model: ActorCriticModel, mode: str = 'greedy'):
        if mode not in {'greedy', 'sampled'}:
            raise ValueError('invalid evaluation mode')
        self.mode = mode
        self.model = copy.deepcopy(model).to("cpu").eval()

    def act(self, view: "PolicyView", rng: random.Random) -> int:
        legal = [card.index for card in view.legal]
        obs = torch.as_tensor(view.neural, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros((1, 52), dtype=torch.bool)
        mask[0, legal] = True
        with torch.no_grad():
            logits = self.model.policy_logits(obs, mask)[0]
        if self.mode == 'sampled':
            return rng.choices(legal, weights=torch.softmax(logits[legal], dim=-1).tolist(), k=1)[0]
        return max(legal, key=lambda action: float(logits[action]))


class FrozenQPolicy:
    """CPU policy snapshot for evaluating a live Q-network."""

    name = "frozen_q"
    requires_neural = True
    requires_tabular = False

    def __init__(self, model: MLPQNetwork):
        self.model = copy.deepcopy(model).to("cpu").eval()

    def act(self, view: "PolicyView", rng: random.Random) -> int:
        legal = [card.index for card in view.legal]
        observation = torch.as_tensor(view.neural, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            values = self.model(observation)[0]
        return max(legal, key=lambda action: float(values[action]))


def load_torch_checkpoint(path: Path, payload: dict[str, object] | None = None) -> tuple[str, nn.Module]:
    checkpoint = payload or torch.load(path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint)
    kind = str(checkpoint.get("kind", ""))
    if kind == "dqn":
        model: nn.Module = MLPQNetwork(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden"]),
            dueling=bool(checkpoint["dueling"]),
            phase_heads=bool(checkpoint.get("phase_heads", False)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return kind, model
    if kind in {"actor_critic", "card_aware_actor_critic"}:
        if kind == "actor_critic":
            model = LinearActorCritic(
                int(checkpoint["input_dim"]),
                hidden=int(checkpoint["hidden"]),
                phase_heads=bool(checkpoint.get("phase_heads", False)),
            )
        else:
            model = CardAwareActorCritic(
                int(checkpoint["input_dim"]),
                hidden=int(checkpoint["hidden"]),
                phase_heads=bool(checkpoint.get("phase_heads", True)),
            )
        model.load_state_dict(checkpoint["state_dict"])
        return kind, model
    raise ValueError(f"unsupported PyTorch checkpoint kind {kind!r}: {path}")
