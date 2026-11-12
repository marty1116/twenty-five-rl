"""Production numerical paths shared by trainers and independent oracle tests."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def compute_gae(rows, gamma: float, gae_lambda: float):
    advantages = np.zeros(len(rows), dtype=np.float32)
    targets = np.zeros(len(rows), dtype=np.float32)
    last = {}
    for i in range(len(rows)-1, -1, -1):
        row = rows[i]
        key = (row.env_id, row.episode_id, row.seat)
        if row.done and row.next_value != 0:
            raise ValueError('terminal rollout rows must have zero next value')
        delta = row.reward + gamma * row.next_value - row.value
        advantages[i] = delta + gamma * gae_lambda * (0.0 if row.done else last.get(key, 0.0))
        targets[i] = advantages[i] + row.value
        last[key] = advantages[i]
    return advantages, targets


def ppo_objective(logits, values, actions, old_log_probabilities, advantages, returns,
                  clip_range=.2, value_coefficient=.5, entropy_coefficient=0., policy_weights=None):
    if any(t.requires_grad for t in (old_log_probabilities, advantages, returns)):
        raise ValueError('rollout log probabilities, advantages and returns must be detached')
    distribution = torch.distributions.Categorical(logits=logits)
    log_ratio = distribution.log_prob(actions) - old_log_probabilities
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1-clip_range, 1+clip_range) * advantages)
    if policy_weights is not None:
        surrogate = surrogate * policy_weights
    policy_loss = -surrogate.mean()
    value_loss = F.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    return loss, policy_loss, value_loss, entropy, log_ratio, ratio
