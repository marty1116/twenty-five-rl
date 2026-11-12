"""Resolved configuration and explicit continuation contracts for trainers."""
from __future__ import annotations

import hashlib
import inspect
import platform
import random
from pathlib import Path

import numpy as np
import torch

from .protocol import source_identity

TRAINER_VERSION = 2


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def resolved_arguments(function, namespace):
    return {name: plain(namespace[name]) for name in inspect.signature(function).parameters}


def continuation_contract(arguments, device):
    # Invocation budget, destination and logging do not alter a completed-boundary trajectory.
    excluded = {'output', 'resume', 'resume_mode', 'updates', 'episodes', 'learning_rate',
                'override_learning_rate', 'log_interval', 'progress', 'tensorboard_dir',
                'checkpoint_interval', 'keep_snapshots', 'opponent_checkpoints'}
    settings = {k: v for k, v in arguments.items() if k not in excluded}
    settings['device'] = str(device)
    settings['opponent_checkpoint_hashes'] = [hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                              for p in arguments.get('opponent_checkpoints') or []]
    return {'trainer_version': TRAINER_VERSION, **source_identity(), 'settings': settings,
            'runtime': {'torch': str(torch.__version__), 'numpy': str(np.__version__),
                        'python': platform.python_version(), 'platform': platform.platform(),
                        'torch_threads': torch.get_num_threads()}}


def validate_continuation(payload, contract):
    old = payload.get('metadata', {}).get('continuation_contract')
    if old != contract:
        raise ValueError('exact continuation requires the same source, runtime, settings and opponent content; '
                         'use resume_mode="warm_start" for an explicit new run')
    state = payload.get('training_state', {})
    required = {'torch_rng_state', 'cuda_rng_states', 'python_rng_state', 'numpy_rng_state', 'optimizer_state_dict'}
    if not required <= state.keys():
        raise ValueError('checkpoint lacks complete continuation state; use resume_mode="warm_start"')


def capture_rng():
    state = np.random.get_state()
    return {'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python_rng_state': random.getstate(),
            'numpy_rng_state': (state[0], state[1].tolist(), state[2], state[3], state[4])}


def restore_rng(state, device):
    torch.set_rng_state(state['torch_rng_state'].cpu())
    if str(device).startswith('cuda'):
        streams = state.get('cuda_rng_states', [])
        if len(streams) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA generator count does not match the runtime')
        torch.cuda.set_rng_state_all([x.cpu() for x in streams])
    if 'python_rng_state' in state:
        random.setstate(state['python_rng_state'])
    if 'numpy_rng_state' in state:
        name, keys, pos, has_gauss, cached = state['numpy_rng_state']
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))


def effective_config(arguments, optimizer=None, **extra):
    kind = ('fresh' if not arguments.get('resume') else
            'warm_start' if arguments.get('resume_mode') == 'warm_start' else
            'approximate' if arguments.get('resume_mode') == 'approximate' else
            'continued_with_explicit_lr_override' if arguments.get('override_learning_rate') else
            'same_contract_boundary_continuation')
    return {**arguments, **extra, 'requested_arguments': dict(arguments), 'trainer_version': TRAINER_VERSION,
            'continuation_kind': kind,
            **({'requested_learning_rate': arguments['learning_rate'],
                'effective_learning_rates': [g['lr'] for g in optimizer.param_groups]} if optimizer is not None else {})}
