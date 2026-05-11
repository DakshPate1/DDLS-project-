"""Decentralized CT-RS-q gossip consensus (Phase 3).

Same parameter split as PF-CT-RS-q: only theta_Pxx and psi_sv are shared;
all other parameters stay local.  No central server — workers exchange
parameters only with their graph neighbors.

Topologies:
  ring  — worker i is neighbors with (i-1)%N and (i+1)%N
  fc    — every worker is neighbors with every other worker (fully connected)

Mixing weights: Metropolis-Hastings weights computed from the degree sequence.
For edge (i,j):  w_{ij} = 1 / (1 + max(d_i, d_j))
Self-weight:     w_{ii} = 1 - sum_{j != i} w_{ij}
This guarantees the weight matrix W is doubly stochastic.
"""
from __future__ import annotations

import numpy as np


SHARED_PARAM_NAMES = ["theta_Pxx", "psi_sv"]


def ring_adjacency(n: int) -> list[list[int]]:
    return [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def fc_adjacency(n: int) -> list[list[int]]:
    return [[j for j in range(n) if j != i] for i in range(n)]


def mh_weights(adjacency: list[list[int]]) -> np.ndarray:
    """Doubly-stochastic Metropolis-Hastings weight matrix from adjacency list."""
    n = len(adjacency)
    degrees = [len(adj) for adj in adjacency]
    W = np.zeros((n, n))
    for i in range(n):
        for j in adjacency[i]:
            W[i, j] = 1.0 / (1 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - sum(W[i, j] for j in range(n) if j != i)
    return W


def gossip_step(workers: list, W: np.ndarray) -> None:
    """One round of gossip averaging for shared parameters (in-place).

    Each worker i computes new_param_i = sum_j W[i,j] * param_j,
    applying only to SHARED_PARAM_NAMES.  ψ and unshared θ stay local.
    """
    param_dicts = [w.get_param_dict() for w in workers]
    n = len(workers)
    new_shared = []
    for i in range(n):
        averaged = {
            name: float(sum(W[i, j] * param_dicts[j][name] for j in range(n)))
            for name in SHARED_PARAM_NAMES
        }
        new_shared.append(averaged)
    for i, w in enumerate(workers):
        w.set_param_dict(new_shared[i], only_names=SHARED_PARAM_NAMES)
