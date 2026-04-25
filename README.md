# Federated Risk-Sensitive Q-Learning in Continuous Time

Reproduction and federated/decentralized extension of:

> Xie (2025). *Risk-Sensitive Q-Learning in Continuous Time with Application to Dynamic Portfolio Selection.* NeurIPS 2025 Workshop. [arXiv:2512.02386](https://arxiv.org/abs/2512.02386)

DDLS course project, Universität Bern, Spring 2026.

---

## Overview

This repo reproduces the paper's main results (Table 1, Figure 1, Figure 2) and extends CT-RS-q to federated and decentralized settings across heterogeneous financial markets.

**Three phases:**
- **Phase 1 (done):** Reproduce the single-agent continuous-time risk-sensitive q-learning results.
- **Phase 2 (in progress):** Federated extension — Local / FedAvg / PF-CT-RS-q across 4 heterogeneous worker markets with different dynamics and risk preferences.
- **Phase 3 (planned):** Decentralized extension — gossip-based consensus on ring and fully-connected topologies, no central server.

---

## Phase 1 Results

| Policy | Cum Return | Std Dev | MV | vs Paper MV |
|---|---|---|---|---|
| Baseline (a=0.5) | 0.221 | 0.095 | 1.217 | +0.0% |
| Optimal (B.1 closed form) | 0.709 | 0.720 | 1.450 | −0.2% |
| CT-RS-q (learned) | 0.729 | 0.791 | 1.416 | −1.5% |

MV objective reproduced within 1.5% of the paper. Baseline and Optimal within Monte Carlo noise.

**Key finding:** The B.2 parameterization has a structural non-identifiability — `ψ_ce1` and `ψ_ce2` enter the q-function loss only through their difference. Individual parameter convergence is impossible; the identified combination `(ψ_ce1 − ψ_ce2)` converges consistently. This motivates the federated design: vanilla FedAvg on raw parameters fails because each worker drifts to a different point in the null-space.

---

## Structure

```
src/
  sde.py              # Euler-Maruyama portfolio SDE (Eq. 35)
  models.py           # J_θ and q_ψ parameterizations (Appendix B.2)
  policies.py         # baseline / optimal (B.1) / trained Gaussian policy
  metrics.py          # MV objective, find_bhat_star fixed-point
  ct_rs_q.py          # Algorithm 2 trainer (Adam, batched episodes)

experiments/
  reproduce.py        # full Table 1 + Figure 1 + Figure 2 pipeline
  check_convergence.py # single-shot sanity check with pass/fail summary

plots/
  figure1_convergence.png
  figure2_time_evolution.png

results/
  table1.txt
  metrics.json
  history.npz
```

---

## Quickstart

```bash
# Full reproduction (~3 min)
python -m experiments.reproduce --eps 1500 --seed 7

# Quick sanity check (~1 min)
python -m experiments.check_convergence

# Top-to-bottom smoke test of all modules
python walkthrough.py
```

Outputs land in `plots/` and `results/`.

---

## Implementation Notes

The paper's Algorithm 2 pseudocode underspecifies three things we resolved:

- **Evaluation uses b̂\* fixed-point.** Table 1 policies are evaluated at `b₀ = −b̂*` where `b̂* = E[X_T]^π` (self-consistent fixed point of the outer OCE). Using `b₀ = 0` as the spec suggests causes the Optimal policy to underperform by 5×.
- **Adam optimizer, lr = 3×10⁻³, batch = 16.** Gradient scales span 4 orders of magnitude across 8 parameters; plain SGD stalls. Single-trajectory TD noise dominates the drift signal; batching 16 trajectories per episode reduces noise by 4×.
- **Warm-start initialization** within ±15% of B.3 ground truth. The paper does not specify initialization; visibly-off starting points push Adam out of the convergence basin.

---

## Requirements

```
numpy
torch
matplotlib
```

No other dependencies.
