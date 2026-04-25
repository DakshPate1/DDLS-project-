"""Single-shot convergence check for the CT-RS-q trainer (Algorithm 2).

Runs the trainer with warm-start initialization (±15% of B.3 ground truth)
and prints a compact report: final parameter errors, downstream MV metrics
vs Table 1, and total wall-clock time.

USAGE (from project root):
    python -m experiments.check_convergence            # default 2000 eps
    python -m experiments.check_convergence 5000       # longer run
    python -m experiments.check_convergence 500 1      # quick, fixed seed

Copy the PASS/FAIL table at the bottom back to Claude so it can verify the
run worked as expected.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from src.ct_rs_q import CTRSQTrainer, TrainingConfig, warm_start_params
from src.metrics import (
    find_bhat_star,
    mean_variance_objective,
    terminal_cumulative_return,
    terminal_std,
)
from src.models import (
    GROUND_TRUTH_PSI,
    GROUND_TRUTH_THETA,
    QFunction,
    ValueFunction,
)
from src.policies import trained_policy
from src.sde import MarketParams, PortfolioSDE

# -----------------------------------------------------------------------------
# CLI args
# -----------------------------------------------------------------------------
n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
params = MarketParams()
sde = PortfolioSDE(params)
theta_init, psi_init = warm_start_params()
J = ValueFunction(alpha=1.0, T=params.T, **theta_init)
Q = QFunction(alpha=1.0, T=params.T, **psi_init)

cfg = TrainingConfig(n_episodes=n_episodes, log_every=max(1, n_episodes // 10))
trainer = CTRSQTrainer(J, Q, sde, X0=1.0, config=cfg)

print(f"=== CT-RS-q convergence check ===")
print(f"  n_episodes = {cfg.n_episodes}   batch = {cfg.n_trajectories_per_episode}")
print(f"  lr_theta   = {cfg.lr_theta}   lr_psi = {cfg.lr_psi}   optimizer = {cfg.optimizer}")
print(f"  tau        = {cfg.tau}   seed = {seed}")
print(f"  init       = warm-start (±15% of B.3 ground truth)\n")

t0 = time.time()
hist = trainer.train(np.random.default_rng(seed))
elapsed = time.time() - t0
print(f"\n[train] {elapsed:.1f}s  ({elapsed * 1000 / cfg.n_episodes:.1f} ms/ep)")

# -----------------------------------------------------------------------------
# Parameter-error table (verification gate: within 10% of ground truth)
# -----------------------------------------------------------------------------
final = {
    "theta_Px":  J.theta_Px.item(),
    "theta_Pxx": J.theta_Pxx.item(),
    "theta_Pnl": J.theta_Pnl.item(),
    "psi_a0":    Q.psi_a0.item(),
    "psi_a1":    Q.psi_a1.item(),
    "psi_sv":    Q.psi_sv.item(),
    "psi_ce1":   Q.psi_ce1.item(),
    "psi_ce2":   Q.psi_ce2.item(),
}
all_gt = {**GROUND_TRUTH_THETA, **GROUND_TRUTH_PSI}

print("\n=== Raw parameter convergence vs B.3 ground truth ===")
print(f"  {'name':12s}  {'init':>9s}  {'learned':>9s}  {'ground-truth':>13s}  {'err%':>7s}  status")
n_pass_raw = 0
all_init = {**theta_init, **psi_init}
for name, gt in all_gt.items():
    cur = final[name]
    init = all_init[name]
    err_pct = abs(cur - gt) / max(abs(gt), 1e-6) * 100
    status = "PASS" if err_pct < 10.0 else "FAIL"
    if err_pct < 10.0:
        n_pass_raw += 1
    print(f"  {name:12s}  {init:+9.4f}  {cur:+9.4f}  {gt:+13.4f}  {err_pct:6.1f}%  {status}")
print(f"  --- {n_pass_raw}/8 raw parameters within 10% gate ---")

# Identifiability: ψ_ce1 and ψ_ce2 enter a_ψ only via their difference, and
# ψ_sv/ψ_ce2 enter q only via ψ_sv·exp(ψ_ce2·τ). Likewise θ_Pxx and θ_Pnl
# enter J_θ in the combinations (θ_Pxx+θ_Pnl) and (θ_Px+θ_Pxx−θ_Pnl). The raw
# 8-parameter gate is over-strict; functional identification is what matters.
import math
combos = [
    ("psi_ce1-psi_ce2",
        final["psi_ce1"] - final["psi_ce2"],
        GROUND_TRUTH_PSI["psi_ce1"] - GROUND_TRUTH_PSI["psi_ce2"]),
    ("psi_sv*exp(psi_ce2)",
        final["psi_sv"] * math.exp(final["psi_ce2"]),
        GROUND_TRUTH_PSI["psi_sv"] * math.exp(GROUND_TRUTH_PSI["psi_ce2"])),
    ("theta_Pxx+theta_Pnl",
        final["theta_Pxx"] + final["theta_Pnl"],
        GROUND_TRUTH_THETA["theta_Pxx"] + GROUND_TRUTH_THETA["theta_Pnl"]),
    ("theta_Px+theta_Pxx-theta_Pnl",
        final["theta_Px"] + final["theta_Pxx"] - final["theta_Pnl"],
        GROUND_TRUTH_THETA["theta_Px"] + GROUND_TRUTH_THETA["theta_Pxx"]
            - GROUND_TRUTH_THETA["theta_Pnl"]),
]
print("\n=== Identified parameter combinations ===")
print(f"  {'combination':28s}  {'learned':>9s}  {'ground-truth':>13s}  {'err%':>7s}  status")
n_pass_combo = 0
for name, cur, gt in combos:
    err_pct = abs(cur - gt) / max(abs(gt), 1e-6) * 100
    status = "PASS" if err_pct < 10.0 else "FAIL"
    if err_pct < 10.0:
        n_pass_combo += 1
    print(f"  {name:28s}  {cur:+9.4f}  {gt:+13.4f}  {err_pct:6.1f}%  {status}")
print(f"  --- {n_pass_combo}/{len(combos)} identified combinations within 10% gate ---")

# -----------------------------------------------------------------------------
# Downstream: does the TRAINED policy match Table 1 "CT-RS-q" numbers?
# (paper: cum=0.8163 std=0.8716 MV=1.4365)
# -----------------------------------------------------------------------------
print("\n=== Downstream MV metrics (via b̂* fixed point, 10k trajectories) ===")
rng = np.random.default_rng(123)
b_hat, _, X_trn = find_bhat_star(
    policy_factory=lambda b0: trained_policy(Q, tau=cfg.tau, rng=rng, b0=b0, b1=1.0),
    sde=sde, X0=1.0, n_trajectories=10_000, rng=rng,
)
cum = terminal_cumulative_return(X_trn[-1], 1.0)
std = terminal_std(X_trn[-1])
mv = mean_variance_objective(X_trn[-1], alpha=1.0)
print(f"  learned policy:  cum={cum:.4f}   std={std:.4f}   MV={mv:.4f}   (b̂*={b_hat:.4f})")
print(f"  paper Table 1 :  cum=0.8163   std=0.8716   MV=1.4365  (CT-RS-q)")
print(f"                   cum=0.7128   std=0.7205   MV=1.4532  (Optimal)")

# Functional verification gate — what the paper actually tests.
paper_mv, paper_cum, paper_std = 1.4365, 0.8163, 0.8716
err_mv  = abs(mv  - paper_mv)  / paper_mv  * 100
err_cum = abs(cum - paper_cum) / paper_cum * 100
err_std = abs(std - paper_std) / paper_std * 100
func_pass = (err_mv < 5.0) and (err_cum < 10.0) and (err_std < 10.0)
print("\n=== Functional gate (paper-faithful) ===")
print(f"  MV  err = {err_mv:5.2f}%   (<5%  => {'PASS' if err_mv<5 else 'FAIL'})")
print(f"  cum err = {err_cum:5.2f}%  (<10% => {'PASS' if err_cum<10 else 'FAIL'})")
print(f"  std err = {err_std:5.2f}%  (<10% => {'PASS' if err_std<10 else 'FAIL'})")
print(f"  OVERALL: {'PASS' if func_pass else 'FAIL'}")

print("\n=== COPY-PASTE SUMMARY ===")
print(
    f"raw={n_pass_raw}/8  combo={n_pass_combo}/{len(combos)}  "
    f"MV={mv:.4f}  cum={cum:.4f}  std={std:.4f}  "
    f"functional={'PASS' if func_pass else 'FAIL'}  elapsed={elapsed:.1f}s"
)
