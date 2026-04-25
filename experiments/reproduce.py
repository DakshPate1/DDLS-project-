"""Phase 1 reproduction: Table 1, Figure 1, Figure 2.

USAGE (from project root):
    python -m experiments.reproduce              # defaults (1500 eps, seed 7)
    python -m experiments.reproduce --eps 3000   # longer training
    python -m experiments.reproduce --seed 42

Outputs:
    plots/figure1_convergence.png    — 8-panel parameter convergence vs ground truth
    plots/figure2_time_evolution.png — E[X_t]-X_0 and MV(X_t) time paths, 3 policies
    results/table1.txt               — Cum Return / Std Dev / MV for each policy
    results/history.npz              — raw training-history arrays (for re-plotting)
    results/metrics.json             — machine-readable summary (for the report)

Evaluation follows the b̂* fixed-point correction to PROJECT_CONTEXT.md (see
project_bhat_star_evaluation.md). Both Optimal and CT-RS-q policies are rolled
out under b_0 = −b̂* where b̂* is the self-consistent E[X_T]^π (MV objective).

Verification gate is FUNCTIONAL — the paper's Appendix B.2 parameterization has
non-identifiable directions (ψ_ce1, ψ_ce2 drift together; θ_Pxx vs θ_Pnl on
their own axis). We check downstream MV/Cum/Std against paper Table 1 and the
identified combinations, not raw per-parameter errors.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ct_rs_q import CTRSQTrainer, TrainingConfig, warm_start_params
from src.metrics import (
    cumulative_return_path,
    cumulative_return_std_path,
    find_bhat_star,
    mean_variance_objective,
    mv_path,
    terminal_cumulative_return,
    terminal_std,
)
from src.models import (
    GROUND_TRUTH_PSI,
    GROUND_TRUTH_THETA,
    QFunction,
    ValueFunction,
)
from src.policies import baseline_policy, optimal_policy, trained_policy
from src.sde import MarketParams, PortfolioSDE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"


# -----------------------------------------------------------------------------
# 1. Training (Algorithm 2) — produces Figure 1 traces + trained Q used in Table
# -----------------------------------------------------------------------------
def run_training(n_episodes: int, seed: int):
    params = MarketParams()
    sde = PortfolioSDE(params)
    theta_init, psi_init = warm_start_params()
    J = ValueFunction(alpha=1.0, T=params.T, **theta_init)
    Q = QFunction(alpha=1.0, T=params.T, **psi_init)
    cfg = TrainingConfig(n_episodes=n_episodes, log_every=max(1, n_episodes // 10))
    trainer = CTRSQTrainer(J, Q, sde, X0=1.0, config=cfg)

    t0 = time.time()
    hist = trainer.train(np.random.default_rng(seed))
    elapsed = time.time() - t0
    print(f"[train] {n_episodes} episodes in {elapsed:.1f}s "
          f"({elapsed*1000/n_episodes:.1f} ms/ep)")
    return params, sde, J, Q, hist, cfg


# -----------------------------------------------------------------------------
# 2. Table 1 — Baseline vs Optimal vs CT-RS-q under b̂* fixed point
# -----------------------------------------------------------------------------
def build_table1(params, sde, Q, cfg, alpha=1.0, X0=1.0, n_eval=10_000):
    rows = []  # (name, cum, std, mv, b_hat, time_path_X)

    rng = np.random.default_rng(101)
    times, X_base, _ = sde.simulate(
        X0, baseline_policy(0.5), n_trajectories=n_eval, rng=rng,
    )
    rows.append((
        "Baseline (a=0.5)",
        terminal_cumulative_return(X_base[-1], X0),
        terminal_std(X_base[-1]),
        mean_variance_objective(X_base[-1], alpha),
        0.0,  # baseline policy doesn't use b0
        X_base,
    ))

    rng = np.random.default_rng(102)
    b_hat_opt, _, X_opt = find_bhat_star(
        policy_factory=lambda b0: optimal_policy(params, alpha=alpha, b0=b0, b1=1.0),
        sde=sde, X0=X0, n_trajectories=n_eval, rng=rng,
    )
    rows.append((
        "Optimal (B.1 closed form)",
        terminal_cumulative_return(X_opt[-1], X0),
        terminal_std(X_opt[-1]),
        mean_variance_objective(X_opt[-1], alpha),
        b_hat_opt,
        X_opt,
    ))

    rng = np.random.default_rng(103)
    b_hat_trn, _, X_trn = find_bhat_star(
        policy_factory=lambda b0: trained_policy(Q, tau=cfg.tau, rng=rng,
                                                 b0=b0, b1=1.0),
        sde=sde, X0=X0, n_trajectories=n_eval, rng=rng,
    )
    rows.append((
        "CT-RS-q (trained)",
        terminal_cumulative_return(X_trn[-1], X0),
        terminal_std(X_trn[-1]),
        mean_variance_objective(X_trn[-1], alpha),
        b_hat_trn,
        X_trn,
    ))
    return times, rows


def write_table1(rows, n_pass_combo, func_pass, path: Path) -> str:
    paper = {
        "Baseline (a=0.5)":          (0.2217, 0.0957, 1.2171),
        "Optimal (B.1 closed form)": (0.7128, 0.7205, 1.4532),
        "CT-RS-q (trained)":         (0.8163, 0.8716, 1.4365),
    }
    lines = ["Table 1 — reproduced (our run vs paper)\n"]
    lines.append(f"{'Policy':30s} | {'Cum Return':>10s} | {'Std Dev':>7s} | {'MV':>6s} | b̂*\n")
    lines.append("-" * 78 + "\n")
    for name, cum, std, mv, b_hat, _ in rows:
        p_cum, p_std, p_mv = paper[name]
        lines.append(
            f"{name:30s} | {cum:8.4f}  | {std:5.4f}  | {mv:5.4f} | {b_hat:6.4f}\n"
            f"{'  (paper)':30s} | {p_cum:8.4f}  | {p_std:5.4f}  | {p_mv:5.4f} | --\n"
        )
    lines.append("-" * 78 + "\n")
    lines.append(f"Identifiable-combination gate: {n_pass_combo}/4\n")
    lines.append(f"Functional verification gate:  {'PASS' if func_pass else 'FAIL'}\n")
    lines.append(
        "\nNote: Optimal / CT-RS-q rows use b̂* fixed-point evaluation per\n"
        "Algorithm 1 (outer OCE). Baseline does not depend on b_0.\n"
    )
    path.write_text("".join(lines))
    return "".join(lines)


# -----------------------------------------------------------------------------
# 3. Figure 1 — 8-panel parameter convergence
# -----------------------------------------------------------------------------
PARAM_LABELS = [
    ("theta_Px",  r"$\theta_{P_x}$"),
    ("theta_Pxx", r"$\theta_{P_{xx}}$"),
    ("theta_Pnl", r"$\theta_{P_{nl}}$"),
    ("psi_a0",    r"$\psi_{a_0}$"),
    ("psi_a1",    r"$\psi_{a_1}$"),
    ("psi_sv",    r"$\psi_{sv}$"),
    ("psi_ce1",   r"$\psi_{c_1}^e$"),
    ("psi_ce2",   r"$\psi_{c_2}^e$"),
]


def plot_figure1(hist, path: Path):
    hist_dict = hist.as_dict()
    all_gt = {**GROUND_TRUTH_THETA, **GROUND_TRUTH_PSI}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, (key, label) in zip(axes.flat, PARAM_LABELS):
        trace = np.asarray(hist_dict[key])
        ax.plot(trace, color="C0", lw=1.0)
        ax.axhline(all_gt[key], color="C3", ls="--", lw=1.0, label="ground truth")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("episode")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Figure 1 — Parameter convergence (Algorithm 2 under warm-start)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Figure 2 — time evolution of E[X_t]-X0 and MV(X_t) under the 3 policies
# -----------------------------------------------------------------------------
def plot_figure2(times, rows, alpha: float, X0: float, path: Path):
    colors = {"Baseline (a=0.5)": "C2",
              "Optimal (B.1 closed form)": "C1",
              "CT-RS-q (trained)": "C0"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, _, _, _, _, X in rows:
        mean_path = cumulative_return_path(X, X0)
        std_path = cumulative_return_std_path(X)
        c = colors[name]
        ax1.plot(times, mean_path, color=c, lw=1.5, label=name)
        ax1.fill_between(times, mean_path - std_path, mean_path + std_path,
                          color=c, alpha=0.15)
        ax2.plot(times, mv_path(X, alpha), color=c, lw=1.5, label=name)
    ax1.set_title(r"$E[X_t] - X_0 \pm \mathrm{Std}(X_t)$")
    ax1.set_xlabel("t"); ax1.set_ylabel("cumulative return")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    ax2.set_title(r"$\mathrm{MV}(X_t) = E[X_t] - \frac{\alpha}{2}\mathrm{Var}(X_t)$")
    ax2.set_xlabel("t"); ax2.set_ylabel("mean-variance")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
    fig.suptitle("Figure 2 — Time evolution of cumulative return and MV", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Identifiability / functional gate
# -----------------------------------------------------------------------------
def evaluate_identified_combinations(J, Q):
    final = dict(
        theta_Px=J.theta_Px.item(), theta_Pxx=J.theta_Pxx.item(),
        theta_Pnl=J.theta_Pnl.item(),
        psi_a0=Q.psi_a0.item(), psi_a1=Q.psi_a1.item(),
        psi_sv=Q.psi_sv.item(), psi_ce1=Q.psi_ce1.item(), psi_ce2=Q.psi_ce2.item(),
    )
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
    n_pass = 0
    out = []
    for name, cur, gt in combos:
        err = abs(cur - gt) / max(abs(gt), 1e-6) * 100
        if err < 10.0:
            n_pass += 1
        out.append({"name": name, "learned": cur, "ground_truth": gt, "err_pct": err})
    return n_pass, out, final


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=int, default=1500,
                        help="Training episodes for CT-RS-q (default 1500).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-eval", type=int, default=10_000,
                        help="Number of trajectories per policy for Table 1.")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    params, sde, J, Q, hist, cfg = run_training(args.eps, args.seed)

    # Table 1 rollouts (shared timegrid across all 3 policies)
    times, rows = build_table1(params, sde, Q, cfg,
                               alpha=1.0, X0=1.0, n_eval=args.n_eval)

    # Identifiability and functional gate
    n_pass_combo, combo_list, final_params = evaluate_identified_combinations(J, Q)
    cum_trn = [r for r in rows if r[0] == "CT-RS-q (trained)"][0]
    _, cum, std, mv, _, _ = cum_trn
    paper_mv, paper_cum, paper_std = 1.4365, 0.8163, 0.8716
    err_mv  = abs(mv  - paper_mv)  / paper_mv  * 100
    err_cum = abs(cum - paper_cum) / paper_cum * 100
    err_std = abs(std - paper_std) / paper_std * 100
    func_pass = (err_mv < 5.0) and (err_cum < 10.0) and (err_std < 10.0)

    # Write Table 1
    table_text = write_table1(rows, n_pass_combo, func_pass,
                              RESULTS_DIR / "table1.txt")
    print("\n" + table_text)

    # Plots
    plot_figure1(hist, PLOT_DIR / "figure1_convergence.png")
    plot_figure2(times, rows, alpha=1.0, X0=1.0,
                 path=PLOT_DIR / "figure2_time_evolution.png")
    print(f"[plots]   {PLOT_DIR/'figure1_convergence.png'}")
    print(f"[plots]   {PLOT_DIR/'figure2_time_evolution.png'}")

    # Raw history + machine-readable summary
    np.savez(RESULTS_DIR / "history.npz",
             **{k: np.asarray(v) for k, v in hist.as_dict().items()})
    summary = {
        "n_episodes": args.eps, "seed": args.seed, "n_eval": args.n_eval,
        "config": {
            "lr_theta": cfg.lr_theta, "lr_psi": cfg.lr_psi, "tau": cfg.tau,
            "n_trajectories_per_episode": cfg.n_trajectories_per_episode,
            "optimizer": cfg.optimizer,
        },
        "learned_params": final_params,
        "identified_combinations": combo_list,
        "table1": [
            {"policy": r[0], "cum": r[1], "std": r[2], "mv": r[3], "b_hat": r[4]}
            for r in rows
        ],
        "functional_gate": {
            "mv_err_pct": err_mv, "cum_err_pct": err_cum, "std_err_pct": err_std,
            "pass": func_pass,
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[results] {RESULTS_DIR/'history.npz'}")
    print(f"[results] {RESULTS_DIR/'metrics.json'}")
    print(f"[results] {RESULTS_DIR/'table1.txt'}")

    print(
        f"\nFunctional gate: "
        f"MV err={err_mv:.2f}% cum err={err_cum:.2f}% std err={err_std:.2f}% "
        f"=> {'PASS' if func_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
