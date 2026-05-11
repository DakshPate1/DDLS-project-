from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.ct_rs_q import CTRSQTrainer, TrainingConfig
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


THETA_NAMES = ["theta_Px", "theta_Pxx", "theta_Pnl"]
PSI_NAMES = ["psi_a0", "psi_a1", "psi_sv", "psi_ce1", "psi_ce2"]
ALL_PARAM_NAMES = THETA_NAMES + PSI_NAMES
SHARED_PARAM_NAMES = ["theta_Pxx", "psi_sv"]


@dataclass
class WorkerSpec:
    name: str
    market_params: MarketParams
    alpha: float
    seed: int


def warm_start_params(rng: np.random.Generator, jitter_pct: float = 0.15):
    """
    initialize within about ±15% of the B.3 values.
    """
    theta_init = {}
    psi_init = {}

    for k, v in GROUND_TRUTH_THETA.items():
        scale = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
        theta_init[k] = float(v * scale)

    for k, v in GROUND_TRUTH_PSI.items():
        scale = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
        psi_init[k] = float(v * scale)

    return theta_init, psi_init


def default_worker_specs(seed: int = 7) -> tuple[list[WorkerSpec], WorkerSpec]:
    """
    4 heterogeneous worker markets + 1 held-out regime.
    """
    workers = [
        WorkerSpec(
            name="worker_1_bull_lowvol",
            market_params=MarketParams(r1=0.16, r2=0.24, sigma1=0.09, sigma2=0.11, T=1.0, dt=0.001),
            alpha=0.90,
            seed=seed + 11,
        ),
        WorkerSpec(
            name="worker_2_bull_highvol",
            market_params=MarketParams(r1=0.17, r2=0.26, sigma1=0.13, sigma2=0.15, T=1.0, dt=0.001),
            alpha=1.10,
            seed=seed + 22,
        ),
        WorkerSpec(
            name="worker_3_bear_mixed",
            market_params=MarketParams(r1=0.12, r2=0.22, sigma1=0.10, sigma2=0.14, T=1.0, dt=0.001),
            alpha=1.20,
            seed=seed + 33,
        ),
        WorkerSpec(
            name="worker_4_asym_vol",
            market_params=MarketParams(r1=0.15, r2=0.23, sigma1=0.08, sigma2=0.16, T=1.0, dt=0.001),
            alpha=1.00,
            seed=seed + 44,
        ),
    ]

    heldout = WorkerSpec(
        name="heldout_regime",
        market_params=MarketParams(r1=0.14, r2=0.27, sigma1=0.11, sigma2=0.17, T=1.0, dt=0.001),
        alpha=1.05,
        seed=seed + 999,
    )
    return workers, heldout


class FederatedWorker:
    def __init__(
        self,
        spec: WorkerSpec,
        n_total_episodes: int,
        log_every: int = 100,
        x0: float = 1.0,
    ):
        self.spec = spec
        self.x0 = float(x0)
        self.rng = np.random.default_rng(spec.seed)

        self.sde = PortfolioSDE(spec.market_params)

        theta_init, psi_init = warm_start_params(self.rng)
        self.J = ValueFunction(alpha=spec.alpha, T=spec.market_params.T, **theta_init)
        self.Q = QFunction(alpha=spec.alpha, T=spec.market_params.T, **psi_init)

        self.cfg = TrainingConfig(
            tau=0.1,
            lr_theta=3e-3,
            lr_psi=3e-3,
            n_episodes=n_total_episodes,
            n_trajectories_per_episode=16,
            log_every=log_every,
            optimizer="adam",
            grad_clip=None,
        )
        self.trainer = CTRSQTrainer(self.J, self.Q, self.sde, X0=self.x0, config=self.cfg)

        # Pin the null-space flat direction: ψ_ce1 and ψ_ce2 enter q_ψ only
        # through their difference, so both cannot be identified individually.
        # Freeze ψ_ce2 at its warm-start value; only ψ_ce1 updates freely.
        # Without this, each worker drifts to a different (ψ_ce1, ψ_ce2) pair
        # making their identified combination incomparable across workers.
        self.Q.psi_ce2.requires_grad_(False)
        self.trainer.psi_params = [p for p in self.trainer.psi_params if p is not self.Q.psi_ce2]
        if self.trainer._opt_psi is not None:
            self.trainer._opt_psi = torch.optim.Adam(self.trainer.psi_params, lr=self.cfg.lr_psi)

    def local_train(self, n_episodes: int):
        """
        Run n_episodes of the already-implemented Phase-1 trainer.
        We call the trainer's per-episode update directly, so we can federate between rounds.
        """
        td_vals = []
        for _ in range(n_episodes):
            info = self.trainer._one_episode(self.rng)
            td_vals.append(info["td_abs_mean"])
        return {
            "avg_td_abs_mean": float(np.mean(td_vals)) if td_vals else 0.0,
            "last_td_abs_mean": float(td_vals[-1]) if td_vals else 0.0,
        }

    def get_param_dict(self) -> dict[str, float]:
        out = {}
        for name in THETA_NAMES:
            out[name] = float(getattr(self.J, name).item())
        for name in PSI_NAMES:
            out[name] = float(getattr(self.Q, name).item())
        return out

    def set_param_dict(self, new_params: dict[str, float], only_names: list[str] | None = None):
        names = only_names if only_names is not None else list(new_params.keys())
        with torch.no_grad():
            for name in names:
                value = float(new_params[name])
                if name in THETA_NAMES:
                    getattr(self.J, name).copy_(torch.tensor(value, dtype=torch.float64))
                elif name in PSI_NAMES:
                    getattr(self.Q, name).copy_(torch.tensor(value, dtype=torch.float64))
                else:
                    raise KeyError(f"Unknown parameter name: {name}")

    def evaluate_on(self, eval_sde: PortfolioSDE, n_eval: int = 5000) -> dict:
        eval_rng = np.random.default_rng(self.spec.seed + 100_000)

        b_hat, times, X = find_bhat_star(
            policy_factory=lambda b0: trained_policy(
                self.Q, tau=self.cfg.tau, rng=eval_rng, b0=b0, b1=1.0
            ),
            sde=eval_sde,
            X0=self.x0,
            n_trajectories=n_eval,
            rng=eval_rng,
        )

        return {
            "b_hat": float(b_hat),
            "cum_return": float(terminal_cumulative_return(X[-1], self.x0)),
            "std": float(terminal_std(X[-1])),
            "mv": float(mean_variance_objective(X[-1], self.spec.alpha)),
            "n_eval": int(n_eval),
            "time_steps": int(len(times)),
        }


def market_vector(spec: WorkerSpec) -> np.ndarray:
    p = spec.market_params
    return np.array([p.r1, p.r2, p.sigma1, p.sigma2, spec.alpha], dtype=float)


def regime_aware_weights(workers: list[FederatedWorker]) -> np.ndarray:
    """
    Lightweight regime-aware weighting:
    workers closer to the cohort mean regime get slightly more weight.
    """
    M = np.stack([market_vector(w.spec) for w in workers], axis=0)
    center = M.mean(axis=0)
    dists = np.linalg.norm(M - center[None, :], axis=1)
    weights = 1.0 / np.maximum(dists, 1e-8)
    weights = weights / weights.sum()
    return weights


def average_params(param_dicts: list[dict[str, float]], names: list[str], weights: np.ndarray | None = None):
    n = len(param_dicts)
    if weights is None:
        weights = np.ones(n, dtype=float) / n
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    avg = {}
    for name in names:
        avg[name] = float(sum(weights[i] * param_dicts[i][name] for i in range(n)))
    return avg


def evaluate_workers(
    method_name: str,
    workers: list[FederatedWorker],
    heldout_spec: WorkerSpec,
    n_eval: int,
) -> list[dict]:
    heldout_sde = PortfolioSDE(heldout_spec.market_params)
    rows = []

    for w in workers:
        own_metrics = w.evaluate_on(w.sde, n_eval=n_eval)
        heldout_metrics = w.evaluate_on(heldout_sde, n_eval=n_eval)

        rows.append(
            {
                "method": method_name,
                "worker": w.spec.name,
                "alpha": w.spec.alpha,
                "market": {
                    "r1": w.spec.market_params.r1,
                    "r2": w.spec.market_params.r2,
                    "sigma1": w.spec.market_params.sigma1,
                    "sigma2": w.spec.market_params.sigma2,
                },
                "own_regime": own_metrics,
                "heldout_regime": heldout_metrics,
                "final_params": w.get_param_dict(),
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    own_mv = [r["own_regime"]["mv"] for r in rows]
    held_mv = [r["heldout_regime"]["mv"] for r in rows]
    own_cum = [r["own_regime"]["cum_return"] for r in rows]
    held_cum = [r["heldout_regime"]["cum_return"] for r in rows]

    return {
        "avg_own_mv": float(np.mean(own_mv)),
        "avg_heldout_mv": float(np.mean(held_mv)),
        "avg_own_cum_return": float(np.mean(own_cum)),
        "avg_heldout_cum_return": float(np.mean(held_cum)),
    }


def build_workers(specs: list[WorkerSpec], total_eps: int) -> list[FederatedWorker]:
    return [FederatedWorker(spec=s, n_total_episodes=total_eps, log_every=max(1, total_eps // 10)) for s in specs]


def run_local_baseline(
    worker_specs: list[WorkerSpec],
    heldout_spec: WorkerSpec,
    total_eps: int,
    n_eval: int,
) -> dict:
    workers = build_workers(worker_specs, total_eps)
    t0 = time.time()

    for w in workers:
        w.local_train(total_eps)

    elapsed = time.time() - t0
    rows = evaluate_workers("local", workers, heldout_spec, n_eval=n_eval)
    return {
        "method": "local",
        "elapsed_sec": elapsed,
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def run_fedavg(
    worker_specs: list[WorkerSpec],
    heldout_spec: WorkerSpec,
    total_eps: int,
    local_eps: int,
    n_eval: int,
) -> dict:
    workers = build_workers(worker_specs, total_eps)
    n_rounds = math.ceil(total_eps / local_eps)
    t0 = time.time()

    for rnd in range(n_rounds):
        remaining = total_eps - rnd * local_eps
        cur_local_eps = min(local_eps, remaining)
        if cur_local_eps <= 0:
            break

        for w in workers:
            w.local_train(cur_local_eps)

        param_dicts = [w.get_param_dict() for w in workers]
        global_avg = average_params(param_dicts, ALL_PARAM_NAMES)

        for w in workers:
            w.set_param_dict(global_avg)

        print(f"[FedAvg] round {rnd+1}/{n_rounds} done")

    elapsed = time.time() - t0
    rows = evaluate_workers("fedavg", workers, heldout_spec, n_eval=n_eval)
    return {
        "method": "fedavg",
        "elapsed_sec": elapsed,
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def run_pf_shared(
    worker_specs: list[WorkerSpec],
    heldout_spec: WorkerSpec,
    total_eps: int,
    local_eps: int,
    n_eval: int,
) -> dict:
    """
    Personalized federated variant:
    - local training on each worker
    - regime-aware weighted averaging
    - only shared parameters are synchronized globally
    """
    workers = build_workers(worker_specs, total_eps)
    n_rounds = math.ceil(total_eps / local_eps)
    t0 = time.time()

    for rnd in range(n_rounds):
        remaining = total_eps - rnd * local_eps
        cur_local_eps = min(local_eps, remaining)
        if cur_local_eps <= 0:
            break

        for w in workers:
            w.local_train(cur_local_eps)

        param_dicts = [w.get_param_dict() for w in workers]
        weights = regime_aware_weights(workers)
        shared_avg = average_params(param_dicts, SHARED_PARAM_NAMES, weights=weights)

        for w in workers:
            w.set_param_dict(shared_avg, only_names=SHARED_PARAM_NAMES)

        print(f"[PF-CT-RS-q] round {rnd+1}/{n_rounds} done; weights={np.round(weights, 3)}")

    elapsed = time.time() - t0
    rows = evaluate_workers("pf_ct_rsq", workers, heldout_spec, n_eval=n_eval)
    return {
        "method": "pf_ct_rsq",
        "elapsed_sec": elapsed,
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def print_summary_table(results: list[dict]):
    print("\n=== Federated Phase-2 summary ===")
    header = f"{'method':<14} {'own MV':>12} {'heldout MV':>12} {'own CR':>12} {'heldout CR':>12} {'sec':>10}"
    print(header)
    print("-" * len(header))
    for res in results:
        s = res["summary"]
        print(
            f"{res['method']:<14} "
            f"{s['avg_own_mv']:>12.4f} "
            f"{s['avg_heldout_mv']:>12.4f} "
            f"{s['avg_own_cum_return']:>12.4f} "
            f"{s['avg_heldout_cum_return']:>12.4f} "
            f"{res['elapsed_sec']:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=int, default=600, help="Total local episodes budget per worker.")
    parser.add_argument("--local-eps", type=int, default=100, help="Local episodes per federated round.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-eval", type=int, default=3000)
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "local", "fedavg", "pf"],
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    worker_specs, heldout_spec = default_worker_specs(seed=args.seed)
    results = []

    if args.method in ("all", "local"):
        results.append(
            run_local_baseline(
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                n_eval=args.n_eval,
            )
        )

    if args.method in ("all", "fedavg"):
        results.append(
            run_fedavg(
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                local_eps=args.local_eps,
                n_eval=args.n_eval,
            )
        )

    if args.method in ("all", "pf"):
        results.append(
            run_pf_shared(
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                local_eps=args.local_eps,
                n_eval=args.n_eval,
            )
        )

    print_summary_table(results)

    out = {
        "config": {
            "eps": args.eps,
            "local_eps": args.local_eps,
            "seed": args.seed,
            "n_eval": args.n_eval,
            "methods_run": [r["method"] for r in results],
        },
        "results": results,
    }

    out_path = RESULTS_DIR / "federated_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()