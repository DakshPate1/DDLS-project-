"""Phase 3 decentralized experiment.

Runs 5 methods across the same 4-worker heterogeneous setting as Phase 2:
  local       — no communication (baseline)
  fedavg      — central-server full-parameter averaging
  pf_ct_rsq   — central-server partial-parameter averaging (Phase 2 best)
  dec_ring    — gossip consensus on ring topology (no server)
  dec_fc      — gossip consensus on fully-connected topology (no server)

Usage:
  python -m experiments.decentralized_exp --eps 600 --local-eps 100 --n-eval 3000
  python -m experiments.decentralized_exp --topology ring --eps 300 --local-eps 50
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.federated_exp import (
    build_workers,
    default_worker_specs,
    evaluate_workers,
    run_fedavg,
    run_local_baseline,
    run_pf_shared,
    summarize_rows,
)
from src.decentralized import fc_adjacency, gossip_step, mh_weights, ring_adjacency


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def run_dec_gossip(
    topology: str,
    worker_specs,
    heldout_spec,
    total_eps: int,
    local_eps: int,
    n_eval: int,
) -> dict:
    """Decentralized gossip training — no central server.

    Workers train locally for local_eps episodes, then exchange shared
    parameters with their graph neighbors using Metropolis-Hastings weights.
    ψ and unshared θ remain local throughout.
    """
    assert topology in ("ring", "fc"), f"unknown topology: {topology!r}"
    method_name = f"dec_{topology}"

    workers = build_workers(worker_specs, total_eps)
    n = len(workers)

    adj = ring_adjacency(n) if topology == "ring" else fc_adjacency(n)
    W = mh_weights(adj)
    n_rounds = math.ceil(total_eps / local_eps)
    t0 = time.time()

    for rnd in range(n_rounds):
        remaining = total_eps - rnd * local_eps
        cur_local_eps = min(local_eps, remaining)
        if cur_local_eps <= 0:
            break

        for w in workers:
            w.local_train(cur_local_eps)

        gossip_step(workers, W)
        print(f"[Dec-{topology}] round {rnd+1}/{n_rounds} done")

    elapsed = time.time() - t0
    rows = evaluate_workers(method_name, workers, heldout_spec, n_eval=n_eval)
    return {
        "method": method_name,
        "elapsed_sec": elapsed,
        "topology": topology,
        "mh_weights": W.tolist(),
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def print_summary_table(results: list[dict]):
    print("\n=== Decentralized Phase-3 summary ===")
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
    parser.add_argument("--eps", type=int, default=600, help="Total episodes per worker.")
    parser.add_argument("--local-eps", type=int, default=100, help="Local episodes per gossip round.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-eval", type=int, default=3000)
    parser.add_argument(
        "--topology",
        type=str,
        default="all",
        choices=["all", "ring", "fc"],
        help="Which decentralized topology to run. 'all' also runs the Phase-2 baselines.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip local/fedavg/pf baselines (useful when re-running only dec topologies).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    worker_specs, heldout_spec = default_worker_specs(seed=args.seed)
    results = []

    if not args.skip_baselines:
        results.append(
            run_local_baseline(
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                n_eval=args.n_eval,
            )
        )
        results.append(
            run_fedavg(
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                local_eps=args.local_eps,
                n_eval=args.n_eval,
            )
        )
        results.append(
            run_pf_shared(
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                local_eps=args.local_eps,
                n_eval=args.n_eval,
            )
        )

    if args.topology in ("all", "ring"):
        results.append(
            run_dec_gossip(
                topology="ring",
                worker_specs=worker_specs,
                heldout_spec=heldout_spec,
                total_eps=args.eps,
                local_eps=args.local_eps,
                n_eval=args.n_eval,
            )
        )

    if args.topology in ("all", "fc"):
        results.append(
            run_dec_gossip(
                topology="fc",
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

    out_path = RESULTS_DIR / f"decentralized_metrics_eps{args.eps}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
