"""Generate figure 4: 5-method comparison across training budgets.

Loads decentralized_metrics_eps*.json files and plots:
  local, fedavg, pf_ct_rsq, dec_ring, dec_fc

Usage:
  python -m experiments.plot_dec_exp
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"


LABEL_MAP = {
    "local": "Local",
    "fedavg": "FedAvg",
    "pf_ct_rsq": "PF-CT-RS-q",
    "dec_ring": "Dec-CT-RS-q (ring)",
    "dec_fc": "Dec-CT-RS-q (FC)",
}

METHOD_ORDER = ["local", "fedavg", "pf_ct_rsq", "dec_ring", "dec_fc"]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_df(files: list[Path]) -> pd.DataFrame:
    rows = []
    for fp in files:
        data = load_json(fp)
        eps = data["config"]["eps"]
        for result in data["results"]:
            s = result["summary"]
            rows.append(
                {
                    "eps": eps,
                    "method": result["method"],
                    "avg_own_mv": s["avg_own_mv"],
                    "avg_heldout_mv": s["avg_heldout_mv"],
                    "avg_own_cum_return": s["avg_own_cum_return"],
                    "avg_heldout_cum_return": s["avg_heldout_cum_return"],
                }
            )
    return pd.DataFrame(rows).sort_values(["eps", "method"]).reset_index(drop=True)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RESULTS_DIR.glob("decentralized_metrics_eps*.json"))
    if not files:
        raise FileNotFoundError(
            "No decentralized_metrics_eps*.json files found in results/.\n"
            "Run: python -m experiments.decentralized_exp --eps 600"
        )

    df = build_df(files)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method in METHOD_ORDER:
        group = df[df["method"] == method].sort_values("eps")
        if group.empty:
            continue
        label = LABEL_MAP.get(method, method)
        style = "--" if method.startswith("dec_") else "-"
        marker = "s" if method.startswith("dec_") else "o"

        axes[0].plot(
            group["eps"], group["avg_heldout_mv"],
            linestyle=style, marker=marker, linewidth=2.5, markersize=7, label=label,
        )
        last = group.iloc[-1]
        axes[0].annotate(
            f"{last['avg_heldout_mv']:.4f}",
            (last["eps"], last["avg_heldout_mv"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

        axes[1].plot(
            group["eps"], group["avg_own_mv"],
            linestyle=style, marker=marker, linewidth=2.5, markersize=7, label=label,
        )
        last = group.iloc[-1]
        axes[1].annotate(
            f"{last['avg_own_mv']:.4f}",
            (last["eps"], last["avg_own_mv"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    for ax, title, ylabel in zip(
        axes,
        ["Held-out regime MV", "Own-regime MV"],
        ["Average held-out MV", "Average own-regime MV"],
    ):
        ax.set_xlabel("Training episodes per worker", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        "Local vs Federated vs Decentralized CT-RS-q (4 heterogeneous workers)",
        fontsize=13,
    )
    plt.tight_layout()

    out_path = PLOTS_DIR / "figure4_decentralized_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    df.to_csv(PLOTS_DIR / "figure4_data.csv", index=False)
    print(f"Saved figure to: {out_path}")
    print(f"Saved data to: {PLOTS_DIR / 'figure4_data.csv'}")

    # print summary table for last eps
    max_eps = df["eps"].max()
    last = df[df["eps"] == max_eps]
    print(f"\n--- eps={max_eps} summary ---")
    print(f"{'method':<22} {'own MV':>10} {'heldout MV':>12} {'own CR':>10} {'heldout CR':>12}")
    for method in METHOD_ORDER:
        row = last[last["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        print(
            f"{LABEL_MAP.get(method, method):<22} "
            f"{r['avg_own_mv']:>10.4f} "
            f"{r['avg_heldout_mv']:>12.4f} "
            f"{r['avg_own_cum_return']:>10.4f} "
            f"{r['avg_heldout_cum_return']:>12.4f}"
        )


if __name__ == "__main__":
    main()
