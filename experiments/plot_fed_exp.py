from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_summary_df(files: list[Path]) -> pd.DataFrame:
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

    files = [
        RESULTS_DIR / "federated_metrics.json",         # 100 eps
        RESULTS_DIR / "federated_metrics_eps300.json",  # 300 eps
        RESULTS_DIR / "federated_metrics_eps600.json",  # 600 eps
    ]

    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required result files:\n" + "\n".join(missing)
        )

    df = build_summary_df(files)

    label_map = {
        "local": "Local",
        "fedavg": "FedAvg",
        "pf_ct_rsq": "PF-CT-RS-q",
    }

    plt.figure(figsize=(8, 5.5))

    for method, group in df.groupby("method"):
        group = group.sort_values("eps")
        plt.plot(
            group["eps"],
            group["avg_heldout_mv"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=label_map.get(method, method),
        )

        # annotate final point at eps=600
        last_row = group.iloc[-1]
        plt.annotate(
            f"{last_row['avg_heldout_mv']:.4f}",
            (last_row["eps"], last_row["avg_heldout_mv"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    plt.xlabel("Training episodes per worker", fontsize=11)
    plt.ylabel("Average held-out MV", fontsize=11)
    plt.title("Held-out Mean-Variance Performance Across Training Budgets", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = PLOTS_DIR / "federated_main_figure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    df.to_csv(PLOTS_DIR / "federated_main_figure_data.csv", index=False)

    print(f"Saved main figure to: {out_path}")
    print(f"Saved figure data to: {PLOTS_DIR / 'federated_main_figure_data.csv'}")


if __name__ == "__main__":
    main()