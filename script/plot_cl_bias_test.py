#!/usr/bin/env python3
"""Plot the failure-biased CL curve against the unbiased control at matched sizes.

Both curves start from the same base checkpoint and are evaluated on the same
fixed probe set, so the only difference is which scenarios the demonstrations
came from. A rising unbiased curve beside a flat biased one implicates the
collection rule; two flat curves implicate the size of the per-block increments
relative to retraining noise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

STYLE = {
    "unbiased": {"color": "#1b7f4f", "marker": "o", "label": "unbiased (random scenarios)"},
    "biased": {"color": "#b3402f", "marker": "s", "label": "failure-biased (CL collection rule)"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default="output/cl_bias_test/bias_test_summary.json")
    p.add_argument("--out", default="output/cl_bias_test/cl_collection_bias.png")
    p.add_argument("--dpi", type=int, default=170)
    return p.parse_args()


def series(rows: List[Dict[str, Any]], key: str) -> tuple[List[float], List[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(key, math.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            xs.append(float(row["demos"]))
            ys.append(value)
    return xs, ys


def panel(ax, data: Dict[str, Any], key: str, title: str, ylabel: str) -> None:
    biased = [row for row in data.get("biased", []) if row["demos"] > 0]
    unbiased = data.get("unbiased", [])
    base = next((row for row in data.get("biased", []) if row["block"] == -1), None)

    if base is not None:
        value = float(base.get(key, math.nan))
        if math.isfinite(value):
            ax.axhline(value, color="#555555", linestyle=":", linewidth=1.3,
                       label=f"shared base checkpoint ({value:.2f})")

    for name, rows in (("unbiased", unbiased), ("biased", biased)):
        xs, ys = series(rows, key)
        if not xs:
            continue
        style = STYLE[name]
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], markersize=6,
                linewidth=1.9, label=style["label"])

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("cumulative demonstrations in the training set")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary = json.loads(summary_path.read_text())
    solvers = [name for name, data in summary.get("solvers", {}).items() if data.get("unbiased")]
    if not solvers:
        raise SystemExit(f"no completed unbiased arms in {summary_path}")

    fig, axes = plt.subplots(2, len(solvers), figsize=(5.6 * len(solvers), 8.0), squeeze=False)
    for column, solver in enumerate(solvers):
        data = summary["solvers"][solver]
        pool = data.get("pool_size", 0)
        panel(axes[0][column], data, "success_rate",
              f"{solver.upper()} teacher — probe success rate", "success rate on fixed probe set")
        panel(axes[1][column], data, "mean_quality",
              f"{solver.upper()} teacher — probe trajectory quality", "mean quality (successes only)")
        axes[0][column].text(0.02, 0.03, f"unbiased pool: {pool} demos", transform=axes[0][column].transAxes,
                             fontsize=8, color="#444444")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Does the CL collection rule or the retraining noise floor flatten the probe curve?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"[plot] {out_path}")


if __name__ == "__main__":
    main()
