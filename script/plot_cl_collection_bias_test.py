#!/usr/bin/env python3
"""Plot the failure-biased and unbiased learning curves side by side.

Reads collection_bias_summary.json written by run_cl_collection_bias_test.py and
draws probe success rate and mean quality against dataset size, one line per
collection rule. Individual repeats are drawn as faint markers so the retraining
spread is visible next to the trend, which is the whole point of the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

STYLE = {
    "biased": {"color": "#c0392b", "marker": "o", "label": "failure-biased (as collected by CL)"},
    "unbiased": {"color": "#1f77b4", "marker": "s", "label": "unbiased (random scenarios)"},
    "dagger": {"color": "#2e8b57", "marker": "^", "label": "DAgger (MPC labels on S1's own states)"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="output/cl_collection_test/collection_bias_summary.json")
    p.add_argument("--out", default="output/cl_collection_test/collection_bias.png")
    p.add_argument("--title", default="Does the collection rule flatten the continual-learning curve? (bugtrap)")
    return p.parse_args()


def panel(ax, data: Dict[str, Dict[int, List[float]]], ylabel: str, title: str) -> None:
    for condition, style in STYLE.items():
        by_size = data.get(condition, {})
        sizes = sorted(by_size)
        if not sizes:
            continue
        for size in sizes:
            for value in by_size[size]:
                ax.plot(size, value, style["marker"], color=style["color"], alpha=0.28, markersize=5)
        means = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
        ax.plot(sizes, means, style["marker"] + "-", color=style["color"],
                linewidth=2, markersize=8, label=style["label"])
    ax.set_xlabel("training demonstrations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    payload = json.loads(summary_path.read_text())
    results = payload["results"]

    rates: Dict[str, Dict[int, List[float]]] = {}
    quality: Dict[str, Dict[int, List[float]]] = {}
    for entry in results.values():
        cond, size = entry["condition"], int(entry["size"])
        rates.setdefault(cond, {}).setdefault(size, []).append(float(entry["success_rate"]))
        q = float(entry["mean_quality"])
        if math.isfinite(q):
            quality.setdefault(cond, {}).setdefault(size, []).append(q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel(axes[0], rates, "probe success rate", "Success rate on the fixed 100-scenario probe")
    panel(axes[1], quality, "mean quality (successes)", "Trajectory quality")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
