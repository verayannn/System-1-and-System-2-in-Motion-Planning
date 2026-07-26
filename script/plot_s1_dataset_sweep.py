#!/usr/bin/env python3
"""Plot the S1 dataset-size sweep written by run_s1_dataset_sweep.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILTER_STYLE = {
    "greedy": {"color": "#c0392b", "marker": "o", "label": "greedy filter (original)"},
    "policy": {"color": "#1f6feb", "marker": "s", "label": "policy filter (deviation-minimising)"},
}


def collect(summary: Dict, metric: str) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    for row in summary["results"].values():
        value = row.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        out.setdefault(row["filter"], {}).setdefault(int(row["size"]), []).append(float(value))
    return out


def panel(ax, data: Dict[str, Dict[int, List[float]]], *, title: str, ylabel: str, cases: int,
          show_noise: bool) -> None:
    for filter_mode, by_size in sorted(data.items()):
        style = FILTER_STYLE.get(filter_mode, {"color": "gray", "marker": "^", "label": filter_mode})
        sizes = sorted(by_size)
        means = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
        ax.plot(sizes, means, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=7, label=style["label"], zorder=3)
        for s in sizes:
            ax.scatter([s] * len(by_size[s]), by_size[s], color=style["color"],
                       alpha=0.35, s=28, zorder=2, edgecolors="none")

    if show_noise and cases:
        se = math.sqrt(0.3 * 0.7 / cases)
        ax.axhspan(0, 0, color="none")
        ax.text(0.02, 0.03, f"±1 s.e. at rate 0.3 with n={cases} is {se:.3f}",
                transform=ax.transAxes, fontsize=8, color="#555555")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("MPC demonstrations used for training")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.grid(alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="output/s1_dataset_sweep/sweep_summary.json")
    p.add_argument("--out", default="output/s1_dataset_sweep/s1_dataset_sweep.png")
    a = p.parse_args()

    summary = json.loads(Path(a.summary).read_text())
    cases = next((int(r["cases"]) for r in summary["results"].values() if r.get("cases")), 100)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    panel(axes[0], collect(summary, "success_rate"),
          title="Does more data help S1 reach the goal?",
          ylabel="success rate", cases=cases, show_noise=True)
    panel(axes[1], collect(summary, "mean_quality"),
          title="Trajectory quality of the successes",
          ylabel="mean quality (duration-invariant)", cases=cases, show_noise=False)

    axes[0].set_ylim(bottom=0)
    ticks = sorted({int(r["size"]) for r in summary["results"].values()})
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")

    fig.suptitle(
        f"S1 neural on {summary.get('family', '?')}: success responds to dataset size only once the "
        f"safety filter stops overriding the policy",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
