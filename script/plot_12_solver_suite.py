#!/usr/bin/env python3
"""

Plot success/runtime summaries from script/run_12_solver_suite.py.


cd /Users/apple/Desktop/sofai

PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/plot_12_solver_suite.py \
  --summary_csv output/benchmark_runs/twelve_solver_suite/twelve_solver_tables.csv \
  --out_dir output/benchmark_runs/twelve_solver_suite/plots

  
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary_csv", default="output/benchmark_runs/twelve_solver_suite/twelve_solver_tables.csv")
    p.add_argument("--out_dir", default="output/benchmark_runs/twelve_solver_suite/plots")
    args = p.parse_args()

    summary_csv = Path(args.summary_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(summary_csv)
    envs = []
    for row in rows:
        env = row["environment"]
        if env not in envs:
            envs.append(env)

    for env in envs:
        group = [r for r in rows if r["environment"] == env]
        labels = [r["solver"] for r in group]
        success = [100.0 * parse_float(r["success_rate"]) for r in group]
        runtime = [parse_float(r["mean_runtime"]) for r in group]

        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.bar(range(len(labels)), success)
        ax.set_title(f"{env}: success rate")
        ax.set_ylabel("success (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.savefig(out_dir / f"{env}_success_rate.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.bar(range(len(labels)), runtime)
        ax.set_title(f"{env}: mean runtime")
        ax.set_ylabel("seconds")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.savefig(out_dir / f"{env}_mean_runtime.png", dpi=200)
        plt.close(fig)

    print(f"[write] plots: {out_dir}")


if __name__ == "__main__":
    main()
