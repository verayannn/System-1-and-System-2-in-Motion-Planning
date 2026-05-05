#!/usr/bin/env python3
"""Run the dense-clutter benchmark suite for all main S1/S2 configurations.

This is a wrapper around the updated run_motion_planning_benchmarks.py.
It runs all dense-clutter scenarios for six configurations:

    1. SOFAI: S1 primitives -> S2 MPC fallback
    2. SOFAI: S1 neural     -> S2 MPC fallback
    3. S1 only: primitives
    4. S1 only: neural
    5. S2 only: MPC 
    6. S2 only: CBF

For each configuration, it calls run_motion_planning_benchmarks.py with only
--s1, --s2, and --run_type changed. Then it reads the generated CSV summaries
and writes one comparison CSV containing success rates and runtime statistics.

Example:
    cd /Users/apple/Desktop/sofai

    PYTHONDONTWRITEBYTECODE=1 \
    /Users/apple/miniconda3/envs/s12_env/bin/python3.10 run_dense_clutter_benchmark_suite.py \
      --timeout_sec 300

Optional histogram generation:
    PYTHONDONTWRITEBYTECODE=1 \
    /Users/apple/miniconda3/envs/s12_env/bin/python3.10 run_dense_clutter_benchmark_suite.py \
      --timeout_sec 300 \
      --make_histograms
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DENSE_CLUTTER_DICTIONARY = "benchmark_dualmp_dense_clutter.json"


CONFIGS = [
    {
        "label": "sofai_primitives_mpc",
        "display_name": "SOFAI: S1 primitives + S2 MPC",
        "run_type": "sofai",
        "s1": "primitives",
        "s2": "mpc",
    },
    {
        "label": "sofai_neural_mpc",
        "display_name": "SOFAI: S1 neural + S2 MPC",
        "run_type": "sofai",
        "s1": "neural",
        "s2": "mpc",
    },
    {
        "label": "s1_primitives",
        "display_name": "S1 only: primitives",
        "run_type": "s1",
        "s1": "primitives",
        "s2": "mpc",  # ignored by run_type=s1, but kept for CLI compatibility
    },
    {
        "label": "s1_neural",
        "display_name": "S1 only: neural",
        "run_type": "s1",
        "s1": "neural",
        "s2": "mpc",  # ignored by run_type=s1, but kept for CLI compatibility
    },
    {
        "label": "s2_mpc",
        "display_name": "S2 only: MPC",
        "run_type": "s2",
        "s1": "primitives",  # ignored by run_type=s2, but kept for CLI compatibility
        "s2": "mpc",
    },
    {
        "label": "s2_cbf",
        "display_name": "S2 only: CBF",
        "run_type": "s2",
        "s1": "primitives",  # ignored by run_type=s2, but kept for CLI compatibility
        "s2": "cbf",
    },
]


def bool_from_csv(value: object) -> bool:
    """Parse booleans written by csv.DictWriter."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def median(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(statistics.median(values)) if values else None


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    weight = pos - lo
    return float(xs[lo] * (1.0 - weight) + xs[hi] * weight)


def read_summary_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_rows(config: Dict[str, str], rows: List[Dict[str, str]]) -> Dict[str, object]:
    total_cases = len(rows)
    ok_rows = [r for r in rows if r.get("status", "") == "ok"]
    ok_cases = len(ok_rows)
    timeout_cases = sum(bool_from_csv(r.get("timed_out")) for r in rows)
    error_cases = sum(1 for r in rows if r.get("status", "") == "error")

    success_all = sum(bool_from_csv(r.get("success")) for r in rows)
    success_ok = sum(bool_from_csv(r.get("success")) for r in ok_rows)
    collision_free_ok = sum(bool_from_csv(r.get("collision_free")) for r in ok_rows)
    goal_reached_ok = sum(bool_from_csv(r.get("goal_reached")) for r in ok_rows)

    # runtime_sec is the planner/runtime measured by run_motion_planning_benchmarks.py.
    # wall_runtime_sec includes child-process spawn overhead, so it is kept separately.
    runtime_values = [
        v for v in (float_or_none(r.get("runtime_sec")) for r in ok_rows) if v is not None
    ]
    selected_runtime_values = [
        v for v in (float_or_none(r.get("selected_runtime_sec")) for r in ok_rows) if v is not None
    ]
    wall_runtime_values = [
        v for v in (float_or_none(r.get("wall_runtime_sec")) for r in ok_rows) if v is not None
    ]

    s1_attempted = sum(bool_from_csv(r.get("s1_attempted")) for r in rows)
    s1_success = sum(bool_from_csv(r.get("s1_success")) for r in rows)
    s2_attempted = sum(bool_from_csv(r.get("s2_attempted")) for r in rows)
    s2_success = sum(bool_from_csv(r.get("s2_success")) for r in rows)

    def safe_rate(num: int, den: int) -> Optional[float]:
        return float(num / den) if den else None

    return {
        "label": config["label"],
        "display_name": config["display_name"],
        "run_type": config["run_type"],
        "s1": config["s1"],
        "s2": config["s2"],
        "total_cases": total_cases,
        "ok_cases": ok_cases,
        "timeout_cases": timeout_cases,
        "error_cases": error_cases,
        # Treat timeout/error as failure. This is usually the headline number.
        "success_rate": safe_rate(success_all, total_cases),
        # Success among cases that actually returned status=ok.
        "success_rate_ok_only": safe_rate(success_ok, ok_cases),
        "collision_free_rate_ok_only": safe_rate(collision_free_ok, ok_cases),
        "goal_reached_rate_ok_only": safe_rate(goal_reached_ok, ok_cases),
        "avg_runtime_sec": mean(runtime_values),
        "median_runtime_sec": median(runtime_values),
        "p90_runtime_sec": percentile(runtime_values, 90),
        "avg_selected_runtime_sec": mean(selected_runtime_values),
        "avg_wall_runtime_sec": mean(wall_runtime_values),
        "s1_attempted_cases": s1_attempted,
        "s1_success_cases": s1_success,
        "s1_success_rate_when_attempted": safe_rate(s1_success, s1_attempted),
        "s2_attempted_cases": s2_attempted,
        "s2_success_cases": s2_success,
        "s2_success_rate_when_attempted": safe_rate(s2_success, s2_attempted),
    }


def write_comparison_csv(path: Path, summaries: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "display_name",
        "run_type",
        "s1",
        "s2",
        "total_cases",
        "ok_cases",
        "timeout_cases",
        "error_cases",
        "success_rate",
        "success_rate_ok_only",
        "collision_free_rate_ok_only",
        "goal_reached_rate_ok_only",
        "avg_runtime_sec",
        "median_runtime_sec",
        "p90_runtime_sec",
        "avg_selected_runtime_sec",
        "avg_wall_runtime_sec",
        "s1_attempted_cases",
        "s1_success_cases",
        "s1_success_rate_when_attempted",
        "s2_attempted_cases",
        "s2_success_cases",
        "s2_success_rate_when_attempted",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def format_pct(value: object) -> str:
    val = float_or_none(value)
    if val is None:
        return "n/a"
    return f"{100.0 * val:.1f}%"


def format_sec(value: object) -> str:
    val = float_or_none(value)
    if val is None:
        return "n/a"
    return f"{val:.3f}s"


def print_summary_table(summaries: List[Dict[str, object]]) -> None:
    print("\n=== Dense clutter benchmark comparison ===")
    header = f"{'mode':32s} {'success':>9s} {'avg runtime':>12s} {'median':>10s} {'p90':>10s} {'ok/total':>10s}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        ok_total = f"{s['ok_cases']}/{s['total_cases']}"
        print(
            f"{s['label']:32s} "
            f"{format_pct(s['success_rate']):>9s} "
            f"{format_sec(s['avg_runtime_sec']):>12s} "
            f"{format_sec(s['median_runtime_sec']):>10s} "
            f"{format_sec(s['p90_runtime_sec']):>10s} "
            f"{ok_total:>10s}"
        )
    print()


def run_config(
    *,
    root: Path,
    runner: Path,
    python_bin: str,
    out_root: Path,
    config: Dict[str, str],
    scenario_ids: str,
    timeout_sec: float,
    mplconfigdir: str,
    same_process: bool,
    dry_run: bool,
) -> Path:
    out_dir = out_root / config["label"]
    out_prefix = config["label"]
    summary_csv = out_dir / f"{out_prefix}_summary.csv"

    cmd = [
        python_bin,
        str(runner),
        "--patterns",
        DENSE_CLUTTER_DICTIONARY,
        "--scenario_ids",
        scenario_ids,
        "--s1",
        config["s1"],
        "--s2",
        config["s2"],
        "--run_type",
        config["run_type"],
        "--timeout_sec",
        str(timeout_sec),
        "--out_dir",
        str(out_dir),
        "--out_prefix",
        out_prefix,
        "--mplconfigdir",
        mplconfigdir,
    ]
    if same_process:
        cmd.append("--same_process")
    if dry_run:
        cmd.append("--dry_run")

    print("\n[config]", config["display_name"])
    print("[cmd]", " ".join(cmd))

    if dry_run:
        subprocess.run(cmd, cwd=root, check=True)
        return summary_csv

    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    subprocess.run(cmd, cwd=root, env=env, check=True)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Expected summary CSV was not created: {summary_csv}")
    return summary_csv


def maybe_make_histograms(out_root: Path, summaries_and_csvs: List[tuple[Dict[str, object], Path]]) -> None:
    """Optional runtime histogram figure for later inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_dir = out_root / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for summary, csv_path in summaries_and_csvs:
        rows = read_summary_csv(csv_path)
        values = [
            v
            for v in (float_or_none(r.get("runtime_sec")) for r in rows if r.get("status") == "ok")
            if v is not None
        ]
        if not values:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(values, bins=20)
        ax.set_title(f"Runtime distribution: {summary['label']}")
        ax.set_xlabel("runtime_sec")
        ax.set_ylabel("number of scenarios")
        fig.tight_layout()
        out_png = hist_dir / f"{summary['label']}_runtime_histogram.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"[write] {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--runner",
        default="run_motion_planning_benchmarks.py",
        help="Path to the updated benchmark runner, relative to --root unless absolute.",
    )
    parser.add_argument("--scenario_ids", default="all", help="Default: all dense-clutter scenarios.")
    parser.add_argument("--timeout_sec", type=float, default=300.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mplconfigdir", default="/private/tmp/mpl")
    parser.add_argument("--out_dir", default="output/benchmark_runs/dense_clutter_suite")
    parser.add_argument("--same_process", action="store_true", help="Forward --same_process to the runner.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--make_histograms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    runner = Path(args.runner).expanduser()
    if not runner.is_absolute():
        runner = root / runner
    if not runner.exists():
        raise FileNotFoundError(f"Cannot find benchmark runner: {runner}")

    out_root = Path(args.out_dir).expanduser()
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    summaries_and_csvs: List[tuple[Dict[str, object], Path]] = []

    for config in CONFIGS:
        summary_csv = run_config(
            root=root,
            runner=runner,
            python_bin=args.python,
            out_root=out_root,
            config=config,
            scenario_ids=args.scenario_ids,
            timeout_sec=args.timeout_sec,
            mplconfigdir=args.mplconfigdir,
            same_process=args.same_process,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue

        rows = read_summary_csv(summary_csv)
        summary = summarize_rows(config, rows)
        summaries.append(summary)
        summaries_and_csvs.append((summary, summary_csv))

    if args.dry_run:
        print("\n[dry_run] No comparison CSV written.")
        return

    comparison_csv = out_root / "dense_clutter_suite_comparison.csv"
    write_comparison_csv(comparison_csv, summaries)
    print_summary_table(summaries)
    print(f"[write] {comparison_csv}")

    if args.make_histograms:
        maybe_make_histograms(out_root, summaries_and_csvs)


if __name__ == "__main__":
    main()
