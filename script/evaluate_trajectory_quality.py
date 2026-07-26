#!/usr/bin/env python3
"""Compare solver arms under the duration-invariant trajectory quality index.

Usage:
    evaluate_trajectory_quality.py --arm mpc=path/to/mpc_runs.jsonl \
                                   --arm cbf=path/to/cbf_runs.jsonl \
                                   --arm nn=path/to/nn_runs.jsonl

Reports per-arm marginals over every successful run and, separately, a paired
comparison restricted to scenarios that every arm solved. The paired view is
the one to trust: arms have different success sets, and the easy scenarios that
a weaker arm solves are not the same population the stronger arm is scored on.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from solvers.trajectory_quality import (
    QUALITY_DEFINITION_VERSION,
    evaluate_trajectory,
    shortest_free_path_length,
)

INDEX_KEYS = ("quality_score", "path_efficiency", "smoothness", "clearance_score")
DIAGNOSTIC_KEYS = (
    "path_length",
    "duration_sec",
    "mean_speed",
    "min_clearance",
    "sparc",
    "ldlj",
    "smoothness_ldlj",
    "peak_control_ratio",
    "control_saturation_frac",
)


def selected_success_attempt(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = record.get("attempts") or []
    successful = [a for a in attempts if a.get("success")]
    if not successful:
        return None
    return successful[-1]


def load_arm(path: Path, ref_cache: Dict[Any, Optional[float]]) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("success"):
                continue
            attempt = selected_success_attempt(record)
            if attempt is None:
                continue
            scenario = record.get("scenario")
            if not isinstance(scenario, dict):
                continue
            sid = int(record.get("scenario_id", record.get("scenario_index", -1)))

            key = (
                tuple(np.round(np.asarray(scenario.get("start", []), dtype=float), 6).tolist()),
                tuple(np.round(np.asarray(scenario.get("goal", []), dtype=float), 6).tolist()),
                len(scenario.get("rectangles") or []),
                sid,
            )
            if key not in ref_cache:
                ref_cache[key] = shortest_free_path_length(
                    scenario.get("start", [0.0, 0.0]),
                    scenario.get("goal", [0.0, 0.0]),
                    scenario.get("rectangles") or [],
                    scenario.get("bounds") or [-10.0, -10.0, 10.0, 10.0],
                )

            metrics = evaluate_trajectory(
                attempt.get("states"),
                attempt.get("dt", 0.075),
                scenario,
                inputs=attempt.get("inputs"),
                reference_path_length=ref_cache[key],
            )
            if metrics is None:
                continue
            metrics["legacy_quality_score"] = float(record.get("quality_score", float("nan")))
            metrics["legacy_mpc_cost"] = float(record.get("quality_j", float("nan")))
            metrics["planning_runtime_sec"] = float(record.get("planning_runtime_sec", float("nan")))
            metrics["dt"] = float(attempt.get("dt", 0.075))
            rows[sid] = metrics
    return rows


def _median(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(st.median(finite)) if finite else float("nan")


def _mean(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(st.fmean(finite)) if finite else float("nan")


def _cell(values: List[float]) -> str:
    return f"{_median(values):.3f}/{_mean(values):.3f}"


def wilcoxon(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value, or None if scipy is unavailable."""
    try:
        from scipy.stats import wilcoxon as _w
    except Exception:
        return None
    diffs = [x - y for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
    if len(diffs) < 6 or all(abs(d) < 1e-12 for d in diffs):
        return None
    try:
        return float(_w(diffs).pvalue)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=RUNS_JSONL")
    parser.add_argument("--total", type=int, default=0, help="Total scenarios attempted, for success rate.")
    parser.add_argument("--label", default="", help="Optional heading for the report.")
    args = parser.parse_args()

    ref_cache: Dict[Any, Optional[float]] = {}
    arms: Dict[str, Dict[int, Dict[str, float]]] = {}
    for spec in args.arm:
        name, _, raw_path = spec.partition("=")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise SystemExit(f"missing runs file for arm {name!r}: {path}")
        arms[name] = load_arm(path, ref_cache)

    heading = f"Trajectory quality report [{QUALITY_DEFINITION_VERSION}]"
    if args.label:
        heading += f" -- {args.label}"
    print("=" * 100)
    print(heading)
    print("=" * 100)

    print("\n[1] Per-arm marginals over all solved scenarios (median/mean)")
    header = f"{'arm':>6} {'n':>4} " + " ".join(f"{k[:11]:>13}" for k in INDEX_KEYS)
    print(header)
    print("-" * len(header))
    for name, rows in arms.items():
        vals = list(rows.values())
        cells = " ".join(f"{_cell([v[k] for v in vals]):>13}" for k in INDEX_KEYS)
        print(f"{name:>6} {len(vals):>4} {cells}")

    print("\n[2] Diagnostics (median), reported outside the quality index")
    header = f"{'arm':>6} " + " ".join(f"{k[:11]:>13}" for k in DIAGNOSTIC_KEYS)
    print(header)
    print("-" * len(header))
    for name, rows in arms.items():
        vals = list(rows.values())
        cells = " ".join(f"{_median([v.get(k, float('nan')) for v in vals]):>13.3f}" for k in DIAGNOSTIC_KEYS)
        print(f"{name:>6} {cells}")

    common = set.intersection(*(set(rows) for rows in arms.values())) if arms else set()
    common_sorted = sorted(common)
    print(f"\n[3] Paired comparison on {len(common_sorted)} scenarios solved by every arm")
    if not common_sorted:
        print("    (no shared scenarios)")
        return

    header = (
        f"{'arm':>6} " + " ".join(f"{k[:11]:>13}" for k in INDEX_KEYS) + f" {'legacyQ':>10} {'legacyJ':>10}"
    )
    print(header)
    print("-" * len(header))
    paired: Dict[str, Dict[str, List[float]]] = {}
    for name, rows in arms.items():
        paired[name] = {
            k: [rows[sid].get(k, float("nan")) for sid in common_sorted]
            for k in (*INDEX_KEYS, "legacy_quality_score", "legacy_mpc_cost", *DIAGNOSTIC_KEYS)
        }
        cells = " ".join(f"{_cell(paired[name][k]):>13}" for k in INDEX_KEYS)
        print(
            f"{name:>6} {cells} "
            f"{_median(paired[name]['legacy_quality_score']):>10.3f} "
            f"{_median(paired[name]['legacy_mpc_cost']):>10.1f}"
        )

    names = list(arms)
    print("\n[4] Per-scenario win rate on the quality index (row beats column)")
    header = f"{'':>6} " + " ".join(f"{n:>10}" for n in names)
    print(header)
    print("-" * len(header))
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append(f"{'-':>10}")
                continue
            wins = sum(
                1
                for x, y in zip(paired[a]["quality_score"], paired[b]["quality_score"])
                if np.isfinite(x) and np.isfinite(y) and x > y
            )
            cells.append(f"{wins / len(common_sorted):>10.2f}")
        print(f"{a:>6} " + " ".join(cells))

    print("\n[5] Wilcoxon signed-rank vs the best arm on the quality index")
    best = max(names, key=lambda n: _median(paired[n]["quality_score"]))
    print(f"    best arm by median quality index: {best}")
    for other in names:
        if other == best:
            continue
        p = wilcoxon(paired[best]["quality_score"], paired[other]["quality_score"])
        delta = _median(paired[best]["quality_score"]) - _median(paired[other]["quality_score"])
        p_text = "n/a (scipy missing)" if p is None else f"p={p:.2e}"
        print(f"    {best} vs {other:>6}: median delta={delta:+.3f}  {p_text}")


if __name__ == "__main__":
    main()
