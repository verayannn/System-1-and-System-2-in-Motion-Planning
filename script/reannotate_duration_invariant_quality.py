#!/usr/bin/env python3
"""Recompute existing benchmark quality without the time-coupled effort axis.

The resulting ``duration_invariant_v1`` score is the geometric mean of:

* shortest-path efficiency;
* SPARC smoothness; and
* obstacle clearance.

It intentionally excludes control effort, duration, and runtime. Existing
``*_runs.jsonl`` and matching ``*_summary.csv`` files are updated in place.
Use ``--backup_dir`` (the default) to retain their prior annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_motion_planning_benchmarks import CSV_FIELDS, flat  # noqa: E402
from solvers._s2_common import benchmark_family_from_dictionary, selected_success_attempt  # noqa: E402
from solvers.trajectory_quality import (  # noqa: E402
    DURATION_INVARIANT_QUALITY_VERSION,
    evaluate_trajectory,
    shortest_free_path_length,
)


QUALITY_COLUMNS = {
    "quality_score": None,
    "quality_path_efficiency": "path_efficiency",
    "quality_smoothness": "smoothness",
    "quality_clearance": "clearance_score",
    "quality_path_length": "path_length",
    "quality_reference_path_length": "reference_path_length",
    "quality_min_clearance": "min_clearance",
    "quality_sparc": "sparc",
    "quality_ldlj": "ldlj",
    "quality_smoothness_ldlj": "smoothness_ldlj",
    "quality_duration_sec": "duration_sec",
    "quality_mean_speed": "mean_speed",
    "quality_peak_control_ratio": "peak_control_ratio",
    "quality_control_saturation_frac": "control_saturation_frac",
}
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temp_path.replace(path)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(flat(row))


def reference_key(scenario: dict[str, Any], scenario_id: Any) -> tuple[Any, ...]:
    return (
        int(scenario_id),
        tuple(scenario.get("start") or []),
        tuple(scenario.get("goal") or []),
        tuple(tuple(rect) for rect in scenario.get("rectangles") or []),
        tuple(scenario.get("bounds") or []),
    )


def annotate(rows: list[dict[str, Any]]) -> tuple[int, list[float]]:
    references: dict[tuple[Any, ...], float | None] = {}
    scores: list[float] = []
    family = benchmark_family_from_dictionary(rows[0].get("dictionary", "")) if rows else ""

    for result in rows:
        result["quality_family"] = family
        result["quality_definition"] = DURATION_INVARIANT_QUALITY_VERSION
        result.pop("quality_effort", None)
        result.pop("quality_specific_effort", None)

        sample = None
        if result.get("success"):
            attempt = selected_success_attempt(result)
            scenario = result.get("scenario")
            if attempt is not None and isinstance(scenario, dict):
                key = reference_key(scenario, result.get("scenario_id", result.get("scenario_index", -1)))
                if key not in references:
                    references[key] = shortest_free_path_length(
                        scenario.get("start", [0.0, 0.0]),
                        scenario.get("goal", [0.0, 0.0]),
                        scenario.get("rectangles") or [],
                        scenario.get("bounds") or [-10.0, -10.0, 10.0, 10.0],
                    )
                sample = evaluate_trajectory(
                    attempt.get("states"),
                    float(attempt.get("dt", 0.075)),
                    scenario,
                    inputs=attempt.get("inputs"),
                    reference_path_length=references[key],
                )

        for column, key in QUALITY_COLUMNS.items():
            if sample is None:
                result[column] = None
            elif key is None:
                result[column] = float(sample["quality_score"])
            else:
                value = sample.get(key)
                result[column] = float(value) if value is not None and math.isfinite(float(value)) else None

        if result.get("quality_score") is not None:
            scores.append(float(result["quality_score"]))

    return len(references), scores


def backup(path: Path, root: Path, backup_dir: Path) -> None:
    destination = backup_dir / path.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=ROOT / "output/benchmark_runs_current",
        help="Root containing benchmark suite results.",
    )
    parser.add_argument(
        "--backup_dir",
        type=Path,
        default=None,
        help="Where to copy the prior JSONL/CSV files; defaults under results_dir/analysis.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    backup_dir = (
        args.backup_dir.expanduser().resolve()
        if args.backup_dir is not None
        else results_dir / "analysis" / "quality_backups_previous_definition"
    )
    # Backups and generated reports live under ``analysis``. Never treat them
    # as source benchmark runs, otherwise a repeat invocation rewrites the
    # backup snapshot rather than only the active suite results.
    run_files = sorted(
        path
        for path in results_dir.rglob("*_runs.jsonl")
        if "analysis" not in path.relative_to(results_dir).parts
    )
    if not run_files:
        raise SystemExit(f"No *_runs.jsonl files under {results_dir}")

    report_rows: list[dict[str, Any]] = []
    for runs_path in run_files:
        summary_path = runs_path.with_name(runs_path.name.replace("_runs.jsonl", "_summary.csv"))
        rows = read_jsonl(runs_path)
        reference_count, scores = annotate(rows)
        success_count = sum(bool(row.get("success")) for row in rows)
        report_rows.append(
            {
                "runs_file": str(runs_path.relative_to(results_dir)),
                "cases": len(rows),
                "successes": success_count,
                "mean_quality": sum(scores) / len(scores) if scores else None,
                "median_quality": statistics.median(scores) if scores else None,
                "reference_paths": reference_count,
                "quality_definition": DURATION_INVARIANT_QUALITY_VERSION,
            }
        )

        print(
            f"[annotate] {runs_path.relative_to(results_dir)} "
            f"success={success_count}/{len(rows)} "
            f"mean_quality={report_rows[-1]['mean_quality']!s}"
        )
        if args.dry_run:
            continue

        backup(runs_path, results_dir, backup_dir)
        if summary_path.exists():
            backup(summary_path, results_dir, backup_dir)
        write_jsonl_atomic(runs_path, rows)
        write_summary(summary_path, rows)

    if not args.dry_run:
        report_path = results_dir / "analysis" / "duration_invariant_quality_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"[write] {report_path}")
        print(f"[backup] {backup_dir}")


if __name__ == "__main__":
    main()
