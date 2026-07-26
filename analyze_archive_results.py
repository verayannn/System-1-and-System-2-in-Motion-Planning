#!/usr/bin/env python3
"""Summarize and plot one current ``run_suite.py`` result directory.

Example:




  
PYTHONDONTWRITEBYTECODE=1 \
python analyze_archive_results.py \
  --suite_dir output/benchmark_runs/nl_dense_clutter_suite_v1 \
  --configs s1_neural s2_mpc sofai_mpc_cl\
  --block_size 50



"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from solvers._s2_common import resolve_mplconfigdir


LABELS = {
    "s1_neural": "S1 neural",
    "s2_cbf": "S2 CBF",
    "s2_mpc": "S2 MPC",
    "s2_mpc_do": "S2 do-mpc",
    "sofai_mpc_cl": "SOFAI MPC CL",
    "sofai_mpc_do_cl": "SOFAI do-mpc CL",
    "s2_mpc_do_cl": "SOFAI do-mpc CL",
}
COLORS = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]
DEFAULT_CONFIGS = ["s1_neural", "s2_mpc", "sofai_mpc_cl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite_dir", default="output/benchmark_runs/nl_bugtrap_suite")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="Defaults to S1 neural, S2 MPC, and SOFAI MPC CL.")
    parser.add_argument("--split_config", default="", help="Config for the S1/S2 success split panel.")
    parser.add_argument(
        "--runtime_field",
        default="planning_runtime_sec",
        help="End-to-end planner time: S1 plus S2 when strict fallback is used.",
    )
    parser.add_argument("--out_dir", default="", help="Defaults to <suite_dir>/analysis.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=0,
        help="Without a manifest, group sequential scenario ids into blocks of this size; 0 means one block.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def family_name(suite_dir: Path) -> str:
    name = suite_dir.name
    return name[3:-6] if name.startswith("nl_") and name.endswith("_suite") else name


def block_index(run: dict[str, Any], default: int) -> int:
    suffix = str(run.get("prefix", "")).rsplit("_block", 1)
    if len(suffix) == 2:
        digits = "".join(char for char in suffix[1] if char.isdigit())
        if digits:
            return int(digits)
    return default


def local_jsonl(suite_dir: Path, config: str, run: dict[str, Any]) -> Path:
    prefix = str(run.get("prefix", "")).strip()
    candidates = []
    if prefix:
        candidates.extend([
            suite_dir / config / "runs" / f"{prefix}_runs.jsonl",
            suite_dir / config / f"{prefix}_runs.jsonl",
        ])
    raw = str(run.get("jsonl", "")).strip()
    if raw:
        path = Path(raw).expanduser()
        candidates.append(path)
        if suite_dir.name in path.parts:
            index = path.parts.index(suite_dir.name)
            candidates.append(suite_dir.joinpath(*path.parts[index + 1 :]))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No JSONL found for {config}/{prefix or raw}")


def local_probe_jsonl(suite_dir: Path, config: str, run: dict[str, Any]) -> Path:
    """Resolve a probe JSONL even when a suite directory was moved."""
    prefix = str(run.get("probe_prefix", "")).strip()
    candidates = [suite_dir / config / "probe" / f"{prefix}_runs.jsonl"] if prefix else []
    raw = str(run.get("probe_jsonl") or run.get("jsonl") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        candidates.append(path)
        if suite_dir.name in path.parts:
            index = path.parts.index(suite_dir.name)
            candidates.append(suite_dir.joinpath(*path.parts[index + 1 :]))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No probe JSONL found for {config}/{prefix or raw}")


def block_lookup(manifest: dict[str, Any]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for block, scenario_ids in enumerate(manifest.get("blocks", [])):
        for scenario_id in scenario_ids:
            lookup[int(scenario_id)] = block
    return lookup


def rows_from_solver_files(suite_dir: Path, config: str, block_size: int = 0) -> list[tuple[int, dict[str, Any]]]:
    """Read result JSONLs directly when no suite manifest is available."""
    files = [
        path for path in sorted((suite_dir / config).rglob("*_runs.jsonl"))
        if "probe" not in path.parts
    ]
    if not files:
        raise FileNotFoundError(f"No result JSONL found in {suite_dir / config}")

    rows: list[tuple[int | None, dict[str, Any]]] = []
    for path in files:
        match = re.search(r"_block(\d+)", path.stem)
        explicit_block = int(match.group(1)) if match else None
        rows.extend((explicit_block, row) for row in read_jsonl(path))

    if any(block is not None for block, _ in rows):
        return [(0 if block is None else block, row) for block, row in rows]
    if block_size <= 0:
        return [(0, row) for _, row in rows]

    ordered = sorted(rows, key=lambda item: int(item[1].get("scenario_index", item[1].get("scenario_id", -1))))
    return [(index // block_size, row) for index, (_, row) in enumerate(ordered)]


def inferred_block_size(suite_dir: Path, configs: Iterable[str]) -> int:
    """Use explicit CL block files to partition standalone solver results."""
    counts = []
    for config in configs:
        for path in (suite_dir / config).rglob("*_block*_runs.jsonl"):
            count = sum(1 for line in path.open() if line.strip())
            if count:
                counts.append(count)
    return int(round(float(np.median(counts)))) if counts else 0


def inferred_block_count(suite_dir: Path, configs: Iterable[str]) -> int:
    blocks = []
    for config in configs:
        for path in (suite_dir / config).rglob("*_block*_runs.jsonl"):
            match = re.search(r"_block(\d+)", path.stem)
            if match:
                blocks.append(int(match.group(1)))
    return 1 + max(blocks) if blocks else 0


def probe_metrics_from_csv(suite_dir: Path, configs: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """Recover prior fixed-probe summaries when their raw JSONLs were moved."""
    path = suite_dir / "analysis" / "probe_metrics_by_block.csv"
    if not path.is_file():
        return {}

    wanted = set(configs)
    metrics: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            config = str(row.get("config", ""))
            if config not in wanted:
                continue
            try:
                parsed = {
                    "block": int(row["block"]),
                    "scenarios": int(row["scenarios"]),
                    "successes": int(row["successes"]),
                    "success_rate": float(row["success_rate"]),
                    "mean_runtime_sec": float(row["mean_runtime_sec"]),
                    "p90_runtime_sec": float(row["p90_runtime_sec"]),
                    "mean_quality": float(row["mean_quality"]),
                    "median_quality": float(row["median_quality"]),
                    "p90_quality": float(row["p90_quality"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            metrics.setdefault(config, []).append(parsed)
    return metrics


def rows_by_block(suite_dir: Path, manifest: dict[str, Any], config: str, block_size: int = 0) -> list[tuple[int, dict[str, Any]]]:
    config_data = manifest.get("configs", {}).get(config)
    lookup = block_lookup(manifest)
    if not config_data:
        # ``run_suite.py`` rewrites its manifest on every invocation. Preserve
        # standalone S1/S2 analysis when their JSONLs were generated separately.
        rows = rows_from_solver_files(suite_dir, config, block_size)
        return [
            (lookup.get(int(row.get("scenario_index", row.get("scenario_id", -1))), block), row)
            for block, row in rows
        ]

    runs = config_data.get("runs", [])
    rows: list[tuple[int, dict[str, Any]]] = []
    for default_block, run in enumerate(runs):
        path = local_jsonl(suite_dir, config, run)
        explicit_block = block_index(run, default_block)
        for row in read_jsonl(path):
            if config.endswith("_cl"):
                block = explicit_block
            else:
                scenario_id = int(row.get("scenario_index", row.get("scenario_id", -1)))
                block = lookup.get(scenario_id, explicit_block)
            rows.append((block, row))
    return rows


def probe_rows_by_block(suite_dir: Path, manifest: dict[str, Any], config: str) -> list[tuple[int, dict[str, Any]]]:
    """Return the base and post-retraining S1-only fixed-probe evaluations."""
    config_data = manifest.get("configs", {}).get(config, {})
    runs = config_data.get("runs", [])
    rows: list[tuple[int, dict[str, Any]]] = []
    base_probe = config_data.get("base_probe", {})
    if base_probe.get("jsonl") or base_probe.get("probe_jsonl") or base_probe.get("probe_prefix"):
        for row in read_jsonl(local_probe_jsonl(suite_dir, config, base_probe)):
            rows.append((-1, row))
    for default_block, run in enumerate(runs):
        if not run.get("probe_jsonl") and not run.get("probe_prefix"):
            continue
        block = block_index(run, default_block)
        for row in read_jsonl(local_probe_jsonl(suite_dir, config, run)):
            rows.append((block, row))
    return rows


def aggregate_probe(rows: Iterable[tuple[int, dict[str, Any]]], n_blocks: int, runtime_field: str) -> list[dict[str, Any]]:
    """Preserve the baseline as block -1, followed by evaluations after each CL update."""
    rows = list(rows)
    baseline = aggregate([(0, row) for block, row in rows if block == -1], 1, runtime_field)[0]
    baseline["block"] = -1
    updates = aggregate([(block, row) for block, row in rows if block >= 0], n_blocks, runtime_field)
    return [baseline, *updates] if baseline["scenarios"] else updates


def value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        try:
            candidate = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            return candidate
    return None


def quality(row: dict[str, Any]) -> float | None:
    """Recompute quality using the current definition when raw data is present."""
    # A versioned score emitted by an archive is authoritative. In particular,
    # duration_invariant_v1 intentionally omits the control-effort axis, while
    # the default live evaluator includes it.
    if str(row.get("quality_definition", "")) == "duration_invariant_v1":
        return value(row, "quality_score")
    if bool(row.get("success")):
        try:
            from solvers._s2_common import (
                benchmark_family_from_dictionary,
                quality_refs_for_result,
                quality_score,
                quality_weights_for_family,
                trajectory_quality_components,
            )

            sample = trajectory_quality_components(row)
            if sample is not None:
                family = benchmark_family_from_dictionary(str(row.get("dictionary", "")))
                return float(quality_score(sample, quality_refs_for_result(row), quality_weights_for_family(str(family))))
        except Exception:
            pass
    score = value(row, "quality_score")
    if score is not None:
        return score
    cost = value(row, "quality_j")
    return None if cost is None else 1.0 / (1.0 + cost)


def aggregate(rows: Iterable[tuple[int, dict[str, Any]]], n_blocks: int, runtime_field: str) -> list[dict[str, Any]]:
    buckets = [[] for _ in range(n_blocks)]
    for block, row in rows:
        if 0 <= block < n_blocks:
            buckets[block].append(row)

    summary = []
    for block, bucket in enumerate(buckets):
        runtimes = [runtime for row in bucket if (runtime := value(row, runtime_field, "selected_runtime_sec", "runtime_sec", "wall_runtime_sec")) is not None]
        qualities = [score for row in bucket if bool(row.get("success")) and (score := quality(row)) is not None]
        successes = sum(bool(row.get("success")) for row in bucket)
        summary.append(
            {
                "block": block,
                "scenarios": len(bucket),
                "successes": successes,
                "success_rate": successes / len(bucket) if bucket else math.nan,
                "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else math.nan,
                "p90_runtime_sec": float(np.percentile(runtimes, 90)) if runtimes else math.nan,
                "mean_quality": float(np.mean(qualities)) if qualities else math.nan,
                "median_quality": float(np.median(qualities)) if qualities else math.nan,
                "p90_quality": float(np.percentile(qualities, 90)) if qualities else math.nan,
            }
        )
    return summary


def s1_s2_split(rows: Iterable[tuple[int, dict[str, Any]]], n_blocks: int) -> list[dict[str, Any]]:
    counts = [{"s1": 0, "s2": 0, "failed": 0} for _ in range(n_blocks)]
    for block, row in rows:
        if not 0 <= block < n_blocks:
            continue
        attempts = row.get("attempts", []) or []
        s1_ok = any(attempt.get("system") == "s1" and bool(attempt.get("success")) for attempt in attempts)
        s2_ok = any(attempt.get("system") == "s2" and bool(attempt.get("success")) for attempt in attempts)
        if s1_ok:
            counts[block]["s1"] += 1
        elif s2_ok:
            counts[block]["s2"] += 1
        else:
            counts[block]["failed"] += 1

    return [
        {
            "block": block,
            "s1_success": count["s1"],
            "s2_only_success": count["s2"],
            "failed": count["failed"],
            "s1_fraction_of_success": count["s1"] / (count["s1"] + count["s2"])
            if count["s1"] + count["s2"]
            else math.nan,
        }
        for block, count in enumerate(counts)
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Suite Metrics",
        "",
        "| Config | Block | Success | Mean runtime (s) | p90 runtime (s) | Mean Q | Median Q |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {block} | {success_rate:.3f} | {mean_runtime_sec:.3f} | {p90_runtime_sec:.3f} | "
            "{mean_quality:.3f} | {median_quality:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def plot(
    metrics: dict[str, list[dict[str, Any]]],
    split: list[dict[str, Any]] | None,
    split_config: str | None,
    probes: dict[str, list[dict[str, Any]]],
    out: Path,
    raw_rows: dict[str, list[tuple[int, dict[str, Any]]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(11, 13))
    axes = axes.ravel()
    for color, (config, rows) in zip(COLORS, metrics.items()):
        blocks = [row["block"] + 1 for row in rows]
        label = LABELS.get(config, config)
        axes[0].plot(blocks, [row["success_rate"] for row in rows], "o-", color=color, label=label)
        axes[1].plot(blocks, [row["mean_runtime_sec"] for row in rows], "o-", color=color, label=f"{label} mean")
        axes[2].plot(blocks, [row["mean_quality"] for row in rows], "o-", color=color, label=label)

    axes[0].set(title="Success rate", ylabel="Rate", ylim=(0.0, 1.05))
    axes[1].set(title="End-to-end planner runtime", ylabel="Seconds")
    axes[2].set(title="Successful-trajectory quality", xlabel="Block", ylabel="Q", ylim=(0.0, 1.05))
    for axis in axes[:3]:
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, fontsize=8)

    if split:
        blocks = np.asarray([row["block"] + 1 for row in split], dtype=float)
        s1 = np.asarray([row["s1_success"] for row in split], dtype=float)
        s2 = np.asarray([row["s2_only_success"] for row in split], dtype=float)
        axes[3].bar(blocks - 0.2, s1, width=0.4, label="S1 success", color="#1f77b4")
        axes[3].bar(blocks + 0.2, s2, width=0.4, label="S2-only success", color="#ff7f0e")
        fraction_axis = axes[3].twinx()
        fraction_axis.plot(blocks, [row["s1_fraction_of_success"] for row in split], "o-", color="#2ca02c", label="S1 fraction")
        axes[3].set(title=f"S1/S2 split: {LABELS.get(split_config or '', split_config or '')}", xlabel="Block", ylabel="Successful scenarios")
        fraction_axis.set(ylabel="S1 fraction", ylim=(0.0, 1.05))
        handles, labels = axes[3].get_legend_handles_labels()
        handles2, labels2 = fraction_axis.get_legend_handles_labels()
        axes[3].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)
        axes[3].grid(axis="y", alpha=0.3)
    else:
        axes[3].set_axis_off()

    runtime_values = {
        config: [
            runtime
            for _, row in raw_rows.get(config, [])
            if (runtime := value(row, "planning_runtime_sec", "selected_runtime_sec", "runtime_sec", "wall_runtime_sec")) is not None
        ]
        for config in metrics
    }
    if any(runtime_values.values()):
        low = min(min(values) for values in runtime_values.values() if values)
        high = max(max(values) for values in runtime_values.values() if values)
        bins = np.linspace(low, high, 13) if high > low else 12
        for color, (config, values) in zip(COLORS, runtime_values.items()):
            if values:
                axes[4].hist(values, bins=bins, histtype="step", linewidth=2.2, color=color, label=LABELS.get(config, config))
        axes[4].set(title="Planning-runtime distribution", xlabel="Seconds", ylabel="Scenarios")
        axes[4].grid(alpha=0.3)
        axes[4].legend(frameon=False, fontsize=8)
    else:
        axes[4].set_axis_off()

    quality_values = {
        config: [
            score
            for _, row in raw_rows.get(config, [])
            if bool(row.get("success")) and (score := quality(row)) is not None
        ]
        for config in metrics
    }
    if any(quality_values.values()):
        low = min(min(values) for values in quality_values.values() if values)
        high = max(max(values) for values in quality_values.values() if values)
        bins = np.linspace(low, high, 13) if high > low else 12
        for color, (config, values) in zip(COLORS, quality_values.items()):
            if values:
                axes[5].hist(values, bins=bins, histtype="step", linewidth=2.2, color=color, label=LABELS.get(config, config))
        axes[5].set(title="Successful-trajectory quality distribution", xlabel="Quality score Q", ylabel="Scenarios")
        axes[5].grid(alpha=0.3)
        axes[5].legend(frameon=False, fontsize=8)
    else:
        axes[5].set_axis_off()

    if probes:
        for color, (config, rows) in zip(COLORS, probes.items()):
            blocks = [row["block"] + 1 for row in rows]
            label = LABELS.get(config, config)
            axes[6].plot(blocks, [row["success_rate"] for row in rows], "o-", color=color, label=label)
            axes[7].plot(blocks, [row["mean_quality"] for row in rows], "o-", color=color, label=label)
        axes[6].set(title="Fixed-probe S1 success", xlabel="CL updates completed", ylabel="Rate", ylim=(0.0, 1.05))
        axes[7].set(title="Fixed-probe successful-trajectory quality", xlabel="CL updates completed", ylabel="Q", ylim=(0.0, 1.05))
        for axis in axes[6:8]:
            axis.grid(alpha=0.3)
            axis.legend(frameon=False, fontsize=8)
    else:
        for axis in axes[6:8]:
            axis.set_axis_off()

    figure.tight_layout()
    figure.savefig(out, dpi=200)
    print(f"[write] {out}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, os.environ.get("MPLCONFIGDIR")))
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    manifest_path = suite_dir / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    configs = args.configs or list(manifest.get("configs", {}))
    if not configs:
        configs = sorted(path.name for path in suite_dir.iterdir() if path.is_dir() and list(path.rglob("*_runs.jsonl")))
    n_blocks = len(manifest.get("blocks", []))
    block_size = max(0, int(args.block_size))
    if not n_blocks and block_size == 0:
        block_size = inferred_block_size(suite_dir, configs)
        if block_size:
            print(f"[info] inferred block_size={block_size} from explicit block JSONLs")
    if not n_blocks:
        n_blocks = inferred_block_count(suite_dir, configs)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else suite_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    metrics: dict[str, list[dict[str, Any]]] = {}
    raw_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    probe_metrics: dict[str, list[dict[str, Any]]] = {}
    for config in configs:
        try:
            raw_rows[config] = rows_by_block(suite_dir, manifest, config, block_size)
        except (KeyError, FileNotFoundError) as error:
            print(f"[warn] {error}; skipping {config}")
            continue
        metrics[config] = aggregate(raw_rows[config], n_blocks, args.runtime_field)
        all_rows.extend({"family": family_name(suite_dir), "config": config, **row} for row in metrics[config])
        try:
            probe_rows = probe_rows_by_block(suite_dir, manifest, config)
        except FileNotFoundError as error:
            print(f"[warn] {error}; skipping probe plot for {config}")
            continue
        if probe_rows:
            probe_metrics[config] = aggregate_probe(probe_rows, n_blocks, args.runtime_field)

    if not metrics:
        raise SystemExit("No result JSONLs were found.")
    if not n_blocks:
        n_blocks = 1 + max(block for rows in raw_rows.values() for block, _ in rows)
        metrics = {config: aggregate(rows, n_blocks, args.runtime_field) for config, rows in raw_rows.items()}
        all_rows = [
            {"family": family_name(suite_dir), "config": config, **row}
            for config, rows in metrics.items()
            for row in rows
        ]
    if not probe_metrics:
        probe_metrics = probe_metrics_from_csv(suite_dir, metrics)
        if probe_metrics:
            print("[info] using saved probe_metrics_by_block.csv; raw probe JSONLs were not found.")
    split_config = args.split_config or next((config for config in metrics if config.endswith("_cl")), "")
    split = s1_s2_split(raw_rows[split_config], n_blocks) if split_config else None

    write_csv(out_dir / "metrics_by_block.csv", all_rows)
    if split:
        write_csv(out_dir / "s1_s2_split_by_block.csv", split)
    probe_output = [
        {"family": family_name(suite_dir), "config": config, **row}
        for config, rows in probe_metrics.items()
        for row in rows
        if row["scenarios"]
    ]
    if probe_output:
        write_csv(out_dir / "probe_metrics_by_block.csv", probe_output)
    else:
        print("[info] No fixed-probe JSONLs found in this suite manifest.")
    write_markdown(out_dir / "metrics_by_block.md", all_rows)
    plot(metrics, split, split_config, probe_metrics, out_dir / "suite_metrics.png", raw_rows)


if __name__ == "__main__":
    main()
