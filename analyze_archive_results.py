#!/usr/bin/env python3
"""Summarize and plot one current ``run_suite.py`` result directory.

Example:
  python analyze_archive_results.py \
    --suite_dir output/benchmark_runs/nl_bugtrap_suite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from solvers._s2_common import resolve_mplconfigdir


LABELS = {
    "s1_neural": "S1 neural",
    "s2_cbf": "S2 CBF",
    "s2_mpc": "S2 MPC",
    "sofai_cbf_cl": "SOFAI CBF CL",
    "sofai_mpc_cl": "SOFAI MPC CL",
}
COLORS = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite_dir", default="output/benchmark_runs/nl_bugtrap_suite")
    parser.add_argument("--configs", nargs="+", default=[], help="Defaults to every config in the manifest.")
    parser.add_argument("--split_config", default="", help="Config for the S1/S2 success split panel.")
    parser.add_argument("--runtime_field", default="selected_runtime_sec")
    parser.add_argument("--out_dir", default="", help="Defaults to <suite_dir>/analysis.")
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


def block_lookup(manifest: dict[str, Any]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for block, scenario_ids in enumerate(manifest.get("blocks", [])):
        for scenario_id in scenario_ids:
            lookup[int(scenario_id)] = block
    return lookup


def rows_by_block(suite_dir: Path, manifest: dict[str, Any], config: str) -> list[tuple[int, dict[str, Any]]]:
    config_data = manifest.get("configs", {}).get(config)
    if not config_data:
        raise KeyError(f"Config is absent from manifest: {config}")

    runs = config_data.get("runs", [])
    lookup = block_lookup(manifest)
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
        runtimes = [runtime for row in bucket if (runtime := value(row, runtime_field, "runtime_sec", "wall_runtime_sec")) is not None]
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


def plot(metrics: dict[str, list[dict[str, Any]]], split: list[dict[str, Any]], split_config: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.ravel()
    for color, (config, rows) in zip(COLORS, metrics.items()):
        blocks = [row["block"] + 1 for row in rows]
        label = LABELS.get(config, config)
        axes[0].plot(blocks, [row["success_rate"] for row in rows], "o-", color=color, label=label)
        axes[1].plot(blocks, [row["mean_runtime_sec"] for row in rows], "o-", color=color, label=f"{label} mean")
        axes[1].plot(blocks, [row["p90_runtime_sec"] for row in rows], "--", color=color, alpha=0.65, label=f"{label} p90")
        axes[2].plot(blocks, [row["mean_quality"] for row in rows], "o-", color=color, label=label)

    axes[0].set(title="Success rate", ylabel="Rate", ylim=(0.0, 1.05))
    axes[1].set(title="Selected solver runtime", ylabel="Seconds")
    axes[2].set(title="Successful-trajectory quality", xlabel="Block", ylabel="Q", ylim=(0.0, 1.05))
    for axis in axes[:3]:
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, fontsize=8)

    blocks = np.asarray([row["block"] + 1 for row in split], dtype=float)
    s1 = np.asarray([row["s1_success"] for row in split], dtype=float)
    s2 = np.asarray([row["s2_only_success"] for row in split], dtype=float)
    axes[3].bar(blocks - 0.2, s1, width=0.4, label="S1 success", color="#1f77b4")
    axes[3].bar(blocks + 0.2, s2, width=0.4, label="S2-only success", color="#ff7f0e")
    fraction_axis = axes[3].twinx()
    fraction_axis.plot(blocks, [row["s1_fraction_of_success"] for row in split], "o-", color="#2ca02c", label="S1 fraction")
    axes[3].set(title=f"S1/S2 split: {LABELS.get(split_config, split_config)}", xlabel="Block", ylabel="Successful scenarios")
    fraction_axis.set(ylabel="S1 fraction", ylim=(0.0, 1.05))
    handles, labels = axes[3].get_legend_handles_labels()
    handles2, labels2 = fraction_axis.get_legend_handles_labels()
    axes[3].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)
    axes[3].grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(out, dpi=200)
    print(f"[write] {out}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, os.environ.get("MPLCONFIGDIR")))
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    manifest = json.loads((suite_dir / "suite_manifest.json").read_text())
    configs = args.configs or list(manifest.get("configs", {}))
    n_blocks = len(manifest.get("blocks", []))
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else suite_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    metrics: dict[str, list[dict[str, Any]]] = {}
    raw_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for config in configs:
        try:
            raw_rows[config] = rows_by_block(suite_dir, manifest, config)
        except (KeyError, FileNotFoundError) as error:
            print(f"[warn] {error}; skipping {config}")
            continue
        metrics[config] = aggregate(raw_rows[config], n_blocks, args.runtime_field)
        all_rows.extend({"family": family_name(suite_dir), "config": config, **row} for row in metrics[config])

    if not metrics:
        raise SystemExit("No result JSONLs were found.")
    split_config = args.split_config or next((config for config in metrics if config.endswith("_cl")), next(iter(metrics)))
    split = s1_s2_split(raw_rows[split_config], n_blocks)

    write_csv(out_dir / "metrics_by_block.csv", all_rows)
    write_csv(out_dir / "s1_s2_split_by_block.csv", split)
    write_markdown(out_dir / "metrics_by_block.md", all_rows)
    plot(metrics, split, split_config, out_dir / "suite_metrics.png")


if __name__ == "__main__":
    main()
