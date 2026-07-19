#!/usr/bin/env python3
"""Analyze archived multi-family run_suite.py outputs.

Example:
    python analyze_archive_results.py \
      --archive_dir output/Archive \
      --configs s1_neural s2_cbf sofai_cbf_cl \
      --out_dir output/Archive/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from plot_suite_results import (
    normalize_config,
    quality_value,
    read_jsonl,
    read_manifest,
    runtime_value,
)
from solvers._s2_common import resolve_mplconfigdir


DEFAULT_CONFIGS = ["s1_neural", "s2_cbf", "sofai_cbf_cl"]
FAMILY_ORDER = [
    "small_open",
    "large_sparse",
    "dense_clutter",
    "wall_gap",
    "serial_walls",
    "maze_branching",
    "bugtrap",
]
LABELS = {
    "s1_neural": "S1 neural",
    "s2_cbf": "S2 CBF",
    "s2_mpc": "S2 MPC",
    "sofai_cbf_cl": "SOFAI CBF CL",
    "sofai_mpc_cl": "SOFAI MPC CL",
}


RowWithBlock = tuple[int, Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive_dir", default="output/Archive")
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    p.add_argument("--out_dir", default="output/Archive/analysis")
    p.add_argument("--runtime_field", default="selected_runtime_sec")
    p.add_argument("--ratio_config", default="sofai_cbf_cl", help="SOFAI CL config for S1/S2 success split plots.")
    return p.parse_args()


def suite_family(suite_dir: Path) -> str:
    name = suite_dir.name
    if name.startswith("nl_") and name.endswith("_suite"):
        return name[3:-6]
    return name


def suite_dirs(archive_dir: Path) -> List[Path]:
    dirs = [p for p in archive_dir.glob("nl_*_suite") if (p / "suite_manifest.json").is_file()]
    return sorted(dirs, key=lambda p: FAMILY_ORDER.index(suite_family(p)) if suite_family(p) in FAMILY_ORDER else 999)


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    return float(np.mean(arr)) if arr.size else math.nan


def finite_p90(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    return float(np.percentile(arr, 90)) if arr.size else math.nan


def resolve_archived_jsonl(suite_dir: Path, cfg: str, run: Dict[str, Any]) -> Path:
    prefix = str(run.get("prefix", "")).strip()
    if prefix:
        local = suite_dir / cfg / "runs" / f"{prefix}_runs.jsonl"
        if local.is_file():
            return local

    raw = str(run.get("jsonl", "")).strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        parts = path.parts
        if suite_dir.name in parts:
            idx = parts.index(suite_dir.name)
            local = suite_dir.joinpath(*parts[idx + 1 :])
            if local.is_file():
                return local

    raise FileNotFoundError(f"could not resolve archived JSONL for {suite_dir.name}/{cfg}/{prefix or raw}")


def resolve_archived_probe_jsonl(suite_dir: Path, cfg: str, run: Dict[str, Any]) -> Path:
    prefix = str(run.get("probe_prefix", "")).strip()
    if prefix:
        local = suite_dir / cfg / "probe" / f"{prefix}_runs.jsonl"
        if local.is_file():
            return local

    raw = str(run.get("probe_jsonl", "")).strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        parts = path.parts
        if suite_dir.name in parts:
            idx = parts.index(suite_dir.name)
            local = suite_dir.joinpath(*parts[idx + 1 :])
            if local.is_file():
                return local

    run_prefix = str(run.get("prefix", "")).strip()
    raise FileNotFoundError(f"could not resolve probe JSONL for {suite_dir.name}/{cfg}/{run_prefix or prefix or raw}")


def run_block_index(run: Dict[str, Any], default: int) -> int:
    prefix = str(run.get("prefix", ""))
    marker = "_block"
    if marker in prefix:
        tail = prefix.rsplit(marker, 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits)
    return default


def block_lookup_from_manifest(manifest: Dict[str, Any]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    blocks = manifest.get("blocks", [])
    if not isinstance(blocks, list):
        return lookup
    for block_idx, ids in enumerate(blocks):
        if not isinstance(ids, list):
            continue
        for sid in ids:
            try:
                lookup[int(sid)] = block_idx
            except (TypeError, ValueError):
                continue
    return lookup


def collect_archived_rows(suite_dir: Path, manifest: Dict[str, Any], cfg: str, block_size: int) -> List[RowWithBlock]:
    cfg_manifest = manifest.get("configs", {}).get(cfg)
    if not cfg_manifest:
        raise KeyError(f"Config {cfg!r} not found in manifest.")

    if cfg in {"s1_neural", "s2_cbf", "s2_mpc"}:
        rows = read_jsonl(suite_dir / cfg / f"{cfg}_runs.jsonl")
        block_lookup = block_lookup_from_manifest(manifest)
        out: List[RowWithBlock] = []
        for i, row in enumerate(rows):
            try:
                sid = int(row.get("scenario_index", row.get("scenario_id", -1)))
            except (TypeError, ValueError):
                sid = -1
            out.append((block_lookup.get(sid, i // block_size), row))
        return out

    rows: List[RowWithBlock] = []
    for i, run in enumerate(cfg_manifest.get("runs", [])):
        block_idx = run_block_index(run, i)
        rows.extend((block_idx, row) for row in read_jsonl(resolve_archived_jsonl(suite_dir, cfg, run)))
    return rows


def aggregate_blocks(rows: Sequence[RowWithBlock], n_blocks: int, runtime_field: str) -> Dict[str, List[float]]:
    success_buckets: List[List[bool]] = [[] for _ in range(n_blocks)]
    runtime_buckets: List[List[float]] = [[] for _ in range(n_blocks)]
    quality_buckets: List[List[float]] = [[] for _ in range(n_blocks)]

    for block_idx, row in rows:
        if not (0 <= block_idx < n_blocks):
            continue
        success_buckets[block_idx].append(bool(row.get("success", False)))

        rt = runtime_value(row, runtime_field)
        if rt is not None:
            runtime_buckets[block_idx].append(rt)

        if bool(row.get("success", False)):
            q = quality_value(row)
            if q is not None:
                quality_buckets[block_idx].append(q)

    def mean_bool(bucket: Sequence[bool]) -> float:
        return sum(bucket) / len(bucket) if bucket else math.nan

    def mean_float(bucket: Sequence[float]) -> float:
        return float(np.mean(bucket)) if bucket else math.nan

    def median_float(bucket: Sequence[float]) -> float:
        return float(np.median(bucket)) if bucket else math.nan

    def p90_float(bucket: Sequence[float]) -> float:
        return float(np.percentile(bucket, 90)) if bucket else math.nan

    return {
        "success": [mean_bool(bucket) for bucket in success_buckets],
        "runtime_mean": [mean_float(bucket) for bucket in runtime_buckets],
        "runtime_p90": [p90_float(bucket) for bucket in runtime_buckets],
        "quality_mean": [mean_float(bucket) for bucket in quality_buckets],
        "quality_median": [median_float(bucket) for bucket in quality_buckets],
        "quality_p90": [p90_float(bucket) for bucket in quality_buckets],
    }


def collect_archive(archive_dir: Path, configs: Sequence[str], runtime_field: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for suite_dir in suite_dirs(archive_dir):
        family = suite_family(suite_dir)
        manifest = read_manifest(suite_dir / "suite_manifest.json")
        block_size = int(manifest["block_size"])
        n_blocks = math.ceil(int(manifest["scenario_count"]) / block_size)

        for cfg_raw in configs:
            cfg = normalize_config(cfg_raw)
            try:
                rows = collect_archived_rows(suite_dir, manifest, cfg, block_size)
            except (KeyError, FileNotFoundError) as exc:
                print(f"[warn] {family}/{cfg}: {exc}; skipping")
                continue
            if not rows:
                print(f"[warn] {family}/{cfg}: no rows; skipping")
                continue

            series = aggregate_blocks(rows, n_blocks, runtime_field)
            success = series["success"]
            runtime_mean = series["runtime_mean"]
            runtime_p90 = series["runtime_p90"]
            quality_mean = series["quality_mean"]
            quality_median = series["quality_median"]
            quality_p90 = series["quality_p90"]

            for i in range(n_blocks):
                block_rows.append(
                    {
                        "family": family,
                        "solver": cfg,
                        "block": i,
                        "block_start": i * block_size,
                        "block_end": min((i + 1) * block_size - 1, int(manifest["scenario_count"]) - 1),
                        "success_rate": success[i],
                        "mean_runtime_sec": runtime_mean[i],
                        "p90_runtime_sec": runtime_p90[i],
                        "mean_quality": quality_mean[i],
                        "median_quality": quality_median[i],
                        "p90_quality": quality_p90[i],
                    }
                )

            summary_rows.append(
                {
                    "family": family,
                    "solver": cfg,
                    "success_rate": finite_mean(success),
                    "mean_runtime_sec": finite_mean(runtime_mean),
                    "p90_runtime_sec": finite_mean(runtime_p90),
                    "mean_quality": finite_mean(quality_mean),
                    "median_quality": finite_mean(quality_median),
                    "p90_quality": finite_mean(quality_p90),
                }
            )

    return block_rows, summary_rows


def attempt_success_split(row: Dict[str, Any]) -> tuple[bool, bool]:
    attempts = row.get("attempts", []) or []
    s1_success = any(str(attempt.get("system")) == "s1" and bool(attempt.get("success")) for attempt in attempts)
    s2_success = any(str(attempt.get("system")) == "s2" and bool(attempt.get("success")) for attempt in attempts)
    return s1_success, s2_success


def collect_s1_s2_ratio_rows(archive_dir: Path, config: str) -> list[dict[str, Any]]:
    ratio_rows: list[dict[str, Any]] = []
    cfg = normalize_config(config)

    for suite_dir in suite_dirs(archive_dir):
        family = suite_family(suite_dir)
        manifest = read_manifest(suite_dir / "suite_manifest.json")
        block_size = int(manifest["block_size"])
        n_blocks = math.ceil(int(manifest["scenario_count"]) / block_size)
        try:
            rows = collect_archived_rows(suite_dir, manifest, cfg, block_size)
        except (KeyError, FileNotFoundError) as exc:
            print(f"[warn] {family}/{cfg}: {exc}; skipping S1/S2 split")
            continue

        s1_counts = [0] * n_blocks
        s2_only_counts = [0] * n_blocks
        failed_counts = [0] * n_blocks
        for block_idx, row in rows:
            if not (0 <= block_idx < n_blocks):
                continue
            s1_success, s2_success = attempt_success_split(row)
            if s1_success:
                s1_counts[block_idx] += 1
            elif s2_success:
                s2_only_counts[block_idx] += 1
            else:
                failed_counts[block_idx] += 1

        for block_idx in range(n_blocks):
            solved = s1_counts[block_idx] + s2_only_counts[block_idx]
            total = solved + failed_counts[block_idx]
            ratio_rows.append(
                {
                    "family": family,
                    "solver": cfg,
                    "block": block_idx,
                    "block_start": block_idx * block_size,
                    "block_end": min((block_idx + 1) * block_size - 1, int(manifest["scenario_count"]) - 1),
                    "s1_success": s1_counts[block_idx],
                    "s2_only_success": s2_only_counts[block_idx],
                    "failed": failed_counts[block_idx],
                    "successful_total": solved,
                    "scenario_total": total,
                    "s1_fraction_of_success": s1_counts[block_idx] / solved if solved else math.nan,
                    "s2_only_fraction_of_success": s2_only_counts[block_idx] / solved if solved else math.nan,
                    "success_rate": solved / total if total else math.nan,
                }
            )

    return ratio_rows


def summarize_result_rows(rows: Sequence[dict[str, Any]], runtime_field: str) -> dict[str, Any]:
    successes = [bool(row.get("success", False)) for row in rows]
    runtimes = [runtime_value(row, runtime_field) for row in rows]
    runtimes = [float(v) for v in runtimes if v is not None]
    qualities = []
    for row in rows:
        if bool(row.get("success", False)):
            q = quality_value(row)
            if q is not None:
                qualities.append(float(q))

    return {
        "scenario_total": len(rows),
        "success_count": sum(successes),
        "success_rate": sum(successes) / len(successes) if successes else math.nan,
        "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else math.nan,
        "p90_runtime_sec": float(np.percentile(runtimes, 90)) if runtimes else math.nan,
        "mean_quality": float(np.mean(qualities)) if qualities else math.nan,
        "median_quality": float(np.median(qualities)) if qualities else math.nan,
        "p90_quality": float(np.percentile(qualities, 90)) if qualities else math.nan,
    }


def collect_probe_rows(archive_dir: Path, configs: Sequence[str], runtime_field: str) -> list[dict[str, Any]]:
    probe_rows: list[dict[str, Any]] = []
    for suite_dir in suite_dirs(archive_dir):
        family = suite_family(suite_dir)
        manifest = read_manifest(suite_dir / "suite_manifest.json")
        for cfg_raw in configs:
            cfg = normalize_config(cfg_raw)
            if not cfg.endswith("_cl"):
                continue
            cfg_manifest = manifest.get("configs", {}).get(cfg)
            if not cfg_manifest:
                continue
            for i, run in enumerate(cfg_manifest.get("runs", [])):
                if not (run.get("probe_jsonl") or run.get("probe_prefix")):
                    continue
                try:
                    rows = read_jsonl(resolve_archived_probe_jsonl(suite_dir, cfg, run))
                except FileNotFoundError as exc:
                    print(f"[warn] {family}/{cfg}: {exc}; skipping probe")
                    continue
                summary = summarize_result_rows(rows, runtime_field)
                probe_rows.append(
                    {
                        "family": family,
                        "solver": cfg,
                        "block": run_block_index(run, i),
                        "model": str(run.get("model", "")),
                        **summary,
                    }
                )
    return probe_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[write] {path}")


def write_markdown(path: Path, summary_rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Archived Benchmark Summary",
        "",
        "| Family | Solver | Success | Mean RT (s) | p90 RT (s) | Mean Q | Median Q | p90 Q |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {family} | {solver} | {success_rate:.3f} | {mean_runtime_sec:.3f} | "
            "{p90_runtime_sec:.3f} | {mean_quality:.3f} | {median_quality:.3f} | {p90_quality:.3f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def write_ratio_markdown(path: Path, ratio_rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# SOFAI CBF Continual-Learning S1/S2 Split",
        "",
        "| Family | Block | S1 success | S2-only success | Failed | S1 fraction of successes | Success rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ratio_rows:
        lines.append(
            "| {family} | {block} | {s1_success} | {s2_only_success} | {failed} | "
            "{s1_fraction_of_success:.3f} | {success_rate:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def write_probe_markdown(path: Path, probe_rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Fixed-Probe S1-Only Continual-Learning Curve",
        "",
        "| Family | Solver | After block | Success | Mean RT (s) | p90 RT (s) | Mean Q | Median Q | p90 Q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in probe_rows:
        lines.append(
            "| {family} | {solver} | {block} | {success_rate:.3f} | {mean_runtime_sec:.3f} | "
            "{p90_runtime_sec:.3f} | {mean_quality:.3f} | {median_quality:.3f} | {p90_quality:.3f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def plot_metric(block_rows: Sequence[dict[str, Any]], out_dir: Path, metric: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in FAMILY_ORDER if any(row["family"] == f for row in block_rows)]
    solvers = sorted({str(row["solver"]) for row in block_rows})
    if not families or not solvers:
        return

    fig, axes = plt.subplots(len(families), 1, figsize=(9.5, 2.35 * len(families)), sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        family_rows = [row for row in block_rows if row["family"] == family]
        for solver in solvers:
            rows = sorted([row for row in family_rows if row["solver"] == solver], key=lambda r: int(r["block"]))
            if not rows:
                continue
            xs = [int(row["block"]) + 1 for row in rows]
            ys = [float(row[metric]) if row[metric] not in (None, "") else math.nan for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=LABELS.get(solver, solver))
        ax.set_title(family)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if metric in {"success_rate", "mean_quality", "median_quality", "p90_quality"}:
            ax.set_ylim(0.0, 1.05)

    axes[-1].set_xlabel("Block")
    axes[0].legend(frameon=False, ncols=min(len(solvers), 3), loc="best")
    fig.tight_layout()
    out = out_dir / f"{metric}_by_family_block.png"
    fig.savefig(out, dpi=200)
    print(f"[write] {out}")


def plot_s1_s2_ratio(ratio_rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in FAMILY_ORDER if any(row["family"] == f for row in ratio_rows)]
    if not families:
        return

    fig, axes = plt.subplots(len(families), 1, figsize=(10.0, 2.55 * len(families)), sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        rows = sorted([row for row in ratio_rows if row["family"] == family], key=lambda r: int(r["block"]))
        xs = np.asarray([int(row["block"]) + 1 for row in rows], dtype=float)
        s1 = np.asarray([int(row["s1_success"]) for row in rows], dtype=float)
        s2 = np.asarray([int(row["s2_only_success"]) for row in rows], dtype=float)
        frac = np.asarray([float(row["s1_fraction_of_success"]) for row in rows], dtype=float)

        width = 0.38
        ax.bar(xs - width / 2.0, s1, width=width, color="#1f77b4", label="S1 success")
        ax.bar(xs + width / 2.0, s2, width=width, color="#ff7f0e", label="S2-only success")
        ax.set_title(family)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)

        ax_frac = ax.twinx()
        ax_frac.plot(xs, frac, color="#2ca02c", marker="o", linewidth=2.0, label="S1 fraction")
        ax_frac.set_ylim(0.0, 1.05)
        ax_frac.set_ylabel("S1 fraction")

        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax_frac.get_legend_handles_labels()
            ax.legend(handles + handles2, labels + labels2, frameon=False, ncols=3, loc="upper center")

    axes[-1].set_xlabel("Block")
    axes[-1].set_xticks(sorted({int(row["block"]) + 1 for row in ratio_rows}))
    fig.tight_layout()
    out = out_dir / "sofai_cbf_cl_s1_s2_success_split_by_family_block.png"
    fig.savefig(out, dpi=200)
    print(f"[write] {out}")


def plot_probe_success(probe_rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in FAMILY_ORDER if any(row["family"] == f for row in probe_rows)]
    solvers = sorted({str(row["solver"]) for row in probe_rows})
    if not families or not solvers:
        return

    fig, axes = plt.subplots(len(families), 1, figsize=(9.5, 2.35 * len(families)), sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        family_rows = [row for row in probe_rows if row["family"] == family]
        for solver in solvers:
            rows = sorted([row for row in family_rows if row["solver"] == solver], key=lambda r: int(r["block"]))
            if not rows:
                continue
            xs = [int(row["block"]) + 1 for row in rows]
            ys = [float(row["success_rate"]) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=LABELS.get(solver, solver))
        ax.set_title(family)
        ax.set_ylabel("S1 probe success")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("After CL block")
    axes[0].legend(frameon=False, ncols=min(len(solvers), 3), loc="best")
    fig.tight_layout()
    out = out_dir / "fixed_probe_s1_success_by_family_block.png"
    fig.savefig(out, dpi=200)
    print(f"[write] {out}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(repo_root, os.environ.get("MPLCONFIGDIR")))

    archive_dir = Path(args.archive_dir).expanduser()
    if not archive_dir.is_absolute():
        archive_dir = repo_root / archive_dir
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    block_rows, summary_rows = collect_archive(archive_dir, args.configs, args.runtime_field)
    write_csv(out_dir / "metrics_by_block.csv", block_rows)
    write_csv(out_dir / "metrics_summary.csv", summary_rows)
    write_markdown(out_dir / "metrics_summary.md", summary_rows)

    ratio_rows = collect_s1_s2_ratio_rows(archive_dir, args.ratio_config)
    write_csv(out_dir / f"{normalize_config(args.ratio_config)}_s1_s2_split_by_block.csv", ratio_rows)
    write_ratio_markdown(out_dir / f"{normalize_config(args.ratio_config)}_s1_s2_split_by_block.md", ratio_rows)
    probe_rows = collect_probe_rows(archive_dir, args.configs, args.runtime_field)
    write_csv(out_dir / "fixed_probe_s1_by_block.csv", probe_rows)
    write_probe_markdown(out_dir / "fixed_probe_s1_by_block.md", probe_rows)

    plot_metric(block_rows, out_dir, "success_rate", "Success rate")
    plot_metric(block_rows, out_dir, "mean_runtime_sec", "Mean runtime (s)")
    plot_metric(block_rows, out_dir, "p90_runtime_sec", "p90 runtime (s)")
    plot_metric(block_rows, out_dir, "mean_quality", "Mean quality Q")
    plot_metric(block_rows, out_dir, "p90_quality", "p90 quality Q")
    plot_s1_s2_ratio(ratio_rows, out_dir)
    plot_probe_success(probe_rows, out_dir)


if __name__ == "__main__":
    main()
