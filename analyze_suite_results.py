#!/usr/bin/env python3
"""Analyze a complete ``run_suite.py`` archive and write all artefacts below ``analysis/``.

Example::

    python analyze_suite_results.py --archive_dir output/benchmark_runs \
      --families dense_clutter large_sparse maze_branching serial_walls long_slalom bugtrap \
      --configs s1_neural s2_cbf sofai_cbf_cl s2_mpc sofai_mpc_cl sofai_mpc_warm_cl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from solvers._s2_common import resolve_mplconfigdir

METHOD_LABELS = {
    "s1_neural": "NN", "s2_mpc": "MPC", "s2_cbf": "CBF",
    "sofai_cbf_cl": "sofai CBF CL", "sofai_mpc_cl": "sofai MPC CL",
    "sofai_mpc_warm_cl": "sofai MPC warm CL",
}
METHOD_ORDER = tuple(METHOD_LABELS)
BOOLEAN_COLUMNS = frozenset({
    "timed_out", "success", "collision_free", "goal_reached", "s1_attempted",
    "s1_success", "s1_collision_free", "s1_goal_reached", "s2_attempted",
    "s2_success", "s2_collision_free", "s2_goal_reached", "s2_skipped",
})
FAMILIES = (
    ("large_sparse", "Large sparse"), ("dense_clutter", "Dense clutter"),
    ("serial_walls", "Serial walls"), ("maze_branching", "Maze branching"),
    ("long_slalom", "Long slalom"), ("bugtrap", "Bugtrap"),
)
ARMS = {
    "sofai_cbf_cl": ("CBF teacher", "#1f77b4", "o"),
    "sofai_mpc_cl": ("MPC teacher", "#d62728", "s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_dir", default="output/benchmark_runs_v1")
    parser.add_argument("--families", nargs="+", default=[])
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--runtime_field", default="planning_runtime_sec")
    parser.add_argument("--block_size", type=int, default=0)
    parser.add_argument("--out_dir", default="", help="Defaults to <archive_dir>/analysis.")
    parser.add_argument("--no_learning_plots", action="store_true")
    parser.add_argument("--skip_paper_figures", action="store_true")
    parser.add_argument("--skip_s1_s2_plots", action="store_true")
    return parser.parse_args()


def family_name(path: Path) -> str:
    name = path.name
    return name[3:-6] if name.startswith("nl_") and name.endswith("_suite") else name


def parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {key: value for key, value in raw.items() if key is not None}
            for key in BOOLEAN_COLUMNS.intersection(row):
                row[key] = parse_bool(row[key])
            for key in ("scenario_index", "scenario_id"):
                try:
                    row[key] = int(float(row[key]))
                except (KeyError, TypeError, ValueError):
                    row.pop(key, None)
            row["attempts"] = [
                {"system": system, "success": bool(row.get(f"{system}_success"))}
                for system in ("s1", "s2") if row.get(f"{system}_attempted")
            ]
            rows.append(row)
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
    if str(row.get("quality_definition", "")) == "duration_invariant_v1":
        return value(row, "quality_score")
    if bool(row.get("success")):
        try:
            from solvers._s2_common import (
                benchmark_family_from_dictionary, quality_refs_for_result, quality_score,
                quality_weights_for_family, trajectory_quality_components,
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


def summary_files(suite_dir: Path, config: str, probe: bool) -> list[Path]:
    config_dir = suite_dir / config
    return [path for path in sorted(config_dir.rglob("*_summary.csv")) if ("probe" in path.parts) is probe] if config_dir.is_dir() else []


def file_block_index(path: Path) -> int | None:
    match = re.search(r"_block(\d+)", path.stem)
    return int(match.group(1)) if match else None


def block_lookup(manifest: dict[str, Any]) -> dict[int, int]:
    return {int(identifier): block for block, ids in enumerate(manifest.get("blocks", [])) for identifier in ids}


def fallback_block_lookup(suite_dir: Path, configs: Iterable[str]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for config in configs:
        for path in summary_files(suite_dir, config, False):
            block = file_block_index(path)
            if block is not None:
                for row in read_summary_csv(path):
                    index = row.get("scenario_index", row.get("scenario_id"))
                    if index is not None:
                        lookup.setdefault(int(index), block)
    return lookup


def resolve_run_file(suite_dir: Path, config: str, run: dict[str, Any], probe: bool = False) -> Path:
    prefix = str(
        (run.get("probe_prefix") or run.get("prefix", "")) if probe else run.get("prefix", "")
    ).strip()
    candidates = []
    if prefix:
        candidates.append(suite_dir / config / ("probe" if probe else "runs") / f"{prefix}_runs.jsonl")
        candidates.append(suite_dir / config / f"{prefix}_runs.jsonl")
    raw = str(
        (run.get("probe_jsonl") or run.get("jsonl") or "") if probe else run.get("jsonl") or ""
    ).strip()
    if raw:
        original = Path(raw).expanduser()
        candidates.append(original)
        if suite_dir.name in original.parts:
            candidates.append(suite_dir.joinpath(*original.parts[original.parts.index(suite_dir.name) + 1:]))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No {'probe ' if probe else ''}JSONL found for {config}/{prefix or raw}")


def resolve_probe_summary_file(suite_dir: Path, config: str, run: dict[str, Any]) -> Path:
    prefix = str(run.get("probe_prefix") or run.get("prefix", "")).strip()
    candidates = []
    if prefix:
        candidates.append(suite_dir / config / "probe" / f"{prefix}_summary.csv")
        candidates.append(suite_dir / config / f"{prefix}_summary.csv")
    raw = str(run.get("probe_summary_csv") or run.get("probe_jsonl") or run.get("jsonl") or "").strip()
    if raw:
        original = Path(raw).expanduser()
        candidates.append(original)
        candidates.append(original.with_name(re.sub(r"_runs\.jsonl$", "_summary.csv", original.name)))
        if suite_dir.name in original.parts:
            relative = original.parts[original.parts.index(suite_dir.name) + 1:]
            candidates.append(suite_dir.joinpath(*relative).with_name(re.sub(r"_runs\.jsonl$", "_summary.csv", original.name)))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No probe summary CSV found for {config}/{prefix or raw}")


def direct_rows(suite_dir: Path, config: str, block_size: int) -> list[tuple[int, dict[str, Any]]]:
    files = [path for path in sorted((suite_dir / config).rglob("*_runs.jsonl")) if "probe" not in path.parts]
    if not files:
        raise FileNotFoundError(f"No results found in {suite_dir / config}")
    rows: list[tuple[int | None, dict[str, Any]]] = []
    for path in files:
        rows.extend((file_block_index(path), row) for row in read_jsonl(path))
    if any(block is not None for block, _ in rows):
        return [(block or 0, row) for block, row in rows]
    ordered = sorted(rows, key=lambda item: int(item[1].get("scenario_index", item[1].get("scenario_id", -1))))
    return [(index // block_size if block_size else 0, row) for index, (_, row) in enumerate(ordered)]


def rows_by_block(suite_dir: Path, manifest: dict[str, Any], config: str, block_size: int, lookup: dict[int, int]) -> list[tuple[int, dict[str, Any]]]:
    config_data = manifest.get("configs", {}).get(config)
    if not config_data:
        rows = direct_rows(suite_dir, config, block_size)
        if config.endswith("_cl"):
            return rows
        return [(lookup.get(int(row.get("scenario_index", row.get("scenario_id", -1))), block), row) for block, row in rows]
    rows: list[tuple[int, dict[str, Any]]] = []
    for default, run in enumerate(config_data.get("runs", [])):
        block = file_block_index(Path(str(run.get("prefix", "")))) or default
        for row in read_jsonl(resolve_run_file(suite_dir, config, run)):
            index = int(row.get("scenario_index", row.get("scenario_id", -1)))
            rows.append((block if config.endswith("_cl") else lookup.get(index, block), row))
    return rows


def rows_from_summaries(suite_dir: Path, config: str, lookup: dict[int, int], block_size: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for path in summary_files(suite_dir, config, False):
        explicit = file_block_index(path)
        for row in read_summary_csv(path):
            index = row.get("scenario_index", row.get("scenario_id"))
            block = explicit if explicit is not None else lookup.get(int(index), int(index) // block_size if block_size and index is not None else 0)
            rows.append((block, row))
    if not rows:
        raise FileNotFoundError(f"No summary CSV found in {suite_dir / config}")
    return rows


def probe_rows(suite_dir: Path, manifest: dict[str, Any], config: str) -> list[tuple[int, dict[str, Any]]]:
    config_data = manifest.get("configs", {}).get(config, {})
    rows: list[tuple[int, dict[str, Any]]] = []
    base = config_data.get("base_probe", {})
    if base:
        try:
            rows.extend((-1, row) for row in read_jsonl(resolve_run_file(suite_dir, config, base, True)))
        except FileNotFoundError:
            try:
                rows.extend((-1, row) for row in read_summary_csv(resolve_probe_summary_file(suite_dir, config, base)))
            except FileNotFoundError:
                pass
    for default, run in enumerate(config_data.get("runs", [])):
        if run.get("probe_jsonl") or run.get("probe_prefix"):
            block = file_block_index(Path(str(run.get("prefix", "")))) or default
            try:
                rows.extend((block, row) for row in read_jsonl(resolve_run_file(suite_dir, config, run, True)))
            except FileNotFoundError:
                try:
                    rows.extend((block, row) for row in read_summary_csv(resolve_probe_summary_file(suite_dir, config, run)))
                except FileNotFoundError:
                    pass
    if rows:
        return rows
    for path in summary_files(suite_dir, config, True):
        block = -1 if "base_probe" in path.stem else file_block_index(path)
        if block is not None:
            rows.extend((block, row) for row in read_summary_csv(path))
    return rows


def aggregate(rows: Iterable[tuple[int, dict[str, Any]]], n_blocks: int, runtime_field: str) -> list[dict[str, Any]]:
    buckets = [[] for _ in range(n_blocks)]
    for block, row in rows:
        if 0 <= block < n_blocks:
            buckets[block].append(row)
    output = []
    for block, bucket in enumerate(buckets):
        runtime = [v for row in bucket if (v := value(row, runtime_field, "selected_runtime_sec", "runtime_sec", "wall_runtime_sec")) is not None]
        scores = [v for row in bucket if bool(row.get("success")) and (v := quality(row)) is not None]
        output.append({
            "block": block, "scenarios": len(bucket), "successes": sum(bool(row.get("success")) for row in bucket),
            "success_rate": sum(bool(row.get("success")) for row in bucket) / len(bucket) if bucket else math.nan,
            "mean_runtime_sec": float(np.mean(runtime)) if runtime else math.nan,
            "p90_runtime_sec": float(np.percentile(runtime, 90)) if runtime else math.nan,
            "mean_quality": float(np.mean(scores)) if scores else math.nan,
            "median_quality": float(np.median(scores)) if scores else math.nan,
            "p90_quality": float(np.percentile(scores, 90)) if scores else math.nan,
        })
    return output


def aggregate_probe(rows: list[tuple[int, dict[str, Any]]], n_blocks: int, runtime_field: str) -> list[dict[str, Any]]:
    baseline = aggregate([(0, row) for block, row in rows if block == -1], 1, runtime_field)[0]
    baseline["block"] = -1
    updates = aggregate([(block, row) for block, row in rows if block >= 0], n_blocks, runtime_field)
    return [baseline, *updates] if baseline["scenarios"] else updates


def s1_s2_split(rows: Iterable[tuple[int, dict[str, Any]]], n_blocks: int) -> list[dict[str, Any]]:
    counts = [{"s1": 0, "s2": 0, "failed": 0} for _ in range(n_blocks)]
    for block, row in rows:
        if 0 <= block < n_blocks:
            attempts = row.get("attempts", []) or []
            s1 = any(item.get("system") == "s1" and bool(item.get("success")) for item in attempts)
            s2 = any(item.get("system") == "s2" and bool(item.get("success")) for item in attempts)
            counts[block]["s1" if s1 else "s2" if s2 else "failed"] += 1
    return [{"block": block, "s1_success": count["s1"], "s2_only_success": count["s2"], "failed": count["failed"],
             "s1_fraction_of_success": count["s1"] / (count["s1"] + count["s2"]) if count["s1"] + count["s2"] else math.nan}
            for block, count in enumerate(counts)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def ordered_configs(rows: Iterable[dict[str, Any]]) -> list[str]:
    present = {str(row["config"]) for row in rows}
    return [config for config in METHOD_ORDER if config in present] + sorted(present.difference(METHOD_ORDER))


def number(value: Any, digits: int = 3, percent: bool = False) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not math.isfinite(value) else f"{100 * value:.1f}" if percent else f"{value:.{digits}f}"


def write_tables(out_dir: Path, family_rows: list[dict[str, Any]], solves: dict[tuple[str, str], tuple[float, float]], name: str, caption: str, label: str) -> None:
    ordered = sorted(family_rows, key=lambda row: ordered_configs(family_rows).index(str(row["config"])))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\resizebox{\linewidth}{!}{",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"\textbf{Method} & \textbf{Succ. (\%)} & \textbf{Mean RT (ms)} & \textbf{P90 RT (ms)} & \textbf{Mean \(Q\)} & \textbf{P90 \(Q\)} & \textbf{S1/S2 Solves} \\", r"\midrule"]
    cells = []
    for row in ordered:
        key = (str(row["family"]), str(row["config"]))
        split = solves.get(key)
        split_text = "-- / --" if split is None else f"{round(split[0])} / {round(split[1])}"
        values = [METHOD_LABELS.get(str(row["config"]), str(row["config"])), number(row["success_rate"], percent=True),
                  number(1000 * float(row["mean_runtime_sec"]), 1), number(1000 * float(row["p90_runtime_sec"]), 1),
                  number(row["mean_quality"]), number(row["p90_quality"]), split_text]
        lines.append(" & ".join(values) + r" \\")
        cells.append(values)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    tex = out_dir / "tables" / f"{name}.tex"
    tex.write_text("\n".join(lines))
    print(f"[write] {tex}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(15, max(2.3, .55 * (len(cells) + 2))))
    axis.set_axis_off()
    table = axis.table(cellText=cells, colLabels=["Method", "Succ. (%)", "Mean RT (ms)", "P90 RT (ms)", "Mean Q", "P90 Q", "S1/S2 Solves"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    figure.tight_layout()
    png = tex.with_suffix(".png"); figure.savefig(png, dpi=220, bbox_inches="tight"); plt.close(figure)
    print(f"[write] {png}")


def plot_family_learning(family: str, metrics: list[dict[str, Any]], probes: list[dict[str, Any]], path: Path) -> None:
    configs = ordered_configs(row for row in metrics if str(row["config"]).endswith("_cl"))
    if not configs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="row")
    panels = ((metrics, "Benchmark", "success_rate", "Success rate", (0, 1.05)), (metrics, "Benchmark", "mean_quality", "Mean quality", (0, 1.05)),
              (metrics, "Benchmark", "mean_runtime_sec", "Mean runtime (s)", None), (probes, "Fixed S1 probe", "success_rate", "Success rate", (0, 1.05)),
              (probes, "Fixed S1 probe", "mean_quality", "Mean quality", (0, 1.05)), (probes, "Fixed S1 probe", "mean_runtime_sec", "Mean runtime (s)", None))
    for axis, (source, title, field, ylabel, ylim) in zip(axes.ravel(), panels):
        for config in configs:
            rows = sorted((row for row in source if row["config"] == config), key=lambda row: int(row["block"]))
            if rows: axis.plot([int(row["block"]) + 1 for row in rows], [row[field] for row in rows], "o-", label=METHOD_LABELS.get(config, config))
        axis.set(title=f"{title}: {ylabel}", xlabel="CL update", ylabel=ylabel)
        if ylim: axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=.3)
    axes[0, 0].legend(frameon=False, fontsize=8); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)
    print(f"[write] {path}")


def plot_paper_figures(probes: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in probes: lookup[(str(row["family"]), str(row["config"]))].append(row)
    for metric, column, ylabel, filename, scale in (
        ("success", "success_rate", "Probe success rate (%)", "continual_learning.pdf", 100.0),
        ("quality", "mean_quality", "Mean trajectory quality $Q$", "continual_learning_quality.pdf", 1.0),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
        for axis, (family, title) in zip(axes.flat, FAMILIES):
            for config, (label, color, marker) in ARMS.items():
                rows = sorted(lookup[(family, config)], key=lambda row: int(row["block"]))
                if rows: axis.plot([int(row["block"]) for row in rows], [scale * float(row[column]) for row in rows], color=color, marker=marker, label=label)
            axis.axvline(-.5, color=".75", linewidth=.8); axis.grid(axis="y", color=".9"); axis.set_title(title, fontsize=10)
            axis.set_xticks([-1, 0, 1, 2, 3, 4]); axis.set_xticklabels(["Base", "B0", "B1", "B2", "B3", "B4"]); axis.set_ylim(0, 100 if metric == "success" else 1)
        for axis in axes[:, 0]: axis.set_ylabel(ylabel)
        for axis in axes[1]: axis.set_xlabel("Cumulative CL block")
        handles, labels = axes.flat[0].get_legend_handles_labels(); figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        figure.tight_layout(rect=(0, 0, 1, .93)); pdf = out_dir / "figures" / filename; pdf.parent.mkdir(exist_ok=True)
        figure.savefig(pdf, bbox_inches="tight"); figure.savefig(pdf.with_suffix(".png"), dpi=300, bbox_inches="tight"); plt.close(figure)
        print(f"[write] {pdf}")


def plot_splits(splits: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in splits: groups[(str(row["family"]), str(row["config"]))].append(row)
    for (family, config), rows in groups.items():
        rows.sort(key=lambda row: int(row["block"])); xs = np.asarray([int(row["block"]) + 1 for row in rows])
        s1 = np.asarray([float(row["s1_success"]) for row in rows]); s2 = np.asarray([float(row["s2_only_success"]) for row in rows])
        figure, axis = plt.subplots(figsize=(7.2, 4.2)); axis.bar(xs-.19, s1, .38, label="S1 success"); axis.bar(xs+.19, s2, .38, label="S2-only success")
        second = axis.twinx(); second.plot(xs, np.divide(s1, s1+s2, out=np.full_like(s1, math.nan), where=s1+s2 > 0), "o-", color="#2ca02c", label="S1 fraction")
        axis.set(title=f"S1 versus S2-only success: {family.replace('_', ' ')} ({config})", xlabel="CL block", ylabel="Successful scenarios", xticks=xs)
        second.set(ylabel="S1 fraction", ylim=(0, 1.05)); axis.grid(axis="y", alpha=.3)
        handles, labels = axis.get_legend_handles_labels(); h2, l2 = second.get_legend_handles_labels(); axis.legend(handles+h2, labels+l2, frameon=False, fontsize=8)
        path = out_dir / "s1_s2_ratio" / f"{family}_{config}.png"; path.parent.mkdir(exist_ok=True); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)
        print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, os.environ.get("MPLCONFIGDIR")))
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else archive_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "tables").mkdir(exist_ok=True)
    suites = {family_name(path): path for path in sorted(archive_dir.glob("nl_*_suite")) if path.is_dir()}
    requested = args.families or list(suites)
    selected = [(family_name(Path(name)), suites.get(family_name(Path(name)), archive_dir / name)) for name in requested]
    metrics, family_metrics, probes, splits, solves = [], [], [], [], {}
    index: dict[str, Any] = {"archive_dir": str(archive_dir), "families": {}}
    for family, suite in selected:
        if not suite.is_dir(): raise FileNotFoundError(suite)
        manifest_path = suite / "suite_manifest.json"; manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        available = sorted(set(manifest.get("configs", {})) | {path.name for path in suite.iterdir() if path.is_dir()})
        configs = args.configs or available
        explicit = [block for config in configs for path in summary_files(suite, config, False) if (block := file_block_index(path)) is not None]
        blocks = max(len(manifest.get("blocks", [])), 1 + max(explicit) if explicit else 0, 1)
        lookup = block_lookup(manifest) or fallback_block_lookup(suite, configs)
        index["families"][family] = {"suite_dir": str(suite), "blocks": blocks, "configs": []}
        for config in configs:
            if config not in available: print(f"[warn] {family}: missing config {config}; skipping"); continue
            try: raw = rows_by_block(suite, manifest, config, args.block_size, lookup)
            except (FileNotFoundError, KeyError):
                try: raw = rows_from_summaries(suite, config, lookup, args.block_size)
                except FileNotFoundError as error: print(f"[warn] {family}/{config}: {error}; skipping"); continue
            rows = aggregate(raw, blocks, args.runtime_field); metrics.extend({"family": family, "config": config, **row} for row in rows)
            overall = aggregate([(0, row) for _, row in raw], 1, args.runtime_field)[0]
            family_metrics.append({"family": family, "config": config, **overall}); index["families"][family]["configs"].append(config)
            probe = probe_rows(suite, manifest, config)
            probes.extend({"family": family, "config": config, **row} for row in aggregate_probe(probe, blocks, args.runtime_field) if row["scenarios"])
            if config.endswith("_cl"):
                split = s1_s2_split(raw, blocks); splits.extend({"family": family, "config": config, **row} for row in split)
                solves[(family, config)] = (sum(row["s1_success"] for row in split), sum(row["s2_only_success"] for row in split))
        if not args.no_learning_plots: plot_family_learning(family, [row for row in metrics if row["family"] == family], [row for row in probes if row["family"] == family], out_dir / f"{family}_continual_learning.png")
    if not family_metrics: raise SystemExit("No result files found.")
    write_csv(out_dir / "metrics_by_block.csv", metrics); write_csv(out_dir / "metrics_by_family.csv", family_metrics)
    if probes: write_csv(out_dir / "probe_metrics_by_block.csv", probes)
    if splits: write_csv(out_dir / "s1_s2_split_by_block.csv", splits)
    summary = ["# Archive Summary", "", "| Family | Config | Scenarios | Success | Mean runtime (s) | Mean Q |", "|---|---|---:|---:|---:|---:|"]
    summary.extend(f"| {row['family']} | {row['config']} | {row['scenarios']} | {number(row['success_rate'])} | {number(row['mean_runtime_sec'])} | {number(row['mean_quality'])} |" for row in family_metrics)
    (out_dir / "family_summary.md").write_text("\n".join(summary) + "\n")
    for family, _ in selected:
        rows = [row for row in family_metrics if row["family"] == family]
        if rows: write_tables(out_dir, rows, solves, f"{family}_results", f"Aggregate benchmark results for the {family.replace('_', ' ')} environment family.", f"tab:{family}_results")
    macro = []
    for config in ordered_configs(family_metrics):
        group = [row for row in family_metrics if row["config"] == config]
        average = {"family": "all_families", "config": config, "families": len(group)}
        for field in ("scenarios", "successes", "success_rate", "mean_runtime_sec", "p90_runtime_sec", "mean_quality", "median_quality", "p90_quality"):
            values = [float(row[field]) for row in group if math.isfinite(float(row[field]))]; average[field] = float(np.mean(values)) if values else math.nan
        macro.append(average)
    macro_solves = {("all_families", config): (float(np.mean([value[0] for (family, key), value in solves.items() if key == config])), float(np.mean([value[1] for (family, key), value in solves.items() if key == config]))) for config in ordered_configs(macro) if any(key == config for _, key in solves)}
    write_csv(out_dir / "metrics_macro_by_config.csv", macro)
    write_tables(out_dir, macro, macro_solves, "aggregate_results", f"Aggregate benchmark results averaged across the {len(selected)} environment families.", "tab:main_results")
    (out_dir / "manifest_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    if probes and not args.skip_paper_figures: plot_paper_figures(probes, out_dir)
    if splits and not args.skip_s1_s2_plots: plot_splits(splits, out_dir)


if __name__ == "__main__":
    main()
