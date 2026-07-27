#!/usr/bin/env python3
"""Analyze all family suites in one ``run_suite.py`` archive.

Example:

PYTHONDONTWRITEBYTECODE=1 \
python analyze_suite.py \
  --archive_dir output/benchmark_runs_replay_dagger \
  --configs sofai_cbf_cl

The archive must contain one directory per family, conventionally named
``nl_<family>_suite``, with a ``suite_manifest.json`` in each directory.
Results are written to ``<archive_dir>/analysis`` by default.
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

import analyze_archive_results as archive
from solvers._s2_common import resolve_mplconfigdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_dir", default="output/benchmark_runs_v1")
    parser.add_argument(
        "--families",
        nargs="+",
        default=[],
        help="Family names (e.g. bugtrap) or suite directory names. Defaults to all discovered suites.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[],
        help="Solver configs to include. Defaults to every config discovered in each family.",
    )
    parser.add_argument(
        "--runtime_field",
        default="planning_runtime_sec",
        help="Runtime field, with the same fallbacks as analyze_archive_results.py.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=0,
        help="Fallback block size only when a suite manifest has no block definition.",
    )
    parser.add_argument("--out_dir", default="", help="Defaults to <archive_dir>/analysis.")
    parser.add_argument("--no_learning_plots", action="store_true", help="Skip per-family continual-learning plots.")
    return parser.parse_args()


# Summary CSVs store Python ``repr`` booleans, so an unconverted string such as
# "False" would otherwise be truthy in every success count.
BOOLEAN_COLUMNS = frozenset(
    {
        "timed_out",
        "success",
        "collision_free",
        "goal_reached",
        "s1_attempted",
        "s1_success",
        "s1_collision_free",
        "s1_goal_reached",
        "s2_attempted",
        "s2_success",
        "s2_collision_free",
        "s2_goal_reached",
        "s2_skipped",
    }
)


def parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    """Read a per-scenario summary CSV into JSONL-compatible row dicts."""
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
            # ``s1_s2_split`` reads per-system outcomes from ``attempts``, which the
            # flat CSV flattens into s1_*/s2_* columns.
            attempts: list[dict[str, Any]] = []
            for system in ("s1", "s2"):
                if row.get(f"{system}_attempted"):
                    attempts.append({"system": system, "success": bool(row.get(f"{system}_success"))})
            row["attempts"] = attempts
            rows.append(row)
    return rows


def summary_files(suite_dir: Path, config: str, *, probe: bool) -> list[Path]:
    config_dir = suite_dir / config
    if not config_dir.is_dir():
        return []
    return [
        path
        for path in sorted(config_dir.rglob("*_summary.csv"))
        if ("probe" in path.parts) is probe
    ]


def result_files(suite_dir: Path, config: str) -> list[Path]:
    config_dir = suite_dir / config
    jsonl = [path for path in config_dir.rglob("*_runs.jsonl")] if config_dir.is_dir() else []
    return jsonl or summary_files(suite_dir, config, probe=False)


def file_block_index(path: Path) -> int | None:
    match = re.search(r"_block(\d+)", path.stem)
    return int(match.group(1)) if match else None


def block_lookup_from_summaries(suite_dir: Path, configs: Iterable[str]) -> dict[int, int]:
    """Recover the CL block partition from per-block summary CSVs."""
    lookup: dict[int, int] = {}
    for config in configs:
        for path in summary_files(suite_dir, config, probe=False):
            block = file_block_index(path)
            if block is None:
                continue
            for row in read_summary_csv(path):
                scenario_index = row.get("scenario_index", row.get("scenario_id"))
                if scenario_index is not None:
                    lookup.setdefault(int(scenario_index), block)
    return lookup


def rows_from_summaries(
    suite_dir: Path,
    config: str,
    lookup: dict[int, int],
    block_size: int,
) -> list[tuple[int, dict[str, Any]]]:
    files = summary_files(suite_dir, config, probe=False)
    if not files:
        raise FileNotFoundError(f"No summary CSV found in {suite_dir / config}")

    rows: list[tuple[int, dict[str, Any]]] = []
    for path in files:
        explicit = file_block_index(path)
        for row in read_summary_csv(path):
            if explicit is not None:
                block = explicit
            else:
                scenario_index = row.get("scenario_index", row.get("scenario_id"))
                block = lookup.get(int(scenario_index), -1) if scenario_index is not None else -1
                if block < 0:
                    block = (int(scenario_index) // block_size) if (block_size and scenario_index is not None) else 0
            rows.append((block, row))
    return rows


def probe_rows_from_summaries(suite_dir: Path, config: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for path in summary_files(suite_dir, config, probe=True):
        if "base_probe" in path.stem:
            block = -1
        else:
            block = file_block_index(path)
            if block is None:
                continue
        rows.extend((block, row) for row in read_summary_csv(path))
    return rows


def discover_suites(archive_dir: Path) -> dict[str, Path]:
    suites: dict[str, Path] = {}
    for path in sorted(archive_dir.glob("nl_*_suite")):
        if path.is_dir() and has_results(path):
            suites[archive.family_name(path)] = path
    return suites


def has_results(suite_dir: Path) -> bool:
    if (suite_dir / "suite_manifest.json").is_file():
        return True
    return any(suite_dir.rglob("*_runs.jsonl")) or any(suite_dir.rglob("*_summary.csv"))


def select_suites(archive_dir: Path, requested: Iterable[str]) -> list[tuple[str, Path]]:
    discovered = discover_suites(archive_dir)
    if not requested:
        return sorted(discovered.items())

    selected: list[tuple[str, Path]] = []
    for raw in requested:
        name = str(raw).strip()
        if not name:
            continue
        family = archive.family_name(Path(name))
        path = discovered.get(family)
        if path is None:
            direct = archive_dir / name
            path = direct if has_results(direct) else archive_dir / f"nl_{family}_suite"
        if not has_results(path):
            raise FileNotFoundError(f"No results found for family {name!r} under {archive_dir}")
        selected.append((archive.family_name(path), path))
    return selected


def suite_configs(suite_dir: Path, manifest: dict[str, Any]) -> list[str]:
    configured = list(manifest.get("configs", {}))
    on_disk = [
        path.name
        for path in suite_dir.iterdir()
        if path.is_dir() and result_files(suite_dir, path.name)
    ]
    return sorted(set(configured) | set(on_disk))


def block_count(suite_dir: Path, manifest: dict[str, Any], configs: list[str], block_size: int) -> tuple[int, int]:
    # ``run_suite.py`` rewrites suite_manifest.json each time it is invoked.
    # A later standalone/S1-only invocation can therefore leave a one-block
    # manifest beside completed multi-block CL runs. Never let that stale
    # manifest truncate block files already present on disk.
    manifest_blocks = len(manifest.get("blocks", []))
    fallback_size = max(0, int(block_size))
    if not manifest_blocks and not fallback_size:
        fallback_size = archive.inferred_block_size(suite_dir, configs)
    jsonl_blocks = archive.inferred_block_count(suite_dir, configs)
    csv_block_indices = [
        block
        for config in configs
        for path in summary_files(suite_dir, config, probe=False)
        if (block := file_block_index(path)) is not None
    ]
    csv_blocks = 1 + max(csv_block_indices) if csv_block_indices else 0
    blocks = max(manifest_blocks, jsonl_blocks, csv_blocks)
    return blocks, fallback_size


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def format_float(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(value) else f"{value:.3f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Archive Summary",
        "",
        "| Family | Config | Scenarios | Success | Mean runtime (s) | Mean Q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['config']} | {row['scenarios']} | "
            f"{format_float(row['success_rate'])} | {format_float(row['mean_runtime_sec'])} | "
            f"{format_float(row['mean_quality'])} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


METHOD_LABELS = {
    "s1_neural": "NN",
    "s2_mpc": "MPC",
    "s2_cbf": "CBF",
    "sofai_cbf_cl": "sofai CBF cl",
    "sofai_mpc_cl": "sofai MPC cl",
    "sofai_mpc_warm_cl": "sofai MPC warm cl",
}
METHOD_ORDER = tuple(METHOD_LABELS)


def ordered_configs(rows: Iterable[dict[str, Any]]) -> list[str]:
    present = {str(row["config"]) for row in rows}
    return [config for config in METHOD_ORDER if config in present] + sorted(present.difference(METHOD_ORDER))


def latex_value(value: Any, *, digits: int = 3, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return f"{100.0 * number:.1f}" if percent else f"{number:.{digits}f}"


def write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    s1_s2_solves: dict[tuple[str, str], tuple[float, float]],
    *,
    caption: str,
    label: str,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Succ. (\%)} & \textbf{Mean RT (ms)} & \textbf{P90 RT (ms)} & \textbf{Mean \(Q\)} & \textbf{P90 \(Q\)} & \textbf{S1/S2 Solves} \\",
        r"\midrule",
    ]
    for row in rows:
        family, config = str(row["family"]), str(row["config"])
        solves = s1_s2_solves.get((family, config))
        solve_text = "-- / --" if solves is None else f"{int(round(solves[0]))} / {int(round(solves[1]))}"
        lines.append(
            f"{METHOD_LABELS.get(config, config)} & "
            f"{latex_value(row['success_rate'], percent=True)} & "
            f"{latex_value(1000.0 * float(row['mean_runtime_sec']), digits=1)} & "
            f"{latex_value(1000.0 * float(row['p90_runtime_sec']), digits=1)} & "
            f"{latex_value(row['mean_quality'])} & "
            f"{latex_value(row['p90_quality'])} & {solve_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines))
    print(f"[write] {path}")


def write_table_png(
    path: Path,
    rows: list[dict[str, Any]],
    s1_s2_solves: dict[tuple[str, str], tuple[float, float]],
    *,
    title: str,
) -> None:
    """Render the paper table as a PNG for quick visual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["Method", "Succ. (%)", "Mean RT (ms)", "P90 RT (ms)", "Mean Q", "P90 Q", "S1/S2 Solves"]
    cells: list[list[str]] = []
    for row in rows:
        family, config = str(row["family"]), str(row["config"])
        solves = s1_s2_solves.get((family, config))
        solve_text = "-- / --" if solves is None else f"{int(round(solves[0]))} / {int(round(solves[1]))}"
        cells.append(
            [
                METHOD_LABELS.get(config, config),
                latex_value(row["success_rate"], percent=True),
                latex_value(1000.0 * float(row["mean_runtime_sec"]), digits=1),
                latex_value(1000.0 * float(row["p90_runtime_sec"]), digits=1),
                latex_value(row["mean_quality"]),
                latex_value(row["p90_quality"]),
                solve_text,
            ]
        )
    figure, axis = plt.subplots(figsize=(15, max(2.3, 0.55 * (len(cells) + 2))))
    axis.set_axis_off()
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.19, 0.11, 0.15, 0.15, 0.10, 0.10, 0.16],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        if row_index == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")
    figure.suptitle(title, fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[write] {path}")


def macro_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in ordered_configs(rows):
        group = [row for row in rows if row["config"] == config]
        summary: dict[str, Any] = {"family": "all_families", "config": config, "families": len(group)}
        for field in (
            "scenarios",
            "successes",
            "success_rate",
            "mean_runtime_sec",
            "p90_runtime_sec",
            "mean_quality",
            "median_quality",
            "p90_quality",
        ):
            values = [float(row[field]) for row in group if math.isfinite(float(row[field]))]
            summary[field] = float(np.mean(values)) if values else math.nan
        output.append(summary)
    return output


def plot_continual_learning(
    family: str,
    metrics: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = ordered_configs(row for row in metrics if str(row["config"]).endswith("_cl"))
    if not configs:
        return
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="row")
    panels = (
        (metrics, "SOFAI benchmark", "success_rate", "Success rate", (0.0, 1.05)),
        (metrics, "SOFAI benchmark", "mean_quality", "Mean quality", (0.0, 1.05)),
        (metrics, "SOFAI benchmark", "mean_runtime_sec", "Mean runtime (s)", None),
        (probes, "Fixed S1 probe", "success_rate", "Success rate", (0.0, 1.05)),
        (probes, "Fixed S1 probe", "mean_quality", "Mean quality", (0.0, 1.05)),
        (probes, "Fixed S1 probe", "mean_runtime_sec", "Mean runtime (s)", None),
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    for axis, (source, subtitle, field, ylabel, ylim) in zip(axes.ravel(), panels):
        for config, color in zip(configs, colors):
            rows = sorted((row for row in source if row["config"] == config), key=lambda row: int(row["block"]))
            if rows:
                x = [int(row["block"]) + 1 for row in rows]
                axis.plot(x, [row[field] for row in rows], "o-", color=color, label=METHOD_LABELS.get(config, config))
        axis.set(title=f"{subtitle}: {ylabel}", xlabel="CL update (0 = base probe)", ylabel=ylabel)
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{family}: continual-learning effect", y=1.02)
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    print(f"[write] {out_path}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, os.environ.get("MPLCONFIGDIR")))
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    if not archive_dir.is_dir():
        raise FileNotFoundError(archive_dir)
    suites = select_suites(archive_dir, args.families)
    if not suites:
        raise SystemExit(f"No family suites found under {archive_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else archive_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_configs = list(args.configs)
    metrics_by_block: list[dict[str, Any]] = []
    metrics_by_family: list[dict[str, Any]] = []
    probe_metrics: list[dict[str, Any]] = []
    split_metrics: list[dict[str, Any]] = []
    s1_s2_solves: dict[tuple[str, str], tuple[float, float]] = {}
    index: dict[str, Any] = {"archive_dir": str(archive_dir), "families": {}}

    for family, suite_dir in suites:
        manifest_path = suite_dir / "suite_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        available = suite_configs(suite_dir, manifest)
        configs = requested_configs or available
        n_blocks, fallback_size = block_count(suite_dir, manifest, configs, args.block_size)
        if not n_blocks:
            n_blocks = 1
        # Standalone S1/S2 runs carry no block tag, so recover the partition from
        # the manifest when present and from the CL block CSVs otherwise.
        lookup = archive.block_lookup(manifest) or block_lookup_from_summaries(suite_dir, configs)
        family_index: dict[str, Any] = {
            "suite_dir": str(suite_dir),
            "blocks": n_blocks,
            "configs": [],
        }
        raw_by_config: dict[str, list[tuple[int, dict[str, Any]]]] = {}

        for config in configs:
            if config not in available:
                print(f"[warn] {family}: missing config {config}; skipping")
                continue
            try:
                raw_rows = archive.rows_by_block(suite_dir, manifest, config, fallback_size)
            except (FileNotFoundError, KeyError):
                try:
                    raw_rows = rows_from_summaries(suite_dir, config, lookup, fallback_size)
                except (FileNotFoundError, KeyError) as error:
                    print(f"[warn] {family}/{config}: {error}; skipping")
                    continue
            raw_by_config[config] = raw_rows
            rows = archive.aggregate(raw_rows, n_blocks, args.runtime_field)
            metrics_by_block.extend({"family": family, "config": config, **row} for row in rows)
            overall = archive.aggregate([(0, row) for _, row in raw_rows], 1, args.runtime_field)[0]
            metrics_by_family.append({"family": family, "config": config, **overall})
            family_index["configs"].append(config)

            try:
                raw_probes = archive.probe_rows_by_block(suite_dir, manifest, config)
            except FileNotFoundError as error:
                print(f"[warn] {family}/{config}: {error}; skipping raw probes")
                raw_probes = []
            if not raw_probes:
                raw_probes = probe_rows_from_summaries(suite_dir, config)
            if raw_probes:
                probe_metrics.extend(
                    {"family": family, "config": config, **row}
                    for row in archive.aggregate_probe(raw_probes, n_blocks, args.runtime_field)
                    if row["scenarios"]
                )

            if config.endswith("_cl"):
                split = archive.s1_s2_split(raw_rows, n_blocks)
                split_metrics.extend({"family": family, "config": config, **row} for row in split)
                s1_s2_solves[(family, config)] = (
                    sum(float(row["s1_success"]) for row in split),
                    sum(float(row["s2_only_success"]) for row in split),
                )

        index["families"][family] = family_index
        if not args.no_learning_plots:
            plot_continual_learning(
                family,
                [row for row in metrics_by_block if row["family"] == family],
                [row for row in probe_metrics if row["family"] == family],
                out_dir / f"{family}_continual_learning.png",
            )

    if not metrics_by_family:
        raise SystemExit("No result JSONLs were found for the requested families/configs.")
    write_csv(out_dir / "metrics_by_block.csv", metrics_by_block)
    write_csv(out_dir / "metrics_by_family.csv", metrics_by_family)
    if probe_metrics:
        write_csv(out_dir / "probe_metrics_by_block.csv", probe_metrics)
    if split_metrics:
        write_csv(out_dir / "s1_s2_split_by_block.csv", split_metrics)
    write_markdown(out_dir / "family_summary.md", metrics_by_family)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for family, _ in suites:
        family_rows = [row for row in metrics_by_family if row["family"] == family]
        write_latex_table(
            tables_dir / f"{family}_results.tex",
            sorted(family_rows, key=lambda row: ordered_configs(family_rows).index(str(row["config"]))),
            s1_s2_solves,
            caption=f"Aggregate benchmark results for the {family.replace('_', ' ')} environment family.",
            label=f"tab:{family}_results",
        )
        write_table_png(
            tables_dir / f"{family}_results.png",
            sorted(family_rows, key=lambda row: ordered_configs(family_rows).index(str(row["config"]))),
            s1_s2_solves,
            title=f"Aggregate benchmark results: {family.replace('_', ' ')}",
        )
    macro_rows = macro_average(metrics_by_family)
    macro_solves = {
        ("all_families", config): (
            float(np.mean([value[0] for (family, key), value in s1_s2_solves.items() if key == config])),
            float(np.mean([value[1] for (family, key), value in s1_s2_solves.items() if key == config])),
        )
        for config in ordered_configs(macro_rows)
        if any(key == config for _, key in s1_s2_solves)
    }
    write_csv(out_dir / "metrics_macro_by_config.csv", macro_rows)
    write_latex_table(
        tables_dir / "aggregate_results.tex",
        macro_rows,
        macro_solves,
        caption=f"Aggregate benchmark results averaged across the {len(suites)} environment families.",
        label="tab:main_results",
    )
    write_table_png(
        tables_dir / "aggregate_results.png",
        macro_rows,
        macro_solves,
        title=f"Aggregate benchmark results averaged across {len(suites)} environment families",
    )
    (out_dir / "manifest_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"[write] {out_dir / 'manifest_index.json'}")


if __name__ == "__main__":
    main()
