#!/usr/bin/env python3
"""

Plot block-wise success rates, runtimes, and trajectory quality for suite results.


python plot_suite_results.py\
  --suite_dir output/benchmark_runs/nl_bugtrap_suite \
  --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl\
  --out output/benchmark_runs/nl_bugtrap_suite/results_by_block.png

  
python plot_suite_results.py\
  --suite_dir output/benchmark_runs/nl_maze_branching_suite \
  --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl\
  --out output/benchmark_runs/nl_maze_branching_suite/results_by_block.png


small_open large_sparse wall_gap serial_walls maze_branching



"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from solvers._s2_common import resolve_mplconfigdir


DEFAULT_CONFIGS = ["s1_neural", "s2_cbf", "s2_mpc", "sofai_cbf_cl"]
CONFIG_ALIASES = {
    "sofai_cbf_cf": "sofai_cbf_cl",
    "sofai_mpc_cf": "sofai_mpc_cl",
}

LABELS = {
    "s1_neural": "S1 neural",
    "s2_cbf": "S2 CBF",
    "s2_mpc": "S2 MPC",
    "sofai_cbf_cl": "SOFAI CBF CL",
    "sofai_mpc_cl": "SOFAI MPC CL",
}

COLORS = {
    "s1_neural": "#1f77b4",
    "s2_cbf": "#d62728",
    "s2_mpc": "#9467bd",
    "sofai_cbf_cl": "#2ca02c",
    "sofai_mpc_cl": "#ff7f0e",
}

MARKERS = {
    "s1_neural": "o",
    "s2_cbf": "s",
    "s2_mpc": "X",
    "sofai_cbf_cl": "^",
    "sofai_mpc_cl": "D",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite_dir",
        default="output/benchmark_runs/nl_dense_clutter_suite",
        help="Directory containing suite_manifest.json and per-config results.",
    )
    p.add_argument(
        "--manifest",
        default="",
        help="Optional explicit path to suite_manifest.json.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Configs to plot. Aliases like sofai_cbf_cf are accepted.",
    )
    p.add_argument(
        "--out",
        default="output/benchmark_runs/nl_dense_clutter_suite/metrics_by_block.png",
        help="Output plot path.",
    )
    p.add_argument(
        "--runtime_field",
        default="selected_runtime_sec",
        help="Primary runtime field to aggregate from each JSONL row.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def normalize_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def block_success_rates(rows: Sequence[Dict[str, Any]], block_size: int, n_blocks: int) -> List[float]:
    buckets: List[List[bool]] = [[] for _ in range(n_blocks)]
    for row in rows:
        sid = int(row.get("scenario_id", -1))
        if sid < 0:
            continue
        block_idx = sid // block_size
        if 0 <= block_idx < n_blocks:
            buckets[block_idx].append(bool(row.get("success", False)))

    out: List[float] = []
    for bucket in buckets:
        out.append(sum(bucket) / len(bucket) if bucket else math.nan)
    return out


def runtime_value(row: Dict[str, Any], runtime_field: str) -> float | None:
    candidates = [runtime_field, "selected_runtime_sec", "runtime_sec", "wall_runtime_sec"]
    for key in candidates:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def block_runtime_stats(
    rows: Sequence[Dict[str, Any]],
    block_size: int,
    n_blocks: int,
    runtime_field: str,
) -> tuple[List[float], List[float]]:
    buckets: List[List[float]] = [[] for _ in range(n_blocks)]
    for row in rows:
        sid = int(row.get("scenario_id", -1))
        if sid < 0:
            continue
        block_idx = sid // block_size
        if 0 <= block_idx < n_blocks:
            value = runtime_value(row, runtime_field)
            if value is not None:
                buckets[block_idx].append(value)

    means: List[float] = []
    p90s: List[float] = []
    for bucket in buckets:
        if not bucket:
            means.append(math.nan)
            p90s.append(math.nan)
            continue
        arr = np.asarray(bucket, dtype=float)
        means.append(float(np.mean(arr)))
        p90s.append(float(np.percentile(arr, 90)))
    return means, p90s


def quality_value(row: Dict[str, Any]) -> float | None:
    for key in ("quality_score", "quality_j"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        return value if key == "quality_score" else 1.0 / (1.0 + value)

    weights = [
        row.get("quality_weight_path_length"),
        row.get("quality_weight_control_effort"),
        row.get("quality_weight_smoothness"),
    ]
    comps = [
        row.get("quality_path_length"),
        row.get("quality_control_effort"),
        row.get("quality_smoothness"),
        row.get("quality_path_length_ref"),
        row.get("quality_control_effort_ref"),
        row.get("quality_smoothness_ref"),
    ]
    if any(v in (None, "") for v in comps):
        return None
    try:
        path_length = float(comps[0])
        control_effort = float(comps[1])
        smoothness = float(comps[2])
        path_ref = float(comps[3])
        control_ref = float(comps[4])
        smooth_ref = float(comps[5])
        if any(v in (None, "") for v in weights):
            weight_path = weight_control = weight_smooth = 1.0 / 3.0
        else:
            weight_path = float(weights[0])
            weight_control = float(weights[1])
            weight_smooth = float(weights[2])
    except (TypeError, ValueError):
        return None
    if min(path_length, control_effort, smoothness, path_ref, control_ref, smooth_ref) <= 0.0:
        return None
    total = weight_path + weight_control + weight_smooth
    if total <= 0.0:
        weight_path = weight_control = weight_smooth = 1.0 / 3.0
        total = 1.0
    j = (
        weight_path * path_length / path_ref
        + weight_control * control_effort / control_ref
        + weight_smooth * smoothness / smooth_ref
    ) / total
    return 1.0 / (1.0 + j)


def block_quality_stats(
    rows: Sequence[Dict[str, Any]],
    block_size: int,
    n_blocks: int,
) -> tuple[List[float], List[float], List[float]]:
    buckets: List[List[float]] = [[] for _ in range(n_blocks)]
    for row in rows:
        if not bool(row.get("success", False)):
            continue
        sid = int(row.get("scenario_id", -1))
        if sid < 0:
            continue
        block_idx = sid // block_size
        if not (0 <= block_idx < n_blocks):
            continue
        value = quality_value(row)
        if value is None:
            continue
        buckets[block_idx].append(value)

    means: List[float] = []
    medians: List[float] = []
    p90s: List[float] = []
    for bucket in buckets:
        if not bucket:
            means.append(math.nan)
            medians.append(math.nan)
            p90s.append(math.nan)
            continue
        arr = np.asarray(bucket, dtype=float)
        means.append(float(np.mean(arr)))
        medians.append(float(np.median(arr)))
        p90s.append(float(np.percentile(arr, 90)))
    return means, medians, p90s


def collect_rows(suite_dir: Path, manifest: Dict[str, Any], cfg: str) -> List[Dict[str, Any]]:
    cfg_manifest = manifest.get("configs", {}).get(cfg)
    if not cfg_manifest:
        raise KeyError(f"Config {cfg!r} not found in manifest.")

    rows: List[Dict[str, Any]] = []
    runs = cfg_manifest.get("runs", [])
    if cfg in {"s1_neural", "s2_cbf", "s2_mpc"}:
        jsonl = suite_dir / cfg / f"{cfg}_runs.jsonl"
        rows.extend(read_jsonl(jsonl))
        return rows

    for run in runs:
        jsonl = Path(run["jsonl"]).expanduser()
        if not jsonl.is_absolute():
            jsonl = suite_dir / jsonl
        rows.extend(read_jsonl(jsonl))
    return rows


def main() -> None:
    args = parse_args()

    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(Path(__file__).resolve().parent, os.environ.get("MPLCONFIGDIR")))
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else suite_dir / "suite_manifest.json"
    manifest = read_manifest(manifest_path)

    block_size = int(manifest["block_size"])
    n_blocks = math.ceil(int(manifest["scenario_count"]) / block_size)

    requested_configs = [normalize_config(c) for c in args.configs]
    configs: List[str] = []
    rows_by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in requested_configs:
        try:
            rows = collect_rows(suite_dir, manifest, cfg)
        except KeyError as exc:
            print(f"[warn] {exc} skipping")
            continue
        if not rows:
            print(f"[warn] no rows found for {cfg!r}, skipping")
            continue
        configs.append(cfg)
        rows_by_cfg[cfg] = rows

    if not configs:
        raise RuntimeError("No requested configs were available in the suite manifest.")

    series: Dict[str, Dict[str, List[float]]] = {}
    for cfg in configs:
        rows = rows_by_cfg[cfg]
        success = block_success_rates(rows, block_size, n_blocks)
        runtime_mean, runtime_p90 = block_runtime_stats(rows, block_size, n_blocks, args.runtime_field)
        quality_mean, quality_median, quality_p90 = block_quality_stats(rows, block_size, n_blocks)
        series[cfg] = {
            "success": success,
            "runtime_mean": runtime_mean,
            "runtime_p90": runtime_p90,
            "quality_mean": quality_mean,
            "quality_median": quality_median,
            "quality_p90": quality_p90,
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_success, ax_runtime, ax_quality) = plt.subplots(3, 1, figsize=(9.0, 10.6), sharex=True)
    xs = list(range(1, n_blocks + 1))

    for cfg in configs:
        ys = series[cfg]["success"]
        ax_success.plot(
            xs,
            ys,
            marker=MARKERS.get(cfg, "o"),
            linewidth=2.2,
            markersize=6,
            color=COLORS.get(cfg, None),
            label=LABELS.get(cfg, cfg),
        )

    for cfg in configs:
        mean_rt = series[cfg]["runtime_mean"]
        p90_rt = series[cfg]["runtime_p90"]
        color = COLORS.get(cfg, None)
        label = LABELS.get(cfg, cfg)
        ax_runtime.plot(
            xs,
            mean_rt,
            marker=MARKERS.get(cfg, "o"),
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"{label} mean/avg",
        )
        ax_runtime.plot(
            xs,
            p90_rt,
            marker=None,
            linewidth=1.7,
            linestyle="--",
            color=color,
            alpha=0.9,
            label=f"{label} p90",
        )

    for cfg in configs:
        mean_q = series[cfg]["quality_mean"]
        p90_q = series[cfg]["quality_p90"]
        color = COLORS.get(cfg, None)
        label = LABELS.get(cfg, cfg)
        ax_quality.plot(
            xs,
            mean_q,
            marker=MARKERS.get(cfg, "o"),
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"{label} mean",
        )
        ax_quality.plot(
            xs,
            p90_q,
            marker=None,
            linewidth=1.7,
            linestyle="--",
            color=color,
            alpha=0.9,
            label=f"{label} p90",
        )

    ax_success.set_ylabel("Success rate")
    ax_success.set_ylim(0.0, 1.05)
    ax_success.grid(True, alpha=0.3)
    ax_success.legend(frameon=False, ncols=2)
    ax_success.set_title("Block-wise success rate, runtime, and quality across solvers")

    ax_runtime.set_ylabel("Runtime (s)")
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.legend(frameon=False, ncols=2)

    ax_quality.set_xlabel("Block")
    ax_quality.set_ylabel("Quality score Q")
    ax_quality.set_ylim(0.0, 1.05)
    ax_quality.grid(True, alpha=0.3)
    ax_quality.legend(frameon=False, ncols=2)
    ax_quality.set_xticks(xs)
    fig.tight_layout()

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = suite_dir / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[write] {out}")

    for cfg in configs:
        print(
            cfg,
            "success=",
            [f"{v:.3f}" if not math.isnan(v) else "nan" for v in series[cfg]["success"]],
            "mean_rt=",
            [f"{v:.3f}" if not math.isnan(v) else "nan" for v in series[cfg]["runtime_mean"]],
            "p90_rt=",
            [f"{v:.3f}" if not math.isnan(v) else "nan" for v in series[cfg]["runtime_p90"]],
            "mean_q=",
            [f"{v:.3f}" if not math.isnan(v) else "nan" for v in series[cfg]["quality_mean"]],
            "median_q=",
            [f"{v:.3f}" if not math.isnan(v) else "nan" for v in series[cfg]["quality_median"]],
            "p90_q=",
            [f"{v:.3f}" if not math.isnan(v) else "nan" for v in series[cfg]["quality_p90"]],
        )


if __name__ == "__main__":
    main()
