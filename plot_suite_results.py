#!/usr/bin/env python3
"""

Plot block-wise success rates and runtimes for suite results.


python plot_suite_results.py\
  --suite_dir output/benchmark_runs/nl_bugtrap_suite \
  --configs s1_neural s2_cbf s2_mpc\
  --out output/benchmark_runs/nl_bugtrap_suite/results_by_block.png

  
python plot_suite_results.py\
  --suite_dir output/benchmark_runs/nl_dense_clutter_suite \
  --configs s1_neural s2_cbf s2_mpc\
  --out output/benchmark_runs/nl_dense_clutter_suite/results_by_block.png

  
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_CONFIGS = ["s1_neural", "s2_cbf", "sofai_cbf_cl"]
CONFIG_ALIASES = {
    "sofai_cbf_cf": "sofai_cbf_cl",
    "sofai_mpc_cf": "sofai_mpc_cl",
}

LABELS = {
    "s1_neural": "S1 neural",
    "s2_cbf": "S2 CBF",
    "sofai_cbf_cl": "SOFAI CBF CL",
    "sofai_mpc_cl": "SOFAI MPC CL",
}

COLORS = {
    "s1_neural": "#1f77b4",
    "s2_cbf": "#d62728",
    "sofai_cbf_cl": "#2ca02c",
    "sofai_mpc_cl": "#ff7f0e",
}

MARKERS = {
    "s1_neural": "o",
    "s2_cbf": "s",
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
        default="output/benchmark_runs/nl_dense_clutter_suite/success_rate_by_block.png",
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
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else suite_dir / "suite_manifest.json"
    manifest = read_manifest(manifest_path)

    block_size = int(manifest["block_size"])
    n_blocks = math.ceil(int(manifest["scenario_count"]) / block_size)

    configs = [normalize_config(c) for c in args.configs]
    series: Dict[str, Dict[str, List[float]]] = {}
    for cfg in configs:
        rows = collect_rows(suite_dir, manifest, cfg)
        success = block_success_rates(rows, block_size, n_blocks)
        runtime_mean, runtime_p90 = block_runtime_stats(rows, block_size, n_blocks, args.runtime_field)
        series[cfg] = {
            "success": success,
            "runtime_mean": runtime_mean,
            "runtime_p90": runtime_p90,
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_success, ax_runtime) = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True)
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

    ax_success.set_ylabel("Success rate")
    ax_success.set_ylim(0.0, 1.05)
    ax_success.grid(True, alpha=0.3)
    ax_success.legend(frameon=False, ncols=2)
    ax_success.set_title("Block-wise success rate and runtime across solvers")

    ax_runtime.set_xlabel("Block")
    ax_runtime.set_ylabel("Runtime (s)")
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.legend(frameon=False, ncols=2)
    ax_runtime.set_xticks(xs)
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
        )


if __name__ == "__main__":
    main()
