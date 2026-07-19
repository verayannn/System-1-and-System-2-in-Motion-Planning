#!/usr/bin/env python3
"""

Plot per-block S1 success versus S2-only success for SOFAI CL runs.

  
python plot_suite_s1_s2_ratio.py \
  --suite_dir output/benchmark_runs/nl_bugtrap_suite \
  --config sofai_mpc_cl \
  --out output/benchmark_runs/nl_bugtrap_suite/s1_s2_ratio_by_block.png

"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite_dir",
        default="output/benchmark_runs/nl_dense_clutter_suite",
        help="Directory containing suite_manifest.json.",
    )
    p.add_argument(
        "--config",
        default="sofai_cbf_cl",
        help="Continual-learning config to plot.",
    )
    p.add_argument(
        "--out",
        default="output/benchmark_runs/nl_dense_clutter_suite/s1_s2_ratio_by_block.png",
        help="Output PNG path.",
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


def run_block_index(run: Dict[str, Any], default: int) -> int:
    prefix = str(run.get("prefix", ""))
    if "_block" in prefix:
        tail = prefix.rsplit("_block", 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits)
    return default


def resolve_jsonl(suite_dir: Path, config: str, run: Dict[str, Any]) -> Path:
    prefix = str(run.get("prefix", "")).strip()
    if prefix:
        local = suite_dir / config / "runs" / f"{prefix}_runs.jsonl"
        if local.is_file():
            return local

    path = Path(str(run["jsonl"])).expanduser()
    if path.is_file():
        return path

    parts = path.parts
    if suite_dir.name in parts:
        idx = parts.index(suite_dir.name)
        local = suite_dir.joinpath(*parts[idx + 1 :])
        if local.is_file():
            return local
    return path


def count_attempts(block_rows: List[tuple[int, Dict[str, Any]]], n_blocks: int) -> tuple[List[int], List[int]]:
    s1 = [0] * n_blocks
    s2_only = [0] * n_blocks
    for b, row in block_rows:
        if b < 0 or b >= n_blocks:
            continue
        attempts = row.get("attempts", [])
        s1_success = any(str(attempt.get("system")) == "s1" and bool(attempt.get("success")) for attempt in attempts)
        s2_success = any(str(attempt.get("system")) == "s2" and bool(attempt.get("success")) for attempt in attempts)
        if s1_success:
            s1[b] += 1
        elif s2_success:
            s2_only[b] += 1
    return s1, s2_only


def main() -> None:
    args = parse_args()
    from solvers._s2_common import resolve_mplconfigdir

    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(Path(__file__).resolve().parent, os.environ.get("MPLCONFIGDIR")))

    suite_dir = Path(args.suite_dir).expanduser().resolve()
    manifest = json.loads((suite_dir / "suite_manifest.json").read_text())
    n_blocks = len(manifest.get("blocks", [])) or math.ceil(int(manifest["scenario_count"]) / int(manifest["block_size"]))

    cfg = manifest["configs"][args.config]
    block_rows: List[tuple[int, Dict[str, Any]]] = []
    for i, run in enumerate(cfg.get("runs", [])):
        block_idx = run_block_index(run, i)
        jsonl = resolve_jsonl(suite_dir, args.config, run)
        block_rows.extend((block_idx, row) for row in read_jsonl(jsonl))

    s1_counts, s2_only_counts = count_attempts(block_rows, n_blocks)
    ratio = [
        (s1_counts[i] / (s1_counts[i] + s2_only_counts[i])) if (s1_counts[i] + s2_only_counts[i]) > 0 else math.nan
        for i in range(n_blocks)
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(1, n_blocks + 1))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([x - width / 2 for x in xs], s1_counts, width=width, color="#1f77b4", label="S1 success")
    ax.bar([x + width / 2 for x in xs], s2_only_counts, width=width, color="#ff7f0e", label="S2 only")
    ax.set_xlabel("Block")
    ax.set_ylabel("Scenario count")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"S1 vs S2-only success by block: {args.config}")
    ax.legend(loc="best")
    fig.tight_layout()

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = suite_dir / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[write] {out}")
    print("s1:", s1_counts)
    print("s2_only:", s2_only_counts)
    print("s1_fraction:", [f"{v:.3f}" if not math.isnan(v) else "nan" for v in ratio])


if __name__ == "__main__":
    main()
