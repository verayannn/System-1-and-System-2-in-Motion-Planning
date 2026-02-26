#!/usr/bin/env python3
"""benchmark_report_maze.py  (Maze benchmark: S1, S2, and S1/S2 hybrids)

Compares five methods across the same maze scenario set:
  - System 1 (S1)
  - System 2 (MPC-only) 
  - System 2 (CBF-only)
  - System 1/2 (MPC fallback)
  - System 1/2 (CBF fallback)

Requested comparison groups:
  1) S1 vs S2(MPC) vs S1/2(MPC)
  2) S1 vs S2(CBF) vs S1/2(CBF)
  3) S2(MPC) vs S2(CBF)
  4) S1/2(MPC) vs S1/2(CBF)

Expected result item format (each list element):
{
  "scenario_id": int,
  "success": bool,
  "runtime_sec": float,
  "collision_free": bool,   # optional
  "goal_reached": bool,     # optional
  ...
}

This script is mildly defensive to schema drift and will try a few common
fallback keys.

Run:
  python maze/benchmark_report_maze.py \
    --scenarios maze/benchmark_scenarios_maze.json \
    --s1 maze/S1_results_maze.json \
    --s2_cbf maze/S2_cbf_results.json \
    --s2_mpc maze/S2_mpc_results.json \
    --s1_s2_cbf maze/S1_S2_cbf_results_maze.json \
    --s1_s2_mpc maze/S1_S2_mpc_results_maze.json \
    --out_md maze/benchmark_report_maze.md \
    --out_csv maze/benchmark_report_maze.csv \
    --out_plot_dir maze/benchmark_report_plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ------------------------------
# Loading + schema helpers
# ------------------------------

_SUCCESS_KEYS = ["success", "is_success", "solved", "ok", "feasible"]
_RUNTIME_KEYS = ["runtime_sec", "runtime", "solve_time", "solver_time", "elapsed", "elapsed_sec", "elapsed_time"]
_COLLISION_KEYS = ["collision_free", "is_collision_free", "collisionFree"]
_GOAL_KEYS = ["goal_reached", "reached_goal", "goalReached"]


def load_results(path: str) -> Dict[int, Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        # allow wrappers
        for k in ("results", "data", "records", "scenarios"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list (or dict wrapper), got {type(data)}")

    out: Dict[int, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        sid = item.get("scenario_id", None)
        if sid is None:
            continue
        out[int(sid)] = item
    return out


def _get_bool(item: Dict[str, Any], keys: List[str]) -> Optional[bool]:
    for k in keys:
        if k not in item:
            continue
        v = item.get(k, None)
        if v is None:
            return None
        return bool(v)
    return None


def _get_float(item: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k not in item:
            continue
        v = item.get(k, None)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return None


# ------------------------------
# Summaries
# ------------------------------

def _rate(xs: List[int]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _avg(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    # nearest-rank
    idx = int(math.ceil(q * len(ys))) - 1
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])


def summarize(name: str, results: Dict[int, Dict[str, Any]], scenario_ids: List[int]) -> Dict[str, Any]:
    succ: List[int] = []
    rt: List[float] = []
    cf: List[int] = []
    gr: List[int] = []

    present = 0
    for sid in scenario_ids:
        item = results.get(sid)
        if item is None:
            continue
        present += 1

        s = _get_bool(item, _SUCCESS_KEYS)
        if s is not None:
            succ.append(1 if s else 0)

        r = _get_float(item, _RUNTIME_KEYS)
        if r is not None:
            rt.append(r)

        c = _get_bool(item, _COLLISION_KEYS)
        if c is not None:
            cf.append(1 if c else 0)

        g = _get_bool(item, _GOAL_KEYS)
        if g is not None:
            gr.append(1 if g else 0)

    return {
        "name": name,
        "n_present": present,
        "success_rate": _rate(succ),
        "collision_free_rate": _rate(cf),
        "goal_reached_rate": _rate(gr),
        "avg_runtime_sec": _avg(rt),
        "median_runtime_sec": _median(rt),
        "p90_runtime_sec": _quantile(rt, 0.90),
        "n_runtime": len(rt),
    }


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.1f}%"


def fmt_f(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


# ------------------------------
# Plotting (hist + box) per group
# ------------------------------

def _runtime_series(results: Dict[int, Dict[str, Any]], scenario_ids: List[int], *, only_success: bool = False) -> List[float]:
    xs: List[float] = []
    for sid in scenario_ids:
        item = results.get(sid)
        if item is None:
            continue
        if only_success:
            s = _get_bool(item, _SUCCESS_KEYS)
            if s is not True:
                continue
        r = _get_float(item, _RUNTIME_KEYS)
        if r is None:
            continue
        xs.append(float(r))
    return xs

def _finite(xs: List[float]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


# ------------------------------
# Plot styling (match "circle" benchmark vibe)
# ------------------------------

PLOT_DPI = 200
FIG_W = 10.5          # not too wide; fits markdown + A4 screenshots well
FIG_H = 4.2
HIST_BINS = 24        # stable bins across groups
ROBUST_Q = 0.95       # cap axis at p95 to avoid one outlier ruining scale
PAD_FRAC = 0.08       # little headroom on axis limits


def _finite(xs: List[float]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


PLOT_DPI = 200
FIG_W = 10.5
FIG_H = 4.2
HIST_BINS = 24

HIST_Q = 0.95        # histogram shows majority behavior
BOX_Q  = 0.995       # box plot keeps outliers visible (near-full range)
PAD_FRAC = 0.08


def _finite(xs: List[float]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def _cap_from_quantile(xs: List[float], q: float) -> float:
    xs = sorted(_finite(xs))
    if not xs:
        return 1.0
    if len(xs) < 10:
        cap = xs[-1]
    else:
        cap = _quantile(xs, q) or xs[-1]
    cap = float(cap)
    cap = max(cap, 1e-6)
    return cap


def _nice_limit(xs: List[float], q: float) -> Tuple[float, float]:
    cap = _cap_from_quantile(xs, q=q)
    hi = cap * (1.0 + PAD_FRAC)
    hi = max(hi, 1e-3)
    return (0.0, hi)


def plot_group_runtime(
    group_title: str,
    methods: List[Tuple[str, Dict[int, Dict[str, Any]]]],
    scenario_ids: List[int],
    out_path: Path,
) -> None:
    """
    One PNG per group:
      - left: histogram with robust scale (HIST_Q)
      - right: boxplot with near-full scale (BOX_Q)
    Scales are computed per-group, so they naturally differ by group.
    """

    series: List[Tuple[str, List[float]]] = []
    all_xs: List[float] = []

    for name, res in methods:
        xs = _finite(_runtime_series(res, scenario_ids, only_success=False))
        series.append((name, xs))
        all_xs.extend(xs)

    # independent limits per panel
    h_x0, h_x1 = _nice_limit(all_xs, q=HIST_Q)
    b_y0, b_y1 = _nice_limit(all_xs, q=BOX_Q)

    # avoid ugly near-zero scaling
    if h_x1 < 1e-2:
        h_x1 = 1e-2
    if b_y1 < 1e-2:
        b_y1 = 1e-2

    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), dpi=PLOT_DPI,
        gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    # ----------------
    # Histogram (left): clip to hist window so bins are meaningful
    # ----------------
    for name, xs in series:
        xs_clip = [x for x in xs if h_x0 <= x <= h_x1]
        if not xs_clip:
            continue
        ax_h.hist(xs_clip, bins=HIST_BINS, range=(h_x0, h_x1), alpha=0.50, label=name)
        ax_h.axvline(float(statistics.median(xs_clip)), linestyle="--", linewidth=1)

    ax_h.set_title(f"Runtime histogram (≤ p{int(HIST_Q*100)})")
    ax_h.set_xlabel("runtime_sec")
    ax_h.set_ylabel("count")
    ax_h.set_xlim(h_x0, h_x1)
    ax_h.grid(True, alpha=0.25)
    ax_h.legend(loc="upper right", frameon=True)

    # ----------------
    # Box plot (right): show outliers; scale is wider
    # ----------------
    labels = [n for n, _ in series]
    data = [xs for _, xs in series]  # keep full data for outliers

    ax_b.boxplot(
        data,
        labels=labels,
        showfliers=True,
        widths=0.55,
        patch_artist=False,
        whis=1.5,
    )
    ax_b.set_title(f"Runtime box plot (≤ p{int(BOX_Q*1000)/10:.1f})")
    ax_b.set_ylabel("runtime_sec")
    ax_b.set_ylim(b_y0, b_y1)
    ax_b.grid(True, axis="y", alpha=0.25)

    fig.suptitle(group_title, y=1.03, fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# Report generation
# ------------------------------

def write_group_section_md(
    lines: List[str],
    group_title: str,
    summaries: List[Dict[str, Any]],
    runtime_plot_path: Optional[Path],
    out_md_path: Path,
) -> None:
    lines.append(f"\n## {group_title}\n")
    lines.append("| Method | N present | Success | Collision-free | Goal-reached | Avg runtime (s) | Median runtime (s) | P90 runtime (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['n_present']} | {fmt_pct(s['success_rate'])} | {fmt_pct(s['collision_free_rate'])} | {fmt_pct(s['goal_reached_rate'])} | "
            f"{fmt_f(s['avg_runtime_sec'])} | {fmt_f(s['median_runtime_sec'])} | {fmt_f(s['p90_runtime_sec'])} |"
        )

    # Embed plot (relative path) if available
    if runtime_plot_path is not None:
        try:
            rel = runtime_plot_path.relative_to(out_md_path.parent)
        except Exception:
            rel = runtime_plot_path
        lines.append("\n**Runtime (histogram + box plot)**")
        lines.append(f"\n![]({rel.as_posix()})\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, default="maze/benchmark_scenarios_maze.json")

    ap.add_argument("--s1", type=str, default="maze/S1_results_maze.json")
    ap.add_argument("--s2_cbf", type=str, default="maze/S2_cbf_results.json")
    ap.add_argument("--s2_mpc", type=str, default="maze/S2_mpc_results.json")
    ap.add_argument("--s1_s2_cbf", type=str, default="maze/S1_S2_cbf_results_maze.json")
    ap.add_argument("--s1_s2_mpc", type=str, default="maze/S1_S2_mpc_results_maze.json")

    ap.add_argument("--out_md", type=str, default="maze/benchmark_report_maze.md")
    ap.add_argument("--out_csv", type=str, default="maze/benchmark_report_maze.csv")
    ap.add_argument("--out_plot_dir", type=str, default="maze/benchmark_report_plots")

    args = ap.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text())
    if not isinstance(scenarios, list):
        raise TypeError("--scenarios must be a JSON list")
    scenario_ids = [int(sc["scenario_id"]) for sc in scenarios]

    # Load all five result bundles
    bundles: List[Tuple[str, str]] = [
        ("S1", args.s1),
        ("S2 (CBF-only)", args.s2_cbf),
        ("S2 (MPC-only)", args.s2_mpc),
        ("S1/2 (CBF)", args.s1_s2_cbf),
        ("S1/2 (MPC)", args.s1_s2_mpc),
    ]

    loaded: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for name, path in bundles:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing: {path}")
        loaded[name] = load_results(path)

    # Define groups in the exact order you asked
    groups: List[Tuple[str, List[str]]] = [
        ("Group 1: S1 vs S2(MPC) vs S1/2(MPC)", ["S1", "S2 (MPC-only)", "S1/2 (MPC)"]),
        ("Group 2: S1 vs S2(CBF) vs S1/2(CBF)", ["S1", "S2 (CBF-only)", "S1/2 (CBF)"]),
        ("Group 3: S2(MPC) vs S2(CBF)", ["S2 (MPC-only)", "S2 (CBF-only)"]),
        ("Group 4: S1/2(MPC) vs S1/2(CBF)", ["S1/2 (MPC)", "S1/2 (CBF)"]),
    ]

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    plot_dir = Path(args.out_plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --------- Markdown report ---------
    md_lines: List[str] = []
    md_lines.append("# Maze Benchmark Report\n")
    md_lines.append(f"- Scenarios: {len(scenario_ids)} ({args.scenarios})\n")
    md_lines.append("This report compares S1 / S2 (CBF, MPC) / S1-S2 hybrids on maze scenarios.\n")

    # --------- Global per-scenario CSV ---------
    # Columns: scenario_id + for each method: success, runtime_sec
    method_order = [name for name, _ in bundles]
    header: List[str] = ["scenario_id"]
    for m in method_order:
        header += [f"{m}:success", f"{m}:rt_sec"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sid in scenario_ids:
            row: List[Any] = [sid]
            for m in method_order:
                item = loaded[m].get(sid, {})
                s = _get_bool(item, _SUCCESS_KEYS)
                r = _get_float(item, _RUNTIME_KEYS)
                row.append("" if s is None else int(bool(s)))
                row.append("" if r is None else float(r))
            w.writerow(row)

    # --------- Group sections + plots ---------
    for gi, (gtitle, methods) in enumerate(groups, start=1):
        summaries = [summarize(m, loaded[m], scenario_ids) for m in methods]

        runtime_plot_path = plot_dir / f"group_{gi}_runtime.png"
        plot_group_runtime(gtitle, [(m, loaded[m]) for m in methods], scenario_ids, runtime_plot_path)

        write_group_section_md(md_lines, gtitle, summaries, runtime_plot_path, out_md)

    # --------- Per-scenario outcomes table (markdown) ---------
    md_lines.append("\n## Per-scenario outcomes\n")
    md_header = ["scenario_id"]
    for m in method_order:
        md_header += [f"{m}:success", f"{m}:rt_sec"]

    md_lines.append("| " + " | ".join(md_header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(md_header)) + "|")

    for sid in scenario_ids:
        row = [str(sid)]
        for m in method_order:
            item = loaded[m].get(sid, {})
            s = _get_bool(item, _SUCCESS_KEYS)
            r = _get_float(item, _RUNTIME_KEYS)
            row.append("NA" if s is None else ("1" if s else "0"))
            row.append("NA" if r is None else f"{float(r):.3f}")
        md_lines.append("| " + " | ".join(row) + " |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines))

    print(f"✅ wrote {out_md}")
    print(f"✅ wrote {out_csv}")
    print(f"✅ plots in {plot_dir}")


if __name__ == "__main__":
    main()
