# =========================
# benchmark_report.py
# =========================
"""
benchmark_report.py — Summarize & visualize S1-only vs S2-only vs S1+S2 benchmark results.

Supports TWO input styles:

(A) One combined JSON from S1_S2_usage.py (recommended):
    [
      {
        "scenario_id": ...,
        "S1_only": {"success": bool, "cost": float|None, "runtime_sec": float|None, ...},
        "S2_only": {"success": bool, "cost": float|None, "runtime_sec": float|None, ...},
        "S1_S2":   {"success": bool, "cost": float|None, "runtime_sec": float|None, ...},
        ...
      },
      ...
    ]

(B) Three separate JSON files (your request):
    - S1_only_results.json
    - S2_results.json
    - S1_S2_results.json

Each of the three files can be either:
  - a list of dicts with keys: scenario_id, success, cost, runtime_sec
  - OR a dict with a "results" list containing such entries
  - OR a list in the combined format above (we'll try to detect)

Outputs:
  - prints a compact table: N, solved, success rate, mean/median cost, mean/median runtime
  - saves a CSV summary (optional)
  - boxplots for cost & runtime (optional)
  - histogram for cost & runtime (optional)

Dependencies:
  - numpy
  - matplotlib

usage:

python circle/benchmark_report.py \
  --s1_json circle/S1_results.json \
  --s2_json circle/S2_results.json \
  --s1s2_json circle/S1_S2_results.json \
  --save_prefix bench \
  --save_csv benchmark_summary.csv


"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- figure sizing (consistent across all plots) ---
FIGSIZE = (6.0, 4.2)   # pick what you like; same used for all figures
DPI = 200



# ============================================================
# Data model
# ============================================================

@dataclass
class MethodMetrics:
    scenario_id: int
    success: Optional[bool]
    cost: Optional[float]
    runtime_sec: Optional[float]


# ============================================================
# Robust loading/parsing
# ============================================================

def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _as_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    # fallback: try to interpret dict as one entry
    return [obj]


def _parse_entry_like_method_metrics(entry: Dict[str, Any]) -> Optional[MethodMetrics]:
    """
    Try to parse an entry that already looks like:
      {"scenario_id": 0, "success": true, "cost": 123.4, "runtime_sec": 0.56}
    """
    if not isinstance(entry, dict):
        return None
    if "scenario_id" not in entry:
        return None
    if not any(k in entry for k in ("success", "cost", "runtime_sec")):
        return None

    sid = int(entry["scenario_id"])
    success = entry.get("success", None)
    if success is not None:
        success = bool(success)

    cost = entry.get("cost", None)
    cost = None if cost is None else float(cost)

    rt = entry.get("runtime_sec", None)
    rt = None if rt is None else float(rt)

    return MethodMetrics(scenario_id=sid, success=success, cost=cost, runtime_sec=rt)


def _parse_combined_format(items: List[Dict[str, Any]]) -> Dict[str, List[MethodMetrics]]:
    """
    Parse combined-format list returned by S1_S2_usage.py:
      item["S1_only"], item["S2_only"], item["S1_S2"]
    """
    out: Dict[str, List[MethodMetrics]] = {"S1_only": [], "S2_only": [], "S1_S2": []}
    for it in items:
        if not isinstance(it, dict):
            continue
        if not all(k in it for k in ("scenario_id", "S1_only", "S2_only", "S1_S2")):
            continue
        sid = int(it["scenario_id"])
        for method_key in ("S1_only", "S2_only", "S1_S2"):
            m = it.get(method_key, {})
            if not isinstance(m, dict):
                m = {}
            out[method_key].append(
                MethodMetrics(
                    scenario_id=sid,
                    success=None if m.get("success", None) is None else bool(m["success"]),
                    cost=None if m.get("cost", None) is None else float(m["cost"]),
                    runtime_sec=None if m.get("runtime_sec", None) is None else float(m["runtime_sec"]),
                )
            )
    # If it parsed nothing, return empty dict to indicate failure
    if sum(len(v) for v in out.values()) == 0:
        return {}
    return out


def load_methods_from_files(
    *,
    combined_json: Optional[str] = None,
    s1_json: Optional[str] = None,
    s2_json: Optional[str] = None,
    s1s2_json: Optional[str] = None,
) -> Dict[str, List[MethodMetrics]]:
    """
    Returns dict:
      {"S1_only": [...], "S2_only": [...], "S1_S2": [...]}

    You can provide either:
      - combined_json (preferred), OR
      - (s1_json, s2_json, s1s2_json)
    """
    if combined_json is not None:
        raw = _read_json(combined_json)
        items = _as_list(raw)
        parsed = _parse_combined_format(items)
        if not parsed:
            raise ValueError(
                f"Could not parse combined JSON format from: {combined_json}\n"
                "Expected a list of dicts with keys: scenario_id, S1_only, S2_only, S1_S2."
            )
        return parsed

    if not (s1_json and s2_json and s1s2_json):
        raise ValueError("Provide either --combined_json OR all of --s1_json --s2_json --s1s2_json")

    def load_one(path: str) -> List[MethodMetrics]:
        raw = _read_json(path)
        items = _as_list(raw)

        # If someone accidentally passes combined file here, try to parse and take the matching key:
        maybe_combined = _parse_combined_format(items if isinstance(items, list) else [])
        if maybe_combined:
            # pick first key that matches this filename hint
            return []  # let caller handle; we won't guess here

        out: List[MethodMetrics] = []
        for it in items:
            mm = _parse_entry_like_method_metrics(it) if isinstance(it, dict) else None
            if mm is not None:
                out.append(mm)
        if len(out) == 0:
            raise ValueError(
                f"Could not parse method-result JSON from: {path}\n"
                "Expected entries like: {'scenario_id':..., 'success':..., 'cost':..., 'runtime_sec':...}"
            )
        return out

    return {
        "S1_only": load_one(s1_json),
        "S2_only": load_one(s2_json),
        "S1_S2": load_one(s1s2_json),
    }


# ============================================================
# Summaries
# ============================================================

def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def summarize_method(method: List[MethodMetrics]) -> Dict[str, Any]:
    scenario_ids = np.array([m.scenario_id for m in method], dtype=int)

    succ_raw = np.array([np.nan if m.success is None else float(bool(m.success)) for m in method], dtype=float)
    cost_raw = np.array([np.nan if m.cost is None else float(m.cost) for m in method], dtype=float)
    rt_raw   = np.array([np.nan if m.runtime_sec is None else float(m.runtime_sec) for m in method], dtype=float)

    valid = _finite(cost_raw) & _finite(rt_raw)  # ran to completion

    # success rate among valid runs
    succ_valid = succ_raw[valid]
    sr = float(np.nanmean(succ_valid)) if succ_valid.size > 0 else 0.0

    out = {
        "n_total": int(len(method)),
        "n_valid": int(np.sum(valid)),
        "n_invalid": int(len(method) - int(np.sum(valid))),
        "success_rate_valid": sr,
        "cost_mean": float(np.nanmean(cost_raw[valid])) if np.any(valid) else float("nan"),
        "cost_median": float(np.nanmedian(cost_raw[valid])) if np.any(valid) else float("nan"),
        "runtime_mean": float(np.nanmean(rt_raw[valid])) if np.any(valid) else float("nan"),
        "runtime_median": float(np.nanmedian(rt_raw[valid])) if np.any(valid) else float("nan"),
        "scenario_ids": scenario_ids,
        "cost": cost_raw,
        "runtime_sec": rt_raw,
        "success": succ_raw,
        "valid_mask": valid,
    }
    return out


def print_summary_table(summaries: Dict[str, Dict[str, Any]]) -> None:
    # simple aligned printing (no pandas needed)
    header = (
        f"{'Method':<10}  {'N':>6}  {'Valid':>6}  {'Succ%':>7}  "
        f"{'Cost(mean)':>11}  {'Cost(med)':>10}  {'RT(mean)':>10}  {'RT(med)':>9}"
    )
    print(header)
    print("-" * len(header))
    for name in ("S1_only", "S2_only", "S1_S2"):
        s = summaries[name]
        succ_pct = 100.0 * float(s["success_rate_valid"]) if s["n_valid"] > 0 else 0.0
        print(
            f"{name:<10}  "
            f"{s['n_total']:>6d}  {s['n_valid']:>6d}  {succ_pct:>6.1f}%  "
            f"{s['cost_mean']:>11.3f}  {s['cost_median']:>10.3f}  "
            f"{s['runtime_mean']:>10.3f}  {s['runtime_median']:>9.3f}"
        )

# ============================================================
# Plots (no seaborn, no manual colors) — UPDATED
#   - cost plots use finite(cost)
#   - runtime plots use finite(runtime)
#   - do NOT require both to be present
# ============================================================

def _extract_metric_values(summary: Dict[str, Any], key: str) -> np.ndarray:
    """Return finite values for a single metric (cost OR runtime_sec)."""
    vals = np.array(summary[key], dtype=float)
    return vals[np.isfinite(vals)]


def plot_boxplots(
    summaries: Dict[str, Dict[str, Any]],
    *,
    save_prefix: Optional[str] = None,
    show: bool = True,
):
    labels = ["S1_only", "S2_only", "S1_S2"]

    # --- cost boxplot: uses finite(cost) only ---
    cost_data = [_extract_metric_values(summaries[k], "cost") for k in labels]
    plt.figure(figsize=FIGSIZE)
    plt.boxplot(cost_data, labels=labels, showfliers=True)
    plt.grid(True, alpha=0.3)
    plt.title("Cost (finite cost entries)")
    plt.ylabel("cost")
    if save_prefix:
        plt.savefig(f"{save_prefix}_box_cost.png", dpi=200, bbox_inches="tight")

    # --- runtime boxplot: uses finite(runtime) only ---
    rt_data = [_extract_metric_values(summaries[k], "runtime_sec") for k in labels]
    plt.figure(figsize=FIGSIZE)
    plt.boxplot(rt_data, labels=labels, showfliers=True)
    plt.grid(True, alpha=0.3)
    plt.title("Runtime (finite runtime entries)")
    plt.ylabel("runtime_sec")
    if save_prefix:
        plt.savefig(f"{save_prefix}_box_runtime.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_histograms(
    summaries: Dict[str, Dict[str, Any]],
    *,
    bins: int = 30,
    save_prefix: Optional[str] = None,
    show: bool = True,
):
    labels = ["S1_only", "S2_only", "S1_S2"]

    # --- cost histogram: uses finite(cost) only ---
    plt.figure(figsize=FIGSIZE)
    for k in labels:
        vals = _extract_metric_values(summaries[k], "cost")
        print(f"[DEBUG] {k}: finite costs = {vals.size}")
        if vals.size > 0:
            print(f"[DEBUG] {k}: cost min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}")
            plt.hist(vals, bins=bins, alpha=0.5, label=k)
            plt.axvline(vals.mean(), linewidth=1, linestyle="--")  # mean marker

    plt.grid(True, alpha=0.3)
    plt.title("Cost histogram (finite cost entries)")
    plt.xlabel("cost")
    plt.ylabel("count")
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_hist_cost.png", dpi=200, bbox_inches="tight")

    # --- runtime histogram: uses finite(runtime) only ---
    plt.figure(figsize=FIGSIZE)
    for k in labels:
        vals = _extract_metric_values(summaries[k], "runtime_sec")
        print(f"[DEBUG] {k}: finite runtimes = {vals.size}")
        if vals.size > 0:
            print(f"[DEBUG] {k}: rt min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}")
            plt.hist(vals, bins=bins, alpha=0.5, label=k)
            plt.axvline(vals.mean(), linewidth=1, linestyle="--")  # mean marker

    plt.grid(True, alpha=0.3)
    plt.title("Runtime histogram (finite runtime entries)")
    plt.xlabel("runtime_sec")
    plt.ylabel("count")
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_hist_runtime.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")



# ============================================================
# Optional CSV output
# ============================================================

def save_summary_csv(summaries: Dict[str, Dict[str, Any]], path: str) -> None:
    headers = [
        "method",
        "n_total",
        "n_valid",
        "success_rate_valid",
        "cost_mean",
        "cost_median",
        "runtime_mean",
        "runtime_median",
    ]
    lines = [",".join(headers)]
    for name in ("S1_only", "S2_only", "S1_S2"):
        s = summaries[name]
        lines.append(
            ",".join([
                name,
                str(s["n_total"]),
                str(s["n_valid"]),
                f"{float(s['success_rate_valid']):.6f}",
                f"{float(s['cost_mean']):.6f}",
                f"{float(s['cost_median']):.6f}",
                f"{float(s['runtime_mean']):.6f}",
                f"{float(s['runtime_median']):.6f}",
            ])
        )
    Path(path).write_text("\n".join(lines))
    print(f"Saved CSV summary -> {path}")


# ============================================================
# CLI
# ============================================================
def _ensure_dir_prefix(path_or_prefix: Optional[str], folder: str = "circle") -> Optional[str]:
    """
    If user passes 'bench', return 'circle/bench'.
    If user passes 'circle/bench' or '/abs/path/bench', keep it.
    """
    if path_or_prefix is None:
        return None
    p = Path(path_or_prefix)
    if p.is_absolute():
        return str(p)
    # already under circle/
    if len(p.parts) > 0 and p.parts[0] == folder:
        return str(p)
    return str(Path(folder) / p)



def main():
    p = argparse.ArgumentParser()

    # input mode A
    p.add_argument("--combined_json", type=str, default=None,
                   help="One JSON containing S1_only/S2_only/S1_S2 per scenario (from S1_S2_usage.py).")

    # input mode B
    p.add_argument("--s1_json", type=str, default=None, help="S1-only results JSON.")
    p.add_argument("--s2_json", type=str, default=None, help="S2-only results JSON.")
    p.add_argument("--s1s2_json", type=str, default=None, help="S1+S2 results JSON.")

    # output
    p.add_argument("--save_csv", type=str, default=None, help="Save a one-line-per-method CSV summary to this path.")
    p.add_argument("--save_prefix", type=str, default=None, help="If set, save plots as <prefix>_*.png.")
    p.add_argument("--no_show", action="store_true", help="Do not show plots (still can save).")

    # plots
    p.add_argument("--no_boxplots", action="store_true")
    p.add_argument("--no_hists", action="store_true")
    p.add_argument("--bins", type=int, default=30)

    args = p.parse_args()

    # force outputs under circle/
    args.save_prefix = _ensure_dir_prefix(args.save_prefix, folder="circle")
    args.save_csv    = _ensure_dir_prefix(args.save_csv,    folder="circle")

    methods = load_methods_from_files(
        combined_json=args.combined_json,
        s1_json=args.s1_json,
        s2_json=args.s2_json,
        s1s2_json=args.s1s2_json,
    )

    summaries = {k: summarize_method(v) for k, v in methods.items()}

    print("\n=== Benchmark Summary ===")
    print_summary_table(summaries)

    if args.save_csv:
        save_summary_csv(summaries, args.save_csv)

    show = not args.no_show
    if not args.no_boxplots:
        plot_boxplots(summaries, save_prefix=args.save_prefix, show=show)

    if not args.no_hists:
        plot_histograms(summaries, bins=int(args.bins), save_prefix=args.save_prefix, show=show)


if __name__ == "__main__":
    main()




