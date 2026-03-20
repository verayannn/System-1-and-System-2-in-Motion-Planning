"""
benchmark_report.py — Final Performance Analysis

Aggregates and compares:
1. S1 (Fast Retrieval)
2. S2 (MATLAB MPC-CBF Expert)
3. Hybrid S1+S2 (Retrieval with Expert Fallback)

Outputs:
- Console Summary Table
- Boxplots (Cost & Runtime)
- Histograms (Cost & Runtime distribution)

python circle/benchmark_report.py \
  --s1_json circle/S1_results.json \
  --s2_json circle/S2_MPCDC_results.json \
  --s1s2_json circle/S1_S2_results.json \
  --save_prefix circle/final_report

"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Configuration for high-quality plots
FIGSIZE = (10.0, 6.0)
DPI = 200

@dataclass
class MethodMetrics:
    scenario_id: int
    success: bool
    cost: float
    runtime_sec: float

# ============================================================
# 1) Data Loading & Parsing
# ============================================================

def load_json_results(path: str) -> List[MethodMetrics]:
    p = Path(path)
    if not p.exists():
        print(f"⚠️  Warning: File {path} not found.")
        return []
    
    raw = json.loads(p.read_text())
    items = raw if isinstance(raw, list) else raw.get("results", [])
    
    parsed = []
    for it in items:
        sid = it.get("scenario_id")
        # Ensure we treat None or missing as False/NaN
        success = bool(it.get("success", False))
        cost = it.get("cost")
        rt = it.get("runtime_sec")
        
        if sid is not None:
            parsed.append(MethodMetrics(
                scenario_id=int(sid),
                success=success,
                cost=float(cost) if cost is not None else np.nan,
                runtime_sec=float(rt) if rt is not None else np.nan
            ))
    return parsed

def get_stats(metrics: List[MethodMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {}
    
    # Filter for finite values for math
    costs = np.array([m.cost for m in metrics if np.isfinite(m.cost)])
    runtimes = np.array([m.runtime_sec for m in metrics if np.isfinite(m.runtime_sec)])
    successes = [m.success for m in metrics]
    
    return {
        "n_total": len(metrics),
        "success_rate": (sum(successes) / len(metrics)) * 100 if len(metrics) > 0 else 0,
        "cost_mean": np.mean(costs) if costs.size > 0 else 0,
        "cost_median": np.median(costs) if costs.size > 0 else 0,
        "rt_mean": np.mean(runtimes) if runtimes.size > 0 else 0,
        "rt_median": np.median(runtimes) if runtimes.size > 0 else 0,
        "costs": costs,
        "runtimes": runtimes
    }

# ============================================================
# 2) Plotting Functions
# ============================================================

def plot_boxplots(summaries: Dict[str, Dict], save_prefix: Optional[str]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    labels = list(summaries.keys())
    
    # Cost Boxplot
    ax1.boxplot([s["costs"] for s in summaries.values()], labels=labels)
    ax1.set_title("Path Cost (Control Effort)")
    ax1.set_ylabel("Total Cost")
    ax1.grid(True, alpha=0.3)

    # Runtime Boxplot
    ax2.boxplot([s["runtimes"] for s in summaries.values()], labels=labels)
    ax2.set_title("Computation Runtime")
    ax2.set_ylabel("Seconds (Log Scale)")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_boxplots.png", dpi=DPI)

def plot_histograms(summaries: Dict[str, Dict], bins: int, save_prefix: Optional[str]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    
    for label, s in summaries.items():
        if s["costs"].size > 0:
            ax1.hist(s["costs"], bins=bins, alpha=0.5, label=label)
            ax1.axvline(s["cost_mean"], linestyle='--', alpha=0.7)
            
        if s["runtimes"].size > 0:
            ax2.hist(s["runtimes"], bins=bins, alpha=0.5, label=label)
            ax2.axvline(s["rt_mean"], linestyle='--', alpha=0.7)

    ax1.set_title("Cost Distribution")
    ax1.set_xlabel("Cost")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.set_title("Runtime Distribution")
    ax2.set_xlabel("Seconds")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_histograms.png", dpi=DPI)

# ============================================================
# 3) Main Runner
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s1_json", type=str, default="circle/S1_results.json")
    p.add_argument("--s2_json", type=str, default="circle/S2_MPCDC_results.json") ###mpccbf
    p.add_argument("--s1s2_json", type=str, default="circle/S1_S2_results.json")
    p.add_argument("--save_prefix", type=str, default="circle/benchmark_report")
    p.add_argument("--bins", type=int, default=25)
    p.add_argument("--no_show", action="store_true")
    args = p.parse_args()

    # Load and process
    data_map = {
        "S1 (Retrieval)": load_json_results(args.s1_json),
        "S2 (DC)": load_json_results(args.s2_json), ## expert: mpccbf
        "Hybrid (S1+S2)": load_json_results(args.s1s2_json)
    }
    
    summaries = {name: get_stats(m) for name, m in data_map.items() if m}

    # Print Summary Table
    print("\n" + "="*85)
    header = f"{'Method':<20} | {'Succ %':>8} | {'Cost (Mean)':>12} | {'RT (Mean)':>12} | {'RT (Med)':>10}"
    print(header)
    print("-" * 85)
    for name, s in summaries.items():
        print(f"{name:<20} | {s['success_rate']:>7.1f}% | {s['cost_mean']:>12.2f} | {s['rt_mean']:>11.4f}s | {s['rt_median']:>9.4f}s")
    print("="*85 + "\n")

    

    # Generate Plots
    if summaries:
        plot_boxplots(summaries, args.save_prefix)
        plot_histograms(summaries, args.bins, args.save_prefix)
        
        if not args.no_show:
            plt.show()
    else:
        print("❌ No data found to plot. Check your file paths.")

if __name__ == "__main__":
    main()