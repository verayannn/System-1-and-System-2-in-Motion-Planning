#!/usr/bin/env python3
"""Compare strict-SOFAI MPC runs with and without S1 warm starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(suite_dir: Path, config: str) -> list[dict]:
    manifest = json.loads((suite_dir / "suite_manifest.json").read_text())
    output = []
    for run in manifest["configs"][config]["runs"]:
        prefix = run["prefix"]
        path = suite_dir / config / "runs" / f"{prefix}_runs.jsonl"
        output.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return output


def summary(label: str, data: list[dict]) -> dict[str, object]:
    planning = [float(row["planning_runtime_sec"]) for row in data]
    quality = [float(row["quality_score"]) for row in data if row.get("success") and row.get("quality_score") is not None]
    s2 = [next((a for a in row.get("attempts", []) if a.get("system") == "s2"), None) for row in data]
    s2 = [attempt for attempt in s2 if attempt is not None]
    used = [attempt for attempt in s2 if (attempt.get("mpc_s1_warm_start") or {}).get("used")]
    return {
        "run": label,
        "scenarios": len(data),
        "success_rate": sum(bool(row.get("success")) for row in data) / max(len(data), 1),
        "mean_planning_sec": float(np.mean(planning)) if planning else float("nan"),
        "p90_planning_sec": float(np.percentile(planning, 90)) if planning else float("nan"),
        "mean_quality": float(np.mean(quality)) if quality else float("nan"),
        "s2_fallbacks": len(s2),
        "s2_success_rate": sum(bool(attempt.get("success")) for attempt in s2) / max(len(s2), 1),
        "mean_s2_sec": float(np.mean([float(attempt["runtime_sec"]) for attempt in s2])) if s2 else float("nan"),
        "warm_start_used": len(used),
        "warm_start_use_rate": len(used) / max(len(s2), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--guided", required=True, type=Path)
    parser.add_argument("--config", default="sofai_mpc_cl")
    args = parser.parse_args()

    records = [
        summary("baseline", rows(args.baseline.resolve(), args.config)),
        summary("s1_warm_start", rows(args.guided.resolve(), args.config)),
    ]
    print("| run | scenarios | success | mean planning (s) | p90 planning (s) | mean Q | S2 fallbacks | S2 success | mean S2 (s) | warm starts used |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for record in records:
        print(
            "| {run} | {scenarios} | {success_rate:.3f} | {mean_planning_sec:.3f} | "
            "{p90_planning_sec:.3f} | {mean_quality:.3f} | {s2_fallbacks} | {s2_success_rate:.3f} | {mean_s2_sec:.3f} | "
            "{warm_start_used} ({warm_start_use_rate:.3f}) |".format(**record)
        )


if __name__ == "__main__":
    main()
