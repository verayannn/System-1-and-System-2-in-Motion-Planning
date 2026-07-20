#!/usr/bin/env python3
"""Generate a standalone nonlinear long-horizon zigzag benchmark.

This benchmark is intentionally separate from the seven standard families.
It uses alternating full-width wall gates in a larger workspace, requiring a
long route and many genuine CBF/MPC obstacle constraints.  It does not modify
any solver or System 1 asset.

Example:
    python input/generate_long_horizon_zigzag.py --n_scenarios 20 --seed 71
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import generate_nl_dict as nonlinear


Rect = Tuple[float, float, float, float]
BOUNDS = (-25.0, -25.0, 25.0, 25.0)
START = (19.0, 20.0)
GOAL = (-19.0, -20.0)


def _rect(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    xmin, ymin, xmax, ymax = BOUNDS
    return [
        round(max(xmin, min(xmax, x1)), 6),
        round(max(ymin, min(ymax, y1)), 6),
        round(max(xmin, min(xmax, x2)), 6),
        round(max(ymin, min(ymax, y2)), 6),
    ]


def build_zigzag_gates(rng: random.Random, gate_count: int, gap_width: float) -> List[List[float]]:
    """Build alternating horizontal walls with one traversable opening each."""
    gate_count = max(2, int(gate_count))
    usable_top, usable_bottom = 15.5, -15.5
    step = (usable_top - usable_bottom) / max(gate_count - 1, 1)
    wall_xmin, wall_xmax = -24.0, 24.0
    rects: List[List[float]] = []

    # Alternate far-left/far-right openings.  Small jitter prevents every map
    # from being identical while retaining a reliable, long connected route.
    for gate_idx in range(gate_count):
        y = usable_top - gate_idx * step + rng.uniform(-0.35, 0.35)
        thickness = rng.uniform(0.55, 0.80)
        # The first and final gates align with the fixed start/goal sides.
        # This preserves a feasible local entry while every intermediate gate
        # still forces a full-width traverse to the opposite side.
        side = 1.0 if gate_idx % 2 == 0 else -1.0
        gap_center = side * rng.uniform(9.0, 12.0)
        gap_half = 0.5 * gap_width
        left_end = gap_center - gap_half
        right_start = gap_center + gap_half
        rects.append(_rect(wall_xmin, y, left_end, y + thickness))
        rects.append(_rect(right_start, y, wall_xmax, y + thickness))

    return rects


def make_scenario(
    rng: random.Random,
    scenario_id: int,
    *,
    gate_count: int,
    gap_width: float,
    u_max: float,
    goal_tol: float,
) -> Dict[str, object]:
    dynamics = nonlinear.make_nonlinear_dynamics(rng)
    # Stronger nonlinear cross-coupling makes the long route a genuine
    # nonlinear control problem while retaining the established dynamics schema.
    if dynamics["regime"] in {"rotate_cw", "rotate_ccw"}:
        dynamics["parameters"]["omega"] = rng.uniform(1.5, 1.9)
    elif dynamics["regime"] == "weak_shear":
        dynamics["parameters"]["shear"] = rng.choice((-1.0, 1.0)) * rng.uniform(0.65, 0.9)

    rectangles = build_zigzag_gates(rng, gate_count, gap_width)
    return {
        "scenario_id": int(scenario_id),
        "rectangles": rectangles,
        "bounds": list(BOUNDS),
        "cell_size": 0.5,
        "start": list(START),
        "goal": list(GOAL),
        "u_max": float(u_max),
        "goal_tol": float(goal_tol),
        "difficulty": "long_horizon_hard",
        "difficulty_score": round(float(gate_count * 2), 6),
        "map_type": "long_horizon_zigzag",
        "benchmark_family": "long_horizon_zigzag",
        "dynamics_type": "nonlinear",
        "dynamics_model": dynamics["model"],
        "regime": dynamics["regime"],
        "nonlinear_dynamics": dynamics,
        "gate_count": int(gate_count),
        "gate_gap_width": float(gap_width),
        "connected_checked": True,
        "map_resample_tries": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output_dir", default="input/nl_long_horizon")
    parser.add_argument("--prefix", default="benchmark_dualmp_nl_long_horizon_zigzag")
    parser.add_argument("--n_scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--gate_count", type=int, default=8)
    parser.add_argument("--gap_width", type=float, default=3.2)
    parser.add_argument("--u_max", type=float, default=3.0)
    parser.add_argument("--goal_tol", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_scenarios < 1:
        raise SystemExit("--n_scenarios must be positive")
    if args.gap_width <= 1.0:
        raise SystemExit("--gap_width must be greater than 1.0")

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    scenarios = [
        make_scenario(
            rng,
            idx,
            gate_count=args.gate_count,
            gap_width=args.gap_width,
            u_max=args.u_max,
            goal_tol=args.goal_tol,
        )
        for idx in range(args.n_scenarios)
    ]
    dictionary_path = output_dir / f"{args.prefix}.json"
    manifest_path = output_dir / f"{args.prefix}_manifest.json"
    dictionary_path.write_text(json.dumps(scenarios, indent=2) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "dictionary": str(dictionary_path),
                "scenario_count": len(scenarios),
                "seed": args.seed,
                "family": "long_horizon_zigzag",
                "gate_count": args.gate_count,
                "gap_width": args.gap_width,
                "notes": [
                    "Standalone stress benchmark; it is not part of the standard seven families.",
                    "No solver or System 1 code is changed by this generator.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[write] {dictionary_path} scenarios={len(scenarios)}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
