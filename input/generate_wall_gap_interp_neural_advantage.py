#!/usr/bin/env python3
"""
Generate a held-out wall-gap interpolation benchmark that slightly favors
neural S1 over primitive trajectory retrieval on the current wall_gap assets.

Why this family:
- It stays close to the wall-gap topology used by the repo assets.
- It shifts gap geometry continuously instead of reusing the exact stored maps.
- It adds a short tab near the gap, which makes exact trajectory replay more
  brittle while leaving the local maneuver class recognizable to the neural S1.

python3 input/generate_wall_gap_interp_neural_advantage.py \
  --count 100 \
  --out benchmark_dualmp_wall_gap_interp_neural_advantage.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List


BOUNDS = (-10.0, -10.0, 10.0, 10.0)
START = (5.0, 5.0)
GOAL = (0.0, 0.0)
DEFAULT_B = [[1.0, 0.0], [0.0, 1.0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="number of scenarios")
    parser.add_argument("--seed", type=int, default=20260506, help="random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_dualmp_wall_gap_interp_neural_advantage.json"),
        help="output json path",
    )
    return parser.parse_args()


def clip_rect(rect: List[float]) -> List[float]:
    x1, y1, x2, y2 = rect
    xmin, ymin, xmax, ymax = BOUNDS
    x1 = max(xmin, min(x1, xmax))
    x2 = max(xmin, min(x2, xmax))
    y1 = max(ymin, min(y1, ymax))
    y2 = max(ymin, min(y2, ymax))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def make_stable_a(rng: random.Random) -> List[List[float]]:
    while True:
        a = rng.uniform(-1.0, 1.0)
        b = rng.uniform(-1.0, 1.0)
        c = rng.uniform(-1.0, 1.0)
        d = rng.uniform(-1.0, 1.0)
        tr = a + d
        det = a * d - b * c
        disc = tr * tr - 4.0 * det
        if disc >= 0.0:
            s = math.sqrt(disc)
            eigs = ((tr + s) / 2.0, (tr - s) / 2.0)
            spectral_rightmost = max(eigs)
        else:
            spectral_rightmost = math.sqrt(det) if det > 0.0 else 999.0
        if spectral_rightmost < -0.1:
            return [
                [round(a, 8), round(b, 8)],
                [round(c, 8), round(d, 8)],
            ]


def build_case(rng: random.Random, scenario_id: int) -> Dict[str, object]:
    wall_y = rng.uniform(1.70, 2.15)
    gap_center = rng.uniform(2.20, 3.10)
    gap_width = rng.uniform(2.20, 2.90)
    thickness = rng.uniform(0.32, 0.42)

    gap_x1 = gap_center - gap_width / 2.0
    gap_x2 = gap_center + gap_width / 2.0

    rectangles = [
        clip_rect([-9.2, wall_y, gap_x1, wall_y + thickness]),
        clip_rect([gap_x2, wall_y, 9.2, wall_y + thickness]),
    ]

    tab_width = rng.uniform(0.18, 0.30)
    tab_height = rng.uniform(0.50, 0.80)
    tab_on_left = rng.random() < 0.5
    tab_x2 = gap_x1 if tab_on_left else gap_x2
    tab_x1 = tab_x2 - tab_width if tab_on_left else tab_x2
    rectangles.append(clip_rect([tab_x1, wall_y - tab_height, tab_x2, wall_y]))

    return {
        "scenario_id": scenario_id,
        "A_query": make_stable_a(rng),
        "B_query": DEFAULT_B,
        "rectangles": rectangles,
        "bounds": list(BOUNDS),
        "cell_size": 0.5,
        "start": list(START),
        "goal": list(GOAL),
        "u_max": 3.0,
        "goal_tol": 0.5,
        "difficulty": "interp_medium",
        "difficulty_score": 3,
        "map_type": "wall_gap_interp_neural_advantage",
        "benchmark_family": "wall_gap_interp_neural_advantage",
        "regime": "weak_shear",
        "B_mode": "identity",
        "connected_checked": False,
        "map_resample_tries": 1,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    scenarios = [build_case(rng, i) for i in range(args.count)]
    args.out.write_text(json.dumps(scenarios, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
