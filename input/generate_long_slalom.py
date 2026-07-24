#!/usr/bin/env python3
"""Generate nonlinear long-slalom stress cases for standalone S2 tests.

The route is long with many nearby, staggered blocks, but it never requires a
globally hidden wall-gap choice. This makes it suitable for an accuracy-first
CBF controller with a global reference path.

Example:
  python input/generate_long_slalom.py --n_scenarios 20 --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from generate_nl_dict import make_nonlinear_dynamics

Rect = Tuple[float, float, float, float]
BOUNDS = (-30.0, -30.0, 30.0, 30.0)
START = (25.0, 24.0)
GOAL = (-25.0, -24.0)


def _round_rect(rect: Rect) -> List[float]:
    return [round(float(value), 6) for value in rect]


def build_long_slalom(rng: random.Random) -> List[Rect]:
    """Create a long, constrained alternating slalom."""
    rects: List[Rect] = []

    # Ten alternating barrier rows: 18, 14, ..., -18.
    for index, y in enumerate(range(18, -19, -4)):
        side = -1.0 if index % 2 == 0 else 1.0

        # Wider primary barriers force alternating detours.
        center_x = side * rng.uniform(5.5, 7.0)
        width = rng.uniform(12.0, 13.5)
        height = rng.uniform(1.8, 2.2)
        rects.append((
            center_x - width / 2.0,
            y - height / 2.0,
            center_x + width / 2.0,
            y + height / 2.0,
        ))

        # Opposite-side blocker removes an easy straight shortcut.
        offset_x = -side * rng.uniform(13.0, 15.0)
        offset_y = y - rng.uniform(1.5, 2.2)
        rects.append((
            offset_x - 1.5,
            offset_y - 1.2,
            offset_x + 1.5,
            offset_y + 1.2,
        ))

    return rects


def make_scenario(rng: random.Random, scenario_id: int) -> Dict[str, object]:
    # The generic weak-shear regime grows with position and can overpower the
    # bounded control input on this deliberately large map. Keep the stress in
    # the geometry and route length, while using bounded nonlinear drifts.
    dynamics = make_nonlinear_dynamics(rng)
    while dynamics["regime"] == "weak_shear":
        dynamics = make_nonlinear_dynamics(rng)
    rects = build_long_slalom(rng)
    return {
        "scenario_id": scenario_id,
        "rectangles": [_round_rect(rect) for rect in rects],
        "bounds": list(BOUNDS),
        "cell_size": 0.4,
        "start": list(START),
        "goal": list(GOAL),
        "u_max": 3.0,
        "goal_tol": 0.6,
        "difficulty": "stress",
        "difficulty_score": float(len(rects)),
        "map_type": "long_slalom",
        "benchmark_family": "long_slalom",
        "dynamics_type": "nonlinear",
        "dynamics_model": dynamics["model"],
        "regime": dynamics["regime"],
        "nonlinear_dynamics": dynamics,
        "connected_checked": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output_dir", default="input/nl_long_slalom")
    parser.add_argument("--prefix", default="benchmark_dualmp_nl_long_slalom")
    parser.add_argument("--n_scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    scenarios = [make_scenario(rng, scenario_id) for scenario_id in range(args.n_scenarios)]
    path = output_dir / f"{args.prefix}.json"
    path.write_text(json.dumps(scenarios, indent=2))
    print(f"[write] {path} scenarios={len(scenarios)}")


if __name__ == "__main__":
    main()
