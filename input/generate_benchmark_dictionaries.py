#!/usr/bin/env python3
"""Generate solver-ready DualMP benchmark dictionaries for this SOFAI folder.

The current motion_planning_solver.py expects JSON files in input/ with a list
of 2D scenarios. Each scenario must provide A_query, B_query, rectangles,
bounds, start, and goal. This generator writes one JSON file per benchmark
family so results can be reported separately by environment type.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]


BOUNDS: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0)
START: Point = (5.0, 5.0)
GOAL: Point = (0.0, 0.0)
DEFAULT_B: List[List[float]] = [[1.0, 0.0], [0.0, 1.0]]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    difficulty: str
    builder: Callable[[random.Random, int], List[Rect]]


def _round_rect(rect: Rect) -> List[float]:
    return [round(float(v), 6) for v in rect]


def _rect_area(rect: Rect) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _point_in_rect(point: Point, rect: Rect, margin: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _overlaps_any(rect: Rect, rects: Sequence[Rect], margin: float = 0.1) -> bool:
    x1, y1, x2, y2 = rect
    for ox1, oy1, ox2, oy2 in rects:
        if not (x2 + margin < ox1 or ox2 + margin < x1 or y2 + margin < oy1 or oy2 + margin < y1):
            return True
    return False


def _clip_rect(rect: Rect, bounds: Tuple[float, float, float, float] = BOUNDS) -> Rect:
    xmin, ymin, xmax, ymax = bounds
    x1, y1, x2, y2 = rect
    x1 = max(xmin + 0.2, min(xmax - 0.2, x1))
    x2 = max(xmin + 0.2, min(xmax - 0.2, x2))
    y1 = max(ymin + 0.2, min(ymax - 0.2, y1))
    y2 = max(ymin + 0.2, min(ymax - 0.2, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _random_box(
    rng: random.Random,
    *,
    x_range: Tuple[float, float] = (-8.8, 8.8),
    y_range: Tuple[float, float] = (-8.8, 8.8),
    w_range: Tuple[float, float] = (0.5, 2.0),
    h_range: Tuple[float, float] = (0.5, 2.0),
) -> Rect:
    w = rng.uniform(*w_range)
    h = rng.uniform(*h_range)
    x1 = rng.uniform(x_range[0], x_range[1] - w)
    y1 = rng.uniform(y_range[0], y_range[1] - h)
    return _clip_rect((x1, y1, x1 + w, y1 + h))


def _sample_clutter(
    rng: random.Random,
    count: int,
    *,
    w_range: Tuple[float, float],
    h_range: Tuple[float, float],
    protect: Sequence[Point] = (START, GOAL),
) -> List[Rect]:
    rects: List[Rect] = []
    tries = 0
    while len(rects) < count and tries < 500:
        tries += 1
        rect = _random_box(rng, w_range=w_range, h_range=h_range)
        if any(_point_in_rect(p, rect, margin=0.9) for p in protect):
            continue
        if _overlaps_any(rect, rects, margin=0.25):
            continue
        rects.append(rect)
    return rects


def build_small_open(rng: random.Random, _: int) -> List[Rect]:
    return _sample_clutter(
        rng,
        rng.randint(0, 2),
        w_range=(0.5, 1.2),
        h_range=(0.5, 1.2),
    )


def build_large_sparse(rng: random.Random, _: int) -> List[Rect]:
    return _sample_clutter(
        rng,
        rng.randint(3, 5),
        w_range=(1.2, 2.8),
        h_range=(1.2, 2.8),
    )


def build_dense_clutter(rng: random.Random, _: int) -> List[Rect]:
    return _sample_clutter(
        rng,
        rng.randint(9, 14),
        w_range=(0.7, 1.8),
        h_range=(0.7, 1.8),
    )


def build_wall_gap(rng: random.Random, _: int) -> List[Rect]:
    y = rng.uniform(-3.0, 2.0)
    thickness = rng.uniform(0.35, 0.75)
    gap_center = rng.uniform(-3.0, 3.0)
    gap_width = rng.uniform(1.5, 3.3)
    left_end = gap_center - gap_width / 2.0
    right_start = gap_center + gap_width / 2.0
    return [
        _clip_rect((-9.2, y, left_end, y + thickness)),
        _clip_rect((right_start, y, 9.2, y + thickness)),
    ]


def build_serial_walls(rng: random.Random, _: int) -> List[Rect]:
    rects: List[Rect] = []
    ys = [-5.5, -2.4, 0.7, 3.8]
    rng.shuffle(ys)
    for k, y in enumerate(sorted(ys[: rng.randint(2, 4)])):
        thickness = rng.uniform(0.35, 0.7)
        gap_center = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 5.5)
        if k % 2:
            gap_center *= -1.0
        gap_width = rng.uniform(1.3, 2.4)
        rects.append(_clip_rect((-9.2, y, gap_center - gap_width / 2.0, y + thickness)))
        rects.append(_clip_rect((gap_center + gap_width / 2.0, y, 9.2, y + thickness)))
    return rects


def build_maze_branching(rng: random.Random, _: int) -> List[Rect]:
    rects: List[Rect] = []
    for x in [-6.0, -2.0, 2.0, 6.0]:
        gap_y = rng.uniform(-5.5, 5.5)
        gap_h = rng.uniform(1.6, 2.8)
        rects.append(_clip_rect((x, -9.2, x + 0.45, gap_y - gap_h / 2.0)))
        rects.append(_clip_rect((x, gap_y + gap_h / 2.0, x + 0.45, 9.2)))
    for y in [-6.5, -1.5, 3.5]:
        if rng.random() < 0.65:
            gap_x = rng.uniform(-5.5, 5.5)
            gap_w = rng.uniform(1.7, 3.0)
            rects.append(_clip_rect((-9.2, y, gap_x - gap_w / 2.0, y + 0.4)))
            rects.append(_clip_rect((gap_x + gap_w / 2.0, y, 9.2, y + 0.4)))
    return rects


def build_zigzag_narrow(rng: random.Random, _: int) -> List[Rect]:
    rects: List[Rect] = []
    y = -7.0
    side = 1
    while y < 6.8:
        gap_width = rng.uniform(1.15, 2.0)
        block_end = 7.6 - gap_width
        if side > 0:
            rects.append(_clip_rect((-9.2, y, block_end, y + rng.uniform(0.45, 0.8))))
        else:
            rects.append(_clip_rect((-block_end, y, 9.2, y + rng.uniform(0.45, 0.8))))
        y += rng.uniform(2.2, 3.0)
        side *= -1
    return rects


def build_bugtrap(rng: random.Random, _: int) -> List[Rect]:
    cx = rng.uniform(-2.0, 2.0)
    cy = rng.uniform(-1.8, 1.8)
    w = rng.uniform(4.0, 6.5)
    h = rng.uniform(4.0, 6.5)
    t = rng.uniform(0.35, 0.75)
    gap_w = rng.uniform(1.0, 1.9)
    opening_side = rng.choice(["left", "right", "top", "bottom"])

    left = cx - w / 2.0
    right = cx + w / 2.0
    bottom = cy - h / 2.0
    top = cy + h / 2.0

    rects: List[Rect] = []
    if opening_side != "bottom":
        rects.append(_clip_rect((left, bottom, right, bottom + t)))
    else:
        gx = rng.uniform(left + 1.0, right - 1.0)
        rects.extend([
            _clip_rect((left, bottom, gx - gap_w / 2.0, bottom + t)),
            _clip_rect((gx + gap_w / 2.0, bottom, right, bottom + t)),
        ])

    if opening_side != "top":
        rects.append(_clip_rect((left, top - t, right, top)))
    else:
        gx = rng.uniform(left + 1.0, right - 1.0)
        rects.extend([
            _clip_rect((left, top - t, gx - gap_w / 2.0, top)),
            _clip_rect((gx + gap_w / 2.0, top - t, right, top)),
        ])

    if opening_side != "left":
        rects.append(_clip_rect((left, bottom, left + t, top)))
    else:
        gy = rng.uniform(bottom + 1.0, top - 1.0)
        rects.extend([
            _clip_rect((left, bottom, left + t, gy - gap_w / 2.0)),
            _clip_rect((left, gy + gap_w / 2.0, left + t, top)),
        ])

    if opening_side != "right":
        rects.append(_clip_rect((right - t, bottom, right, top)))
    else:
        gy = rng.uniform(bottom + 1.0, top - 1.0)
        rects.extend([
            _clip_rect((right - t, bottom, right, gy - gap_w / 2.0)),
            _clip_rect((right - t, gy + gap_w / 2.0, right, top)),
        ])

    return [r for r in rects if not any(_point_in_rect(p, r, margin=0.7) for p in (START, GOAL))]


FAMILIES: Tuple[FamilySpec, ...] = (
    FamilySpec("small_open", "easy", build_small_open),
    FamilySpec("large_sparse", "easy_medium", build_large_sparse),
    FamilySpec("dense_clutter", "medium_hard", build_dense_clutter),
    FamilySpec("wall_gap", "medium", build_wall_gap),
    FamilySpec("serial_walls", "hard", build_serial_walls),
    FamilySpec("maze_branching", "hard", build_maze_branching),
    FamilySpec("bugtrap", "hard", build_bugtrap),
)


def make_stable_A(rng: random.Random) -> Tuple[List[List[float]], str]:
    regime = rng.choice(["sink", "rotate_cw", "rotate_ccw", "weak_shear"])
    if regime == "sink":
        a = rng.uniform(0.25, 0.85)
        shear = rng.uniform(-0.45, 0.45)
        A = [[-a, shear], [-shear * 0.5, -rng.uniform(0.25, 0.85)]]
    elif regime == "rotate_cw":
        omega = rng.uniform(0.8, 1.9)
        damp = rng.uniform(0.25, 0.75)
        A = [[-damp, omega], [-omega, -damp]]
    elif regime == "rotate_ccw":
        omega = rng.uniform(0.8, 1.9)
        damp = rng.uniform(0.25, 0.75)
        A = [[-damp, -omega], [omega, -damp]]
    else:
        A = [
            [-rng.uniform(0.35, 0.9), rng.uniform(-1.2, 1.2)],
            [rng.uniform(-0.5, 0.5), -rng.uniform(0.35, 0.9)],
        ]
    return [[round(v, 8) for v in row] for row in A], regime


def scenario_difficulty_score(rects: Sequence[Rect]) -> float:
    xmin, ymin, xmax, ymax = BOUNDS
    workspace_area = (xmax - xmin) * (ymax - ymin)
    occupancy = sum(_rect_area(r) for r in rects) / workspace_area
    return round(float(len(rects) + 20.0 * occupancy), 6)


def make_scenario(
    rng: random.Random,
    scenario_id: int,
    family: FamilySpec,
    family_index: int,
    *,
    u_max: float,
    goal_tol: float,
) -> Dict[str, object]:
    A, regime = make_stable_A(rng)
    rects = family.builder(rng, family_index)
    return {
        "scenario_id": scenario_id,
        "A_query": A,
        "B_query": DEFAULT_B,
        "rectangles": [_round_rect(r) for r in rects],
        "bounds": list(BOUNDS),
        "cell_size": 0.5,
        "start": list(START),
        "goal": list(GOAL),
        "u_max": float(u_max),
        "goal_tol": float(goal_tol),
        "difficulty": family.difficulty,
        "difficulty_score": scenario_difficulty_score(rects),
        "map_type": family.name,
        "benchmark_family": family.name,
        "regime": regime,
        "B_mode": "identity",
        "connected_checked": False,
        "map_resample_tries": 1,
    }


def selected_families(names: Iterable[str]) -> List[FamilySpec]:
    wanted = list(names)
    if not wanted or wanted == ["all"]:
        return list(FAMILIES)
    by_name = {f.name: f for f in FAMILIES}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown benchmark family/families: {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def generate(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.root).expanduser().resolve()
    input_dir = Path(args.output_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    families = selected_families(args.families)
    all_scenarios: List[Dict[str, object]] = []
    manifest_families: List[Dict[str, object]] = []
    next_id = 0

    for family in families:
        family_scenarios: List[Dict[str, object]] = []
        for family_idx in range(args.n_per_family):
            scenario = make_scenario(
                rng,
                family_idx,
                family,
                family_idx,
                u_max=args.u_max,
                goal_tol=args.goal_tol,
            )
            family_scenarios.append(scenario)
            combined_scenario = dict(scenario)
            combined_scenario["scenario_id"] = next_id
            combined_scenario["family_scenario_id"] = family_idx
            all_scenarios.append(combined_scenario)
            next_id += 1

        out_name = f"{args.prefix}_{family.name}.json"
        out_path = input_dir / out_name
        write_json(out_path, family_scenarios)
        manifest_families.append(
            {
                "family": family.name,
                "difficulty": family.difficulty,
                "count": len(family_scenarios),
                "file": str(out_path),
                "scenario_ids": [
                    int(family_scenarios[0]["scenario_id"]),
                    int(family_scenarios[-1]["scenario_id"]),
                ],
            }
        )
        print(f"[write] {out_path} scenarios={len(family_scenarios)}")

    combined_file = None
    if args.write_combined:
        combined_file = input_dir / f"{args.prefix}_all.json"
        write_json(combined_file, all_scenarios)
        print(f"[write] {combined_file} scenarios={len(all_scenarios)}")

    manifest = {
        "prefix": args.prefix,
        "seed": args.seed,
        "root": str(root),
        "input_dir": str(input_dir),
        "n_per_family": args.n_per_family,
        "total_scenarios": len(all_scenarios),
        "families": manifest_families,
        "combined_file": str(combined_file) if combined_file else None,
        "schema": {
            "solver": "motion_planning_solver.py",
            "problem_dictionary_location": "input/",
            "problem_name_format": "<json_stem>_sc_<scenario_id>",
            "required_keys": [
                "scenario_id",
                "A_query",
                "B_query",
                "rectangles",
                "bounds",
                "start",
                "goal",
            ],
        },
        "notes": [
            "Generated dictionaries are 2D because the current solver is 2D.",
            "The optional *_all.json file is for aggregate sweeps.",
        ],
    }
    manifest_path = input_dir / f"{args.prefix}_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[write] {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parent)
    parser.add_argument("--output_dir", default="input")
    parser.add_argument("--prefix", default="benchmark_dualmp")
    parser.add_argument("--n_per_family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--u_max", type=float, default=3.0)
    parser.add_argument("--goal_tol", type=float, default=0.5)
    parser.add_argument("--families", nargs="+", default=["all"], help="Family names or all.")
    parser.add_argument("--write_combined", action="store_true")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
