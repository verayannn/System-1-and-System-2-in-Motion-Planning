#!/usr/bin/env python3
"""Generate DualMP benchmark dictionaries with nonlinear query dynamics.

The obstacle families, bounds, start/goal, and metadata stay aligned with
generate_benchmark_dictionaries.py. The difference is the query dynamics:
each scenario now stores an explicit nonlinear model description instead of
linearized A/B matrices.

cd /Users/apple/Desktop/sofai
python input/generate_nl_dict.py \
  --families dense_clutter \
  --n_per_family 50 \
  --seed 7 \
  --write_combined


"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import generate_benchmark_dictionaries as base


Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]

BOUNDS = base.BOUNDS
START = base.START
GOAL = base.GOAL
FAMILIES = base.FAMILIES
selected_families = base.selected_families
scenario_difficulty_score = base.scenario_difficulty_score
write_json = base.write_json


def _round_point(point: Point) -> List[float]:
    return [round(float(v), 6) for v in point]


def make_nonlinear_dynamics(rng: random.Random) -> Dict[str, object]:
    """Describe a genuinely nonlinear 2D control-affine model.

    The payload is explicit and self-contained so downstream code can consume
    it without relying on linearized Jacobians.
    """
    regime = rng.choice(["sink", "rotate_cw", "rotate_ccw", "weak_shear"])
    x0 = rng.uniform(-1.5, 1.5)
    y0 = rng.uniform(-1.5, 1.5)

    if regime == "sink":
        a = rng.uniform(0.25, 0.85)
        b = rng.uniform(0.25, 0.85)
        shear = rng.uniform(-0.6, 0.6)
        equations = {
            "x_dot": "u_x - a*tanh(x) + shear*sin(y)",
            "y_dot": "u_y - b*tanh(y) - 0.5*shear*sin(x)",
        }
        params = {"a": a, "b": b, "shear": shear}
    elif regime == "rotate_cw":
        damp = rng.uniform(0.25, 0.75)
        omega = rng.uniform(0.8, 1.9)
        equations = {
            "x_dot": "u_x - damp*tanh(x) + omega*sin(y)",
            "y_dot": "u_y - damp*tanh(y) - omega*sin(x)",
        }
        params = {"damp": damp, "omega": omega}
    elif regime == "rotate_ccw":
        damp = rng.uniform(0.25, 0.75)
        omega = rng.uniform(0.8, 1.9)
        equations = {
            "x_dot": "u_x - damp*tanh(x) - omega*sin(y)",
            "y_dot": "u_y - damp*tanh(y) + omega*sin(x)",
        }
        params = {"damp": damp, "omega": omega}
    else:
        a = rng.uniform(0.35, 0.9)
        b = rng.uniform(0.35, 0.9)
        shear = rng.uniform(-0.9, 0.9)
        equations = {
            "x_dot": "u_x - a*tanh(x) + 2*shear*y",
            "y_dot": "u_y - b*tanh(y) + 0.25*shear*x",
        }
        params = {"a": a, "b": b, "shear": shear}

    return {
        "model": "control_affine_tanh_trig_2d",
        "state_dim": 2,
        "control_dim": 2,
        "state_names": ["x", "y"],
        "control_names": ["u_x", "u_y"],
        "regime": regime,
        "linearization_point": _round_point((x0, y0)),
        "parameters": params,
        "equations": equations,
        "control_map": "identity",
        "nonlinear": True,
    }


def make_scenario(
    rng: random.Random,
    scenario_id: int,
    family: base.FamilySpec,
    family_index: int,
    *,
    u_max: float,
    goal_tol: float,
) -> Dict[str, object]:
    dynamics = make_nonlinear_dynamics(rng)
    rects = family.builder(rng, family_index)
    return {
        "scenario_id": scenario_id,
        "rectangles": [base._round_rect(r) for r in rects],
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
        "dynamics_type": "nonlinear",
        "dynamics_model": dynamics["model"],
        "regime": dynamics["regime"],
        "nonlinear_dynamics": dynamics,
        "connected_checked": False,
        "map_resample_tries": 1,
    }


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
                "rectangles",
                "bounds",
                "start",
                "goal",
                "dynamics_type",
                "nonlinear_dynamics",
            ],
        },
        "notes": [
            "Generated dictionaries keep the same maze geometry and metadata as the linear benchmark generator.",
            "Nonlinear query dynamics are described explicitly through the nonlinear_dynamics block.",
            "No linearized A_query/B_query surrogate is emitted in this schema.",
        ],
    }
    manifest_path = input_dir / f"{args.prefix}_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[write] {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parent)
    parser.add_argument("--output_dir", default="nl")
    parser.add_argument("--prefix", default="benchmark_dualmp_nl")
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
