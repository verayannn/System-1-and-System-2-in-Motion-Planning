#!/usr/bin/env python3
"""Collect MPC recovery demonstrations from states visited by an S1 checkpoint.

The output JSONL can be appended to ``train_s1_nonlinear.py --results_jsonl``.
Each row carries a scenario override whose start is the visited S1 state, so the
trainer reconstructs the correct map, goal, and nonlinear dynamics context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def parse_ids(value: str, count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(count))
    selected: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            selected.extend(range(int(left), int(right) + 1))
        else:
            selected.append(int(token))
    return [index for index in dict.fromkeys(selected) if 0 <= index < count]


def scenario_override(problem: Any, start: np.ndarray) -> dict[str, Any]:
    return {
        "scenario_id": int(problem.scenario_id),
        "rectangles": [list(map(float, rect)) for rect in problem.rects],
        "start": list(map(float, start[:2])),
        "goal": list(map(float, problem.goal[:2])),
        "bounds": list(map(float, problem.bounds)),
        "u_max": float(problem.u_max),
        "goal_tol": float(problem.goal_tol),
        "dynamics_type": str(problem.dynamics_type),
        "dynamics_model": str(problem.dynamics_model),
        "nonlinear_dynamics": problem.nonlinear_dynamics,
    }


def sample_indices(length: int, count: int) -> Iterable[int]:
    if length < 2:
        return []
    return np.unique(np.linspace(0, length - 2, max(1, count), dtype=int)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--s1_model", required=True)
    parser.add_argument("--scenario_ids", default="all")
    parser.add_argument("--states_per_scenario", type=int, default=4)
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    for path in (root, root / "sofai"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from input.input_handler import load_scenarios
    from solvers.S2_mpc import solve_MPC_with_info
    from solvers._s2_common import resolve_mplconfigdir
    from solvers.s1_nonlinear import load_s1_checkpoint, rollout_policy

    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, os.environ.get("MPLCONFIGDIR")))
    dictionary = Path(args.dictionary).expanduser()
    if not dictionary.is_absolute():
        dictionary = root / dictionary
    model_path = Path(args.s1_model).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    out_jsonl = Path(args.out_jsonl).expanduser()
    if not out_jsonl.is_absolute():
        out_jsonl = root / out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, norm, meta = load_s1_checkpoint(model_path, device)
    settings = dict(meta.get("dataset_meta", {}))
    scenarios = load_scenarios(str(dictionary))
    ids = parse_ids(args.scenario_ids, len(scenarios))
    collected = 0
    attempted = 0

    with out_jsonl.open("w") as handle:
        for scenario_index in ids:
            scenario = scenarios[scenario_index]
            states, _inputs, _info = rollout_policy(
                model,
                scenario,
                norm,
                device,
                total_steps=int(settings.get("n_steps_nom", 200)),
                dt_nom=float(settings.get("dt_nom", 0.05)),
                u_max_nom=float(settings.get("u_max_nom", scenario.u_max)),
                collision_margin=0.2,
                goal_tol=float(scenario.goal_tol),
                grid_n=int(settings.get("grid_n", 25)),
                n_steps_nom=int(settings.get("n_steps_nom", 200)),
                buffer_cells=int(settings.get("buffer_cells", 2)),
                stop_tol=float(settings.get("stop_tol", scenario.goal_tol)),
            )
            for state_index in sample_indices(len(states), int(args.states_per_scenario)):
                attempted += 1
                override = scenario_override(scenario, states[state_index])
                result = solve_MPC_with_info(override)
                if not result or not bool(result.get("success")):
                    continue
                attempt = {
                    "name": "s2_mpc_dagger",
                    "system": "s2",
                    "success": True,
                    "states": np.asarray(result["states"], dtype=float).tolist(),
                    "inputs": np.asarray(result.get("inputs", []), dtype=float).tolist(),
                    "dt": float(result.get("dt", settings.get("dt_nom", 0.05))),
                }
                row = {
                    "run_type": "s2",
                    "dictionary": str(dictionary),
                    "scenario_index": int(scenario_index),
                    "scenario_override": override,
                    "dagger": True,
                    "attempts": [attempt],
                    "selected_attempt": attempt["name"],
                    "success": True,
                }
                handle.write(json.dumps(row) + "\n")
                collected += 1
            print(f"[dagger] scenario={scenario_index} collected={collected} attempted={attempted}")
    print(f"[write] {out_jsonl} successful_mpc_recoveries={collected} attempted={attempted}")


if __name__ == "__main__":
    main()
