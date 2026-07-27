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
from dataclasses import replace
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


def valid_recovery(
    result: dict[str, Any] | None,
    requested_start: np.ndarray,
) -> tuple[bool, str]:
    """Reject malformed S2 recoveries before they can enter training."""
    if not isinstance(result, dict):
        return False, "missing_result"
    if not bool(result.get("success", result.get("solved", False))):
        return False, "unsolved"
    if not bool(result.get("collision_free", False)):
        return False, "collision"
    states = np.asarray(result.get("states"), dtype=float)
    inputs = np.asarray(result.get("inputs"), dtype=float)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 2:
        return False, "invalid_states"
    # ``maybe_patch_goal_trajectory`` can append one final goal state without
    # an associated control, so MPC legitimately returns one fewer input than
    # state transitions. The trainer derives any missing final control from
    # adjacent states.
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[0] > states.shape[0] - 1:
        return False, "invalid_inputs"
    if not (np.all(np.isfinite(states)) and np.all(np.isfinite(inputs))):
        return False, "nonfinite"
    if not np.allclose(states[0, :2], requested_start[:2], atol=1e-5, rtol=0.0):
        return False, "start_mismatch"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--s1_model", required=True)
    parser.add_argument("--scenario_ids", default="all")
    parser.add_argument("--states_per_scenario", type=int, default=4)
    parser.add_argument(
        "--s2_solver",
        choices=["mpc", "cbf"],
        default="mpc",
        help="Teacher used to label S1-visited states.",
    )
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    for path in (root, root / "sofai"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from input.input_handler import load_scenarios
    from solvers._s2_common import resolve_mplconfigdir
    from solvers.s1_nonlinear import load_s1_checkpoint, rollout_policy
    if args.s2_solver == "mpc":
        from solvers.S2_mpc import solve_MPC_with_info as solve_with_info
    else:
        from solvers.S2_cbf import solve_CBF_with_info as solve_with_info

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
    rejected: dict[str, int] = {}

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
                recovery_start = np.asarray(states[state_index], dtype=float)
                override = scenario_override(scenario, recovery_start)
                # S2 solvers consume MazeProblem attributes, not a serialised
                # dictionary. Passing ``override`` directly silently substituted
                # their zero-valued defaults and produced fake [0, 0] recoveries.
                recovery_problem = replace(
                    scenario,
                    start=tuple(map(float, recovery_start[:2])),
                )
                result = solve_with_info(recovery_problem)
                valid, reason = valid_recovery(result, recovery_start)
                if not valid:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                attempt = {
                    "name": f"s2_{args.s2_solver}_dagger",
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
    print(
        f"[write] {out_jsonl} teacher={args.s2_solver} "
        f"valid_recoveries={collected} attempted={attempted} rejected={rejected}"
    )


if __name__ == "__main__":
    main()
