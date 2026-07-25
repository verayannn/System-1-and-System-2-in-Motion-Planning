#!/usr/bin/env python3
"""Run one SOFAI motion-planning case and plot the result.

This is the same idea as `motion_planning_solver.py`, but packaged as a
single-case visualization tool:

- load one scenario from a benchmark dictionary
- run S1, S2, or SOFAI
- save the raw result JSON
- save a 2D trajectory plot over the obstacle map

Example:

cd /Users/apple/Documents/GitHub/System-1-and-System-2-in-Motion-Planning


SOFAI_MPC_HORIZON=200 \
SOFAI_MPC_DT=0.01 \
SOFAI_MPC_STEPS=5000 \
SOFAI_MPC_MAX_ITER=500 \
SOFAI_MPC_REFERENCE_GRID=0.20 \
SOFAI_MPC_Q_POS=75 \
SOFAI_MPC_R_DU=0.75 \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
python visualize_mp.py \
  --problem_dictionary nl/benchmark_dualmp_nl_long_slalom_eval_long_slalom.json \
  --scenario_ids 5 \
  --s1 neural \
  --s2 cbf \
  --run_type s1 \
  --out_dir output/visualize_mp \
  --out_prefix s1_sc5_ls


PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
python visualize_mp.py \
  --problem_dictionary nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --scenario_ids 6 \
  --s1 neural \
  --s2 cbf \
  --run_type s1 \
  --out_dir output/visualize_mp \
  --out_prefix s1_sc6

mpc_do

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def configure_imports(root: Path, mplconfigdir: str) -> None:
    for path in (root, root / "sofai", root / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from solvers._s2_common import bootstrap_acados_backend, resolve_mplconfigdir

    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, mplconfigdir))

    try:
        bootstrap_acados_backend()
    except Exception:
        pass


def parse_single_scenario_id(raw: str) -> int:
    raw = str(raw).strip()
    if not raw:
        raise SystemExit("--scenario_ids must contain exactly one integer")
    if "," in raw or "-" in raw or raw.lower() == "all":
        raise SystemExit("This script runs exactly one case. Use one integer, e.g. --scenario_ids 7")
    return int(raw)


def jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def scenario_to_dict(scenario: Any) -> Dict[str, Any]:
    payload = {
        "scenario_id": int(scenario.scenario_id),
        "A_query": jsonable(getattr(scenario, "A", None)),
        "B_query": jsonable(getattr(scenario, "B", None)),
        "rectangles": [list(r) for r in scenario.rects],
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "bounds": list(scenario.bounds),
        "u_max": float(scenario.u_max),
        "goal_tol": float(getattr(scenario, "goal_tol", 0.6)),
    }
    dynamics_type = getattr(scenario, "dynamics_type", None)
    dynamics_model = getattr(scenario, "dynamics_model", None)
    nonlinear_dynamics = getattr(scenario, "nonlinear_dynamics", None)
    if dynamics_type is not None:
        payload["dynamics_type"] = dynamics_type
    if dynamics_model:
        payload["dynamics_model"] = dynamics_model
    if nonlinear_dynamics is not None:
        payload["nonlinear_dynamics"] = jsonable(nonlinear_dynamics)
    return payload


def add_metrics(attempt: Dict[str, Any], scenario: Any) -> Dict[str, Any]:
    from solvers._s2_common import collision_free_rectangles, goal_reached

    states_raw = attempt.get("states")
    if states_raw is None:
        attempt.update(
            {
                "success": False,
                "collision_free": False,
                "goal_reached": False,
                "final_goal_error": None,
                "path_length": None,
                "num_states": 0,
                "correctness": 0.0,
            }
        )
        return attempt

    states = np.asarray(states_raw, dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        attempt["states"] = None
        return add_metrics(attempt, scenario)

    collision_free = bool(collision_free_rectangles(states[:, :2], scenario.rects))
    reached = bool(goal_reached(states[:, :2], scenario.goal, getattr(scenario, "goal_tol", 0.6)))
    diffs = states[1:, :2] - states[:-1, :2]
    path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(states) > 1 else 0.0
    final_error = float(math.dist(states[-1, :2], scenario.goal))
    success = bool(collision_free and reached)

    attempt.update(
        {
            "success": success,
            "collision_free": collision_free,
            "goal_reached": reached,
            "final_goal_error": final_error,
            "path_length": path_length,
            "num_states": int(states.shape[0]),
            "correctness": 1.0 if success else 0.0,
        }
    )
    return attempt


def add_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    from solvers._s2_common import (
        benchmark_family_from_dictionary,
        quality_refs_for_result,
        quality_score,
        quality_weights_for_family,
        selected_success_attempt,
        trajectory_quality_components,
    )

    selected = selected_success_attempt(result)
    family = benchmark_family_from_dictionary(result.get("dictionary", ""))
    weights = quality_weights_for_family(family)
    result["quality_family"] = family
    result["quality_weight_path_length"] = weights["path_length"]
    result["quality_weight_control_effort"] = weights["control_effort"]
    result["quality_weight_smoothness"] = weights["smoothness"]

    if selected is None or not bool(selected.get("success", False)):
        result.update(
            {
                "quality_path_length": None,
                "quality_control_effort": None,
                "quality_smoothness": None,
                "quality_j": None,
                "quality_score": None,
                "quality_path_length_ref": None,
                "quality_control_effort_ref": None,
                "quality_smoothness_ref": None,
            }
        )
        return result

    sample = trajectory_quality_components(result)
    if sample is None:
        result.update(
            {
                "quality_path_length": None,
                "quality_control_effort": None,
                "quality_smoothness": None,
                "quality_j": None,
                "quality_score": None,
                "quality_path_length_ref": None,
                "quality_control_effort_ref": None,
                "quality_smoothness_ref": None,
            }
        )
        return result

    refs = quality_refs_for_result(result)

    j = (
        weights["path_length"] * float(sample["path_length"]) / refs["path_length"]
        + weights["control_effort"] * float(sample["control_effort"]) / refs["control_effort"]
        + weights["smoothness"] * float(sample["smoothness"]) / refs["smoothness"]
    )

    result.update(
        {
            "quality_path_length": float(sample["path_length"]),
            "quality_control_effort": float(sample["control_effort"]),
            "quality_smoothness": float(sample["smoothness"]),
            "quality_j": float(j),
            "quality_score": float(quality_score(
                sample,
                refs,
                weights,
            )),
            "quality_path_length_ref": refs["path_length"],
            "quality_control_effort_ref": refs["control_effort"],
            "quality_smoothness_ref": refs["smoothness"],
        }
    )
    return result


def run_s1(scenario: Any, mode: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if mode == "neural":
        from solvers.S1_memory_neural import solveMemoryNeural

        states, confidence, info = solveMemoryNeural(scenario, return_info=True)
        inputs = info.get("inputs")
        dt = info.get("dt")
    else:
        from solvers.S1_motion_primitives import solveMotionPrimitives

        states, confidence = solveMotionPrimitives(scenario)
        inputs, dt = None, None

    return {
        "name": f"s1_{mode}",
        "system": "s1",
        "mode": mode,
        "states": None if states is None else states.tolist(),
        "inputs": inputs,
        "dt": dt,
        "confidence": float(confidence),
        "runtime_sec": time.perf_counter() - t0,
    }


def run_s2(scenario: Any, mode: str, *, s1_guidance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if mode == "cbf":
        from solvers.S2_cbf import solve_CBF_with_info

        info = solve_CBF_with_info(scenario)
    elif mode == "mpc_do":
        from solvers.S2_mpc_do import solve_MPC_DO_with_info

        info = solve_MPC_DO_with_info(
            scenario,
            s1_states=None if s1_guidance is None else s1_guidance.get("states"),
            s1_inputs=None if s1_guidance is None else s1_guidance.get("inputs"),
            s1_dt=None if s1_guidance is None else s1_guidance.get("dt"),
        )
    else:
        from solvers.S2_mpc import solve_MPC_with_info

        info = solve_MPC_with_info(
            scenario,
            s1_states=None if s1_guidance is None else s1_guidance.get("states"),
            s1_inputs=None if s1_guidance is None else s1_guidance.get("inputs"),
            s1_dt=None if s1_guidance is None else s1_guidance.get("dt"),
        )
    states = info.get("states") if isinstance(info, dict) else None

    return {
        "name": f"s2_{mode}",
        "system": "s2",
        "mode": mode,
        "states": None if states is None else states.tolist(),
        "inputs": None if not isinstance(info, dict) or info.get("inputs") is None else np.asarray(info["inputs"]).tolist(),
        "dt": None if not isinstance(info, dict) else info.get("dt"),
        "confidence": 1.0 if states is not None else 0.0,
        "runtime_sec": time.perf_counter() - t0,
        "mpc_s1_warm_start": None if not isinstance(info, dict) else info.get("s1_warm_start"),
        "mpc_solver_status": None if not isinstance(info, dict) else info.get("solver_status"),
    }


def choose_selected(attempts: List[Dict[str, Any]], run_type: str) -> Optional[Dict[str, Any]]:
    if not attempts:
        return None
    if run_type in {"s1", "s2"}:
        return attempts[0]
    for attempt in attempts:
        if attempt.get("success"):
            return attempt
    return attempts[-1]


def _resolve_dictionary_path(root: Path, problem_dictionary: str | Path) -> Path:
    path = Path(problem_dictionary).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                root / path,
                root / "input" / path,
                root / "input" / "nl" / path,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve benchmark dictionary: {problem_dictionary}")


def run_case(args: argparse.Namespace) -> Dict[str, Any]:
    from input.input_handler import load_scenarios

    dictionary_path = _resolve_dictionary_path(Path(args.root), args.problem_dictionary)

    scenarios = load_scenarios(str(dictionary_path))
    scenario_id = parse_single_scenario_id(args.scenario_ids)
    if scenario_id < 0 or scenario_id >= len(scenarios):
        raise SystemExit(f"scenario_id {scenario_id} is outside 0..{len(scenarios) - 1}")

    scenario = scenarios[scenario_id]
    attempts: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    if args.run_type == "s1":
        attempts.append(add_metrics(run_s1(scenario, args.s1), scenario))
    elif args.run_type == "s2":
        attempts.append(add_metrics(run_s2(scenario, args.s2), scenario))
    else:
        s1_attempt = add_metrics(run_s1(scenario, args.s1), scenario)
        attempts.append(s1_attempt)
        if args.run_all_attempts or not s1_attempt.get("success", False):
            attempts.append(add_metrics(run_s2(scenario, args.s2, s1_guidance=s1_attempt), scenario))

    selected = choose_selected(attempts, args.run_type)
    result = {
        "dictionary": dictionary_path.name,
        "dictionary_path": str(dictionary_path),
        "scenario_id": scenario_id,
        "run_type": args.run_type,
        "s1": args.s1,
        "s2": args.s2,
        "scenario": scenario_to_dict(scenario),
        "attempts": attempts,
        "selected_attempt": None if selected is None else selected["name"],
        "success": bool(selected and selected.get("success", False)),
        "selected_runtime_sec": None if selected is None else selected.get("runtime_sec"),
        "planning_runtime_sec": float(sum(float(attempt.get("runtime_sec", 0.0) or 0.0) for attempt in attempts)),
        "runtime_sec": time.perf_counter() - t0,
    }
    return add_quality(result)


def plot_result(result: Dict[str, Any], out_png: Path, *, plot_all_attempts: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scenario = result["scenario"]
    bounds = scenario["bounds"]
    rects = scenario["rectangles"]
    start = scenario["start"]
    goal = scenario["goal"]

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)

    for rect in rects:
        x1, y1, x2, y2 = rect
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                facecolor="#3b3b3b",
                edgecolor="#111111",
                linewidth=0.8,
                alpha=0.82,
            )
        )

    ax.scatter([start[0]], [start[1]], c="#1f77b4", s=90, marker="o", label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], c="#2ca02c", s=120, marker="*", label="goal", zorder=5)

    selected_name = result.get("selected_attempt")
    attempts = result["attempts"]
    if not plot_all_attempts:
        attempts = [a for a in attempts if a["name"] == selected_name]

    colors = ["#d62728", "#9467bd", "#ff7f0e", "#17becf"]
    for idx, attempt in enumerate(attempts):
        states = attempt.get("states")
        if states is None:
            continue
        arr = np.asarray(states, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        label = attempt["name"]
        if attempt["name"] == selected_name:
            label += " selected"
        ax.plot(arr[:, 0], arr[:, 1], color=colors[idx % len(colors)], linewidth=2.2, label=label)
        ax.scatter(arr[-1, 0], arr[-1, 1], color=colors[idx % len(colors)], s=45, marker="x", zorder=6)

    title = f"{result['dictionary']} scenario {result['scenario_id']} | {selected_name}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_result(result: Dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parent)
    parser.add_argument("--problem_dictionary", default="benchmark_dualmp_wall_gap.json")
    parser.add_argument("--scenario_ids", required=True, help="Exactly one scenario id, e.g. 1")
    parser.add_argument("--s1", choices=["neural", "primitives"], default="primitives")
    parser.add_argument("--s2", choices=["cbf", "mpc", "mpc_do"], default="mpc")
    parser.add_argument("--run_type", choices=["sofai", "s1", "s2"], default="sofai")
    parser.add_argument("--run_all_attempts", action="store_true")
    parser.add_argument("--plot_all_attempts", action="store_true")
    parser.add_argument("--out_dir", default="output/visualize_mp")
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--mplconfigdir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else repo_root
    args.root = root
    configure_imports(root, args.mplconfigdir)

    result = run_case(args)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    stem = Path(result["dictionary"]).stem
    prefix = args.out_prefix.strip() or f"{stem}_sc_{result['scenario_id']}_{args.run_type}_{args.s2}"
    out_json = out_dir / f"{prefix}_result.json"
    out_png = out_dir / f"{prefix}_trajectory.png"

    write_result(result, out_json)
    plot_result(result, out_png, plot_all_attempts=args.plot_all_attempts)

    print(f"[write] {out_json}")
    print(f"[write] {out_png}")
    quality_text = "nan" if result.get("quality_score") is None else f"{float(result['quality_score']):.3f}"
    print(
        "[summary] "
        f"selected={result['selected_attempt']} "
        f"success={result['success']} "
        f"runtime={result['runtime_sec']:.3f}s "
        f"quality={quality_text}"
    )


if __name__ == "__main__":
    main()
