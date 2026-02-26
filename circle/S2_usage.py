"""
S2_usage.py — System-2 only (MPC from scratch) benchmark using do-mpc

Reads:
  - circle/benchmark_scenarios.json  (from generate_new_scenarios.py)

Runs:
  - do-mpc MPC solve for each scenario using A_query + obstacle_center

Writes:
  - circle/S2_results.json

Note:
- This is S2-only baseline (no retrieval, no warm-start).
- Goal is (0,0) by default; success is based on distance-to-goal tolerance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import do_mpc
from casadi import *  # noqa: F401,F403  (do-mpc relies on CasADi symbols)


# ============================================================
# 1) MPC (matches your style)
# ============================================================

def create_mpc(
    A: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]],
    dt: float = 0.01,
    n_horizon: int = 6,
):
    # Ensure A is a real-valued numpy array
    A = np.asarray(A, dtype=float)

    model = do_mpc.model.Model("continuous")
    B = np.array([[0.0], [1.0]], dtype=float)

    x1 = model.set_variable(var_type="_x", var_name="x1", shape=(1, 1))
    x2 = model.set_variable(var_type="_x", var_name="x2", shape=(1, 1))
    u  = model.set_variable(var_type="_u", var_name="u",  shape=(1, 1))

    model.set_rhs("x1", A[0, 0] * x1 + A[0, 1] * x2 + B[0, 0] * u)
    model.set_rhs("x2", A[1, 0] * x1 + A[1, 1] * x2 + B[1, 0] * u)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=int(n_horizon),
        t_step=float(dt),
        state_discretization="collocation",
        collocation_type="radau",
        store_full_solution=True,
    )

    # Cost
    mterm = 10 * x1**2 + x2**2
    lterm = 10 * x1**2 + x2**2 + u**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.0)

    # Bounds
    mpc.bounds["lower", "_x", "x1"] = -10
    mpc.bounds["upper", "_x", "x1"] =  10
    mpc.bounds["lower", "_x", "x2"] = -10
    mpc.bounds["upper", "_x", "x2"] =  10
    mpc.bounds["lower", "_u", "u"]  = -5
    mpc.bounds["upper", "_u", "u"]  =  5

    # Obstacle constraints
    if obstacles is not None:
        for i, obs in enumerate(obstacles):
            cx, cy = obs["center"]
            r = obs["radius"]
            cons = -((x1 - float(cx))**2 + (x2 - float(cy))**2) + float(r)**2
            mpc.set_nl_cons(f"obstacle_{i}", cons, ub=0.0)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(integration_tool="cvodes", t_step=float(dt))
    simulator.setup()

    return model, mpc, simulator


def simulate_scenario(
    A: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]],
    dt: float = 0.01,
    n_steps: int = 500,
    start_point: Tuple[float, float] = (5.0, 5.0),
    n_horizon: int = 6,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # Start from chosen point (force float)
    x0 = np.array([[float(start_point[0])],
                   [float(start_point[1])]], dtype=float)

    _, mpc, simulator = create_mpc(A, obstacles, dt=dt, n_horizon=n_horizon)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    state_hist = [x0.squeeze().copy()]  # (2,)
    input_hist = []
    total_cost = 0.0

    t0 = time.perf_counter()

    for _ in range(int(n_steps)):
        u0 = mpc.make_step(x0)
        x_next = simulator.make_step(u0)

        # Ensure numpy float arrays
        x_next = np.array(x_next, dtype=float).reshape(-1, 1)
        u0     = np.array(u0,     dtype=float).reshape(-1, 1)

        state_hist.append(x_next.squeeze().copy())
        input_hist.append(u0.squeeze().copy())

        x1_val, x2_val = x_next[0, 0], x_next[1, 0]
        u_val = u0[0, 0]
        stage_cost = 10 * x1_val**2 + x2_val**2 + u_val**2
        total_cost += stage_cost * float(dt)

        x0 = x_next

    # Terminal cost (keep your convention)
    x1_val, x2_val = x0[0, 0], x0[1, 0]
    total_cost += (10 * x1_val**2 + x2_val**2) * float(dt)

    runtime = float(time.perf_counter() - t0)

    states = np.array(state_hist, dtype=float)   # (T+1, 2)
    inputs = np.array(input_hist, dtype=float)   # (T,) typically
    return states, inputs, float(total_cost), runtime


# ============================================================
# 2) Success metric
# ============================================================

def goal_reached(states: np.ndarray, goal: Tuple[float, float] = (0.0, 0.0), tol: float = 0.5) -> bool:
    dx = float(states[-1, 0]) - float(goal[0])
    dy = float(states[-1, 1]) - float(goal[1])
    return (dx * dx + dy * dy) <= tol * tol


# ============================================================
# 3) Run S2-only benchmark
# ============================================================

def run_s2_on_scenarios(
    *,
    scenarios_json: str = "circle/benchmark_scenarios.json",
    out_json: str = "circle/S2_results.json",
    dt: float = 0.1,
    n_steps: int = 100,
    n_horizon: int = 6,
    start_point: Tuple[float, float] = (5.0, 5.0),
    goal_point: Tuple[float, float] = (0.0, 0.0),
    goal_tol: float = 0.5,
    obstacle_radius: float = 0.5,
) -> List[Dict[str, Any]]:

    scenarios = json.loads(Path(scenarios_json).read_text())
    results: List[Dict[str, Any]] = []

    for sc in scenarios:
        scenario_id = int(sc["scenario_id"])
        A_query = np.array(sc["A_query"], dtype=float)
        center = tuple(sc["obstacle_center"])

        obstacles = [{
            "type": "car",
            "center": (float(center[0]), float(center[1])),
            "radius": float(obstacle_radius),
        }]

        try:
            states, inputs, cost, runtime = simulate_scenario(
                A=A_query,
                obstacles=obstacles,
                dt=dt,
                n_steps=n_steps,
                start_point=start_point,
                n_horizon=n_horizon,
            )

            succ = bool(goal_reached(states, goal=goal_point, tol=goal_tol))

            results.append({
                "scenario_id": scenario_id,
                "success": succ,
                "cost": float(cost),
                "runtime_sec": float(runtime),
                "used_center": [float(center[0]), float(center[1])],
            })

        except Exception as e:
            results.append({
                "scenario_id": scenario_id,
                "success": False,
                "cost": None,
                "runtime_sec": None,
                "used_center": [float(center[0]), float(center[1])],
                "error": repr(e),
            })

        if (len(results) % 10 == 0) or (len(results) == len(scenarios)):
            print(f"[S2] {len(results)}/{len(scenarios)} done")

    Path(out_json).write_text(json.dumps(results, indent=2))
    print(f"✅ Saved S2 results → {out_json}")

    return results


if __name__ == "__main__":
    run_s2_on_scenarios()
