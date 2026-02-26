"""
S2_mpc_maze.py — System-2 MPC (do-mpc) for MAZE rectangles
Aligned with:
  - generate_benchmark_scenarios_maze.py
  - S2_cbf_maze.py benchmarking format

Success definition:
    success = collision_free AND goal_reached

Run:
  python maze/S2_mpc_maze.py \
      --scenarios maze/benchmark_scenarios_maze.json \
      --out maze/S2_mpc_results.json \
      --dt 0.05 --n_steps 800 --n_horizon 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import do_mpc
from casadi import fabs, fmax, fmin, sqrt, exp, log


Rect = Tuple[float, float, float, float]


# ============================================================
# Signed distance utilities (unchanged math)
# ============================================================

def sdf_aabb(px, py, xmin, xmax, ymin, ymax):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)

    dx = fabs(px - cx) - hx
    dy = fabs(py - cy) - hy

    ax = fmax(dx, 0)
    ay = fmax(dy, 0)

    outside = sqrt(ax * ax + ay * ay)
    inside  = fmin(fmax(dx, dy), 0)
    return outside + inside


def smooth_min(vals, kappa: float = 20.0):
    s = 0
    for v in vals:
        s = s + exp(-kappa * v)
    return -(1.0 / kappa) * log(s + 1e-12)


# ============================================================
# MPC creation
# ============================================================

def create_mpc(
    A: np.ndarray,
    B: np.ndarray,
    rectangles: List[List[float]],
    goal: Tuple[float, float],
    *,
    dt: float,
    n_horizon: int,
    u_max: float,
    bounds: Tuple[float, float, float, float],
    wall_margin: float,
    smooth_kappa: float,
):

    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)
    m = B.shape[1]

    gx, gy = float(goal[0]), float(goal[1])

    model = do_mpc.model.Model("continuous")

    x1 = model.set_variable("_x", "x1")
    x2 = model.set_variable("_x", "x2")
    u  = model.set_variable("_u", "u", shape=(m,1))

    model.set_rhs("x1", A[0,0]*x1 + A[0,1]*x2 + (B[0,:].reshape(1,-1) @ u)[0,0])
    model.set_rhs("x2", A[1,0]*x1 + A[1,1]*x2 + (B[1,:].reshape(1,-1) @ u)[0,0])
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=int(n_horizon),
        t_step=float(dt),
        state_discretization="collocation",
        collocation_type="radau",
        store_full_solution=False,
    )

    dx = x1 - gx
    dy = x2 - gy

    mterm = 10.0 * dx**2 + 10.0 * dy**2
    lterm = 10.0 * dx**2 + 10.0 * dy**2 + 0.2 * (u.T @ u)

    mpc.set_objective(mterm=mterm, lterm=lterm)

    xmin, ymin, xmax, ymax = bounds
    mpc.bounds["lower","_x","x1"] = xmin
    mpc.bounds["upper","_x","x1"] = xmax
    mpc.bounds["lower","_x","x2"] = ymin
    mpc.bounds["upper","_x","x2"] = ymax

    for j in range(m):
        mpc.bounds["lower","_u","u",j] = -float(u_max)
        mpc.bounds["upper","_u","u",j] =  float(u_max)

    if rectangles:
        dists = []
        for r in rectangles:
            xmin_r, ymin_r, xmax_r, ymax_r = map(float, r)
            dists.append(sdf_aabb(x1, x2, xmin_r, xmax_r, ymin_r, ymax_r))
        dmin = smooth_min(dists, kappa=float(smooth_kappa))
        mpc.set_nl_cons("rect_clearance", -dmin + float(wall_margin), ub=0.0)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(integration_tool="cvodes", t_step=float(dt))
    simulator.setup()

    return mpc, simulator


# ============================================================
# Collision check (explicit, like CBF version)
# ============================================================

def collision_free_rectangles(states: np.ndarray, rects: List[Rect], margin: float=0.0):
    xs = states[:,0]
    ys = states[:,1]
    for (xmin,ymin,xmax,ymax) in rects:
        if np.any((xs >= xmin-margin)&(xs<=xmax+margin)&
                  (ys >= ymin-margin)&(ys<=ymax+margin)):
            return False
    return True


def goal_reached(states: np.ndarray, goal: Tuple[float,float], tol: float):
    dx = states[-1,0] - goal[0]
    dy = states[-1,1] - goal[1]
    return (dx*dx + dy*dy) <= tol*tol


# ============================================================
# Simulation
# ============================================================

def simulate_scenario(
    A, B, rectangles, start, goal,
    *,
    dt, n_steps, n_horizon,
    u_max, bounds,
    wall_margin, smooth_kappa,
    goal_tol
):

    mpc, simulator = create_mpc(
        A, B, rectangles, goal,
        dt=dt,
        n_horizon=n_horizon,
        u_max=u_max,
        bounds=bounds,
        wall_margin=wall_margin,
        smooth_kappa=smooth_kappa
    )

    x = np.array([[start[0]],[start[1]]], dtype=float)
    mpc.x0 = x
    simulator.x0 = x
    mpc.set_initial_guess()

    states = [x.squeeze()]
    inputs = []

    t0 = time.perf_counter()

    for _ in range(int(n_steps)):
        u = mpc.make_step(x)
        x = simulator.make_step(u)

        x = np.array(x).reshape(2,1)
        u = np.array(u).reshape(-1,1)

        states.append(x.squeeze())
        inputs.append(u.squeeze())

        if goal_reached(np.array(states), goal, goal_tol):
            break

    runtime = time.perf_counter() - t0

    states = np.array(states)
    inputs = np.array(inputs)

    collision_free = collision_free_rectangles(states, rectangles)
    reached = goal_reached(states, goal, goal_tol)
    success = bool(collision_free and reached)

    return states, inputs, runtime, success


# ============================================================
# Runner
# ============================================================

def run_s2_mpc_on_scenarios(
    scenarios_json: str,
    out_json: str,
    *,
    dt, n_steps, n_horizon,
    wall_margin, smooth_kappa,
    goal_tol
):

    scenarios = json.loads(Path(scenarios_json).read_text())
    results = []

    total_success = 0
    runtimes = []

    for idx, sc in enumerate(scenarios):
        sid = int(sc.get("scenario_id", idx))

        A = sc["A_query"]
        B = sc["B_query"]
        rects = sc["rectangles"]
        bounds = tuple(sc["bounds"])
        start = tuple(sc["start"])
        goal  = tuple(sc["goal"])
        u_max = float(sc.get("u_max", 3.0))

        states, inputs, runtime, success = simulate_scenario(
            A, B, rects, start, goal,
            dt=dt,
            n_steps=n_steps,
            n_horizon=n_horizon,
            u_max=u_max,
            bounds=bounds,
            wall_margin=wall_margin,
            smooth_kappa=smooth_kappa,
            goal_tol=goal_tol
        )

        total_success += int(success)
        runtimes.append(runtime)

        print(f"[S2-MPC] {sid} success={success} runtime={runtime:.3f}s")

        results.append({
            "scenario_id": sid,
            "success": success,
            "runtime_sec": float(runtime),
            "states": states.tolist(),
            "inputs": inputs.tolist(),
        })

    Path(out_json).write_text(json.dumps(results, indent=2))

    print("\n=== MPC Summary ===")
    print("scenarios:", len(results))
    print("success:", total_success)
    print("success_rate:", total_success/len(results))
    print("avg_runtime:", np.mean(runtimes))


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="maze/benchmark_scenarios_maze.json")
    ap.add_argument("--out", default="maze/S2_mpc_results.json")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=800)
    ap.add_argument("--n_horizon", type=int, default=20)
    ap.add_argument("--wall_margin", type=float, default=0.2)
    ap.add_argument("--smooth_kappa", type=float, default=20.0)
    ap.add_argument("--goal_tol", type=float, default=0.5)
    args = ap.parse_args()

    run_s2_mpc_on_scenarios(
        scenarios_json=args.scenarios,
        out_json=args.out,
        dt=args.dt,
        n_steps=args.n_steps,
        n_horizon=args.n_horizon,
        wall_margin=args.wall_margin,
        smooth_kappa=args.smooth_kappa,
        goal_tol=args.goal_tol,
    )


if __name__ == "__main__":
    main()