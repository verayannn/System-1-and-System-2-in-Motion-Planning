"""
S1_all_data.py — Load S1 DB JSON and run MPC for ALL scenarios (dyn_id × obstacle instance),
saving trajectories + metrics.

Assumes your JSON (from updated S1_layers.py) has:
  payload["db"]["dyn_nodes"][str(dyn_id)]["A"]   -> real 2x2
  payload["db"]["dyn_nodes"][str(dyn_id)]["obs_types"]["car"][k]["center"] -> [cx, cy]

Outputs:
  1) NPZ file with trajectories:
       - states:   (N, T+1, 2) float32
       - inputs:   (N, T,   1) float32
       - dyn_id:   (N,) int32
       - inst_idx: (N,) int32
       - center:   (N, 2) float32
       - cost:     (N,) float64
       - runtime:  (N,) float64
       - success / collision_free / goal_reached: (N,) int8
       - errors: list of strings (stored in a separate JSON)
  2) CSV file with per-scenario metrics
  3) summary JSON

Run:
  python S1_all_data.py --db_json S1_database_single_obstacle.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import do_mpc
from casadi import *  # do-mpc uses casadi symbols


# ============================================================
# 1) MPC creation + simulation (same structure as your S1_data.py)
# ============================================================

def create_mpc(A: np.ndarray, obstacles: Optional[List[Dict[str, Any]]], dt: float = 0.01, n_horizon: int = 6):
    model = do_mpc.model.Model("continuous")
    B = np.array([[0.0], [1.0]])

    x1 = model.set_variable(var_type="_x", var_name="x1", shape=(1, 1))
    x2 = model.set_variable(var_type="_x", var_name="x2", shape=(1, 1))
    u  = model.set_variable(var_type="_u", var_name="u",  shape=(1, 1))

    model.set_rhs("x1", A[0, 0] * x1 + A[0, 1] * x2 + B[0, 0] * u)
    model.set_rhs("x2", A[1, 0] * x1 + A[1, 1] * x2 + B[1, 0] * u)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=n_horizon,
        t_step=dt,
        state_discretization="collocation",
        collocation_type="radau",
        store_full_solution=True,
    )

    mterm = 10 * x1**2 + x2**2
    lterm = 10 * x1**2 + x2**2 + u**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.0)

    mpc.bounds["lower", "_x", "x1"] = -10
    mpc.bounds["upper", "_x", "x1"] =  10
    mpc.bounds["lower", "_x", "x2"] = -10
    mpc.bounds["upper", "_x", "x2"] =  10
    mpc.bounds["lower", "_u", "u"]  = -5
    mpc.bounds["upper", "_u", "u"]  =  5

    if obstacles is not None:
        for i, obs in enumerate(obstacles):
            cx, cy = obs["center"]
            r = obs["radius"]
            cons = -((x1 - cx) ** 2 + (x2 - cy) ** 2) + r**2
            mpc.set_nl_cons(f"obstacle_{i}", cons, ub=0.0)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(integration_tool="cvodes", t_step=dt)
    simulator.setup()

    return model, mpc, simulator


def simulate_scenario(
    A: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]],
    dt: float,
    n_steps: int,
    start_point: Tuple[float, float],
    n_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x0 = np.array([[start_point[0]], [start_point[1]]], dtype=float)

    _, mpc, simulator = create_mpc(A, obstacles, dt=dt, n_horizon=n_horizon)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    state_hist = [x0.squeeze().copy()]
    input_hist = []
    total_cost = 0.0

    t0 = time.perf_counter()

    for _ in range(n_steps):
        u0 = mpc.make_step(x0)
        x_next = simulator.make_step(u0)

        x_next = np.array(x_next).reshape(-1, 1)
        u0 = np.array(u0).reshape(-1, 1)

        state_hist.append(x_next.squeeze().copy())
        input_hist.append(u0.squeeze().copy())

        # Option A: stage cost * dt
        x1_val, x2_val = float(x_next[0, 0]), float(x_next[1, 0])
        u_val = float(u0[0, 0])
        total_cost += (10.0 * x1_val**2 + x2_val**2 + u_val**2) * dt

        x0 = x_next

    # Option A: terminal cost (NO dt)
    x1_T, x2_T = float(x0[0, 0]), float(x0[1, 0])
    total_cost += (10.0 * x1_T**2 + x2_T**2)

    runtime = float(time.perf_counter() - t0)

    return np.array(state_hist), np.array(input_hist), float(total_cost), runtime


# ============================================================
# 2) Metrics
# ============================================================

def collision_free(states: np.ndarray, obstacles: List[Dict[str, Any]], margin: float = 0.0) -> bool:
    xs = states[:, 0]
    ys = states[:, 1]
    for obs in obstacles:
        cx, cy = obs["center"]
        r = obs["radius"] + margin
        if np.any((xs - cx) ** 2 + (ys - cy) ** 2 < r**2):
            return False
    return True


def goal_reached(states: np.ndarray, goal: Tuple[float, float], tol: float = 0.5) -> bool:
    dx = float(states[-1, 0]) - float(goal[0])
    dy = float(states[-1, 1]) - float(goal[1])
    return (dx * dx + dy * dy) <= tol * tol


# ============================================================
# 3) Main loop over all scenarios + saving
# ============================================================

def run_all(
    db_json_path: str,
    obs_type: str,
    base_radius: float,
    dt: float,
    n_steps: int,
    n_horizon: int,
    start_point: Tuple[float, float],
    goal_point: Tuple[float, float],
    goal_tol: float,
    collision_margin: float,
    out_npz: str,
    out_csv: str,
    out_summary: str,
    out_errors_json: str,
    max_dyn: Optional[int] = None,
    max_instances_per_dyn: Optional[int] = None,
) -> Dict[str, Any]:
    payload = json.loads(Path(db_json_path).read_text())
    dyn_nodes = payload["db"]["dyn_nodes"]

    dyn_id_keys = sorted(dyn_nodes.keys(), key=lambda s: int(s))
    if max_dyn is not None:
        dyn_id_keys = dyn_id_keys[: int(max_dyn)]

    # count scenarios first
    scenario_specs: List[Tuple[int, int, Tuple[float, float]]] = []
    for k in dyn_id_keys:
        node = dyn_nodes[k]
        if "A" not in node:
            raise KeyError(
                f"dyn_id={k} missing 'A' in JSON. Regenerate JSON with updated S1_layers.py."
            )
        obs_types = node.get("obs_types", {})
        if obs_type not in obs_types:
            continue
        instances = obs_types[obs_type]
        if max_instances_per_dyn is not None:
            instances = instances[: int(max_instances_per_dyn)]
        for inst_idx, inst in enumerate(instances):
            c = inst["center"]
            scenario_specs.append((int(k), inst_idx, (float(c[0]), float(c[1]))))

    N = len(scenario_specs)
    if N == 0:
        raise RuntimeError("No scenarios found to run. Check obs_type and JSON content.")

    T = int(n_steps)

    # allocate arrays (store float32 to keep file smaller)
    states_all = np.zeros((N, T + 1, 2), dtype=np.float32)
    inputs_all = np.zeros((N, T, 1), dtype=np.float32)

    dyn_id_all = np.zeros((N,), dtype=np.int32)
    inst_all = np.zeros((N,), dtype=np.int32)
    center_all = np.zeros((N, 2), dtype=np.float32)

    cost_all = np.full((N,), np.nan, dtype=np.float64)
    runtime_all = np.full((N,), np.nan, dtype=np.float64)

    success_all = np.zeros((N,), dtype=np.int8)
    cf_all = np.zeros((N,), dtype=np.int8)
    gr_all = np.zeros((N,), dtype=np.int8)

    errors: List[Dict[str, Any]] = []

    t_global0 = time.perf_counter()

    for idx, (dyn_id, inst_idx, center) in enumerate(scenario_specs):
        node = dyn_nodes[str(dyn_id)]
        A = np.array(node["A"], dtype=float)

        obstacles = [{
            "center": center,
            "radius": float(base_radius),
            "type": obs_type,
        }]

        dyn_id_all[idx] = dyn_id
        inst_all[idx] = inst_idx
        center_all[idx, :] = np.array(center, dtype=np.float32)

        try:
            states, inputs, cost, runtime = simulate_scenario(
                A=A,
                obstacles=obstacles,
                dt=dt,
                n_steps=n_steps,
                start_point=start_point,
                n_horizon=n_horizon,
            )

            # store trajectory
            states_all[idx, :, :] = states.astype(np.float32)
            # inputs might come out as shape (T,) sometimes; enforce (T,1)
            inputs = np.array(inputs).reshape(T, 1)
            inputs_all[idx, :, :] = inputs.astype(np.float32)

            cost_all[idx] = cost
            runtime_all[idx] = runtime

            cf = collision_free(states, obstacles, margin=collision_margin)
            gr = goal_reached(states, goal=goal_point, tol=goal_tol)
            succ = bool(cf and gr)

            cf_all[idx] = int(cf)
            gr_all[idx] = int(gr)
            success_all[idx] = int(succ)

        except Exception as e:
            errors.append({
                "global_idx": idx,
                "dyn_id": dyn_id,
                "instance_idx": inst_idx,
                "center": center,
                "error": repr(e),
            })

        if (idx + 1) % 20 == 0 or (idx + 1) == N:
            print(f"[{idx+1:>5}/{N}] done")

    t_global = float(time.perf_counter() - t_global0)

    # ---- Save trajectories ----
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        states=states_all,
        inputs=inputs_all,
        dyn_id=dyn_id_all,
        instance_idx=inst_all,
        center=center_all,
        cost=cost_all,
        runtime_sec=runtime_all,
        success=success_all,
        collision_free=cf_all,
        goal_reached=gr_all,
        dt=np.array([dt], dtype=np.float32),
        n_steps=np.array([n_steps], dtype=np.int32),
        n_horizon=np.array([n_horizon], dtype=np.int32),
        start=np.array(start_point, dtype=np.float32),
        goal=np.array(goal_point, dtype=np.float32),
    )

    # ---- Save CSV metrics ----
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "global_idx", "dyn_id", "instance_idx", "center_x", "center_y",
        "cost", "runtime_sec", "collision_free", "goal_reached", "success"
    ]
    with open(out_csv, "w") as f:
        f.write(",".join(headers) + "\n")
        for i in range(N):
            f.write(
                f"{i},{int(dyn_id_all[i])},{int(inst_all[i])},"
                f"{float(center_all[i,0])},{float(center_all[i,1])},"
                f"{float(cost_all[i])},{float(runtime_all[i])},"
                f"{int(cf_all[i])},{int(gr_all[i])},{int(success_all[i])}\n"
            )

    # ---- Save errors ----
    out_errors_json = Path(out_errors_json)
    out_errors_json.parent.mkdir(parents=True, exist_ok=True)
    out_errors_json.write_text(json.dumps(errors, indent=2))

    # ---- Summary ----
    valid = np.isfinite(cost_all) & np.isfinite(runtime_all)
    summary = {
        "n_scenarios_total": int(N),
        "n_scenarios_successfully_solved": int(np.sum(valid)),
        "n_errors": int(len(errors)),
        "success_rate": float(np.mean(success_all[valid])) if np.any(valid) else 0.0,
        "avg_cost": float(np.mean(cost_all[valid])) if np.any(valid) else float("nan"),
        "avg_runtime_sec": float(np.mean(runtime_all[valid])) if np.any(valid) else float("nan"),
        "total_wall_time_sec": float(t_global),
        "paths": {
            "db_json": str(db_json_path),
            "npz": str(out_npz),
            "csv": str(out_csv),
            "summary_json": str(out_summary),
            "errors_json": str(out_errors_json),
        },
        "settings": {
            "obs_type": obs_type,
            "base_radius": base_radius,
            "dt": dt,
            "n_steps": n_steps,
            "n_horizon": n_horizon,
            "start_point": start_point,
            "goal_point": goal_point,
            "goal_tol": goal_tol,
            "collision_margin": collision_margin,
            "max_dyn": max_dyn,
            "max_instances_per_dyn": max_instances_per_dyn,
        }
    }

    out_summary = Path(out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))

    print("\nDone. Summary:")
    print(json.dumps(summary, indent=2))

    return summary


# ============================================================
# 4) CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_json", type=str, default="S1_database_single_obstacle.json")

    p.add_argument("--obs_type", type=str, default="car")
    p.add_argument("--base_radius", type=float, default=0.5)

    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--n_horizon", type=int, default=6)

    p.add_argument("--start_x", type=float, default=5.0)
    p.add_argument("--start_y", type=float, default=5.0)
    p.add_argument("--goal_x", type=float, default=0.0)
    p.add_argument("--goal_y", type=float, default=0.0)

    p.add_argument("--goal_tol", type=float, default=0.5)
    p.add_argument("--collision_margin", type=float, default=0.0)

    p.add_argument("--out_npz", type=str, default="s1_mpc_trajectories.npz")
    p.add_argument("--out_csv", type=str, default="s1_mpc_metrics.csv")
    p.add_argument("--out_summary", type=str, default="s1_mpc_summary.json")
    p.add_argument("--out_errors", type=str, default="s1_mpc_errors.json")

    p.add_argument("--max_dyn", type=int, default=None)
    p.add_argument("--max_instances_per_dyn", type=int, default=None)

    args = p.parse_args()

    run_all(
        db_json_path=args.db_json,
        obs_type=args.obs_type,
        base_radius=args.base_radius,
        dt=args.dt,
        n_steps=args.n_steps,
        n_horizon=args.n_horizon,
        start_point=(args.start_x, args.start_y),
        goal_point=(args.goal_x, args.goal_y),
        goal_tol=args.goal_tol,
        collision_margin=args.collision_margin,
        out_npz=args.out_npz,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        out_errors_json=args.out_errors,
        max_dyn=args.max_dyn,
        max_instances_per_dyn=args.max_instances_per_dyn,
    )


if __name__ == "__main__":
    main()
