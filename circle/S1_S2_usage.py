"""
S1_S2_usage.py — "Always try S1, then fallback to S2 (from scratch)"

For each benchmark scenario:
  1) Retrieve S1 candidate (cluster -> dyn using M_query, then closest obstacle inst).
  2) Load retrieved S1 trajectory from NPZ.
  3) Re-check the retrieved S1 trajectory against the QUERY obstacle center (and goal).
     If it works (success==True) -> use S1 result.
     Otherwise -> run S2 MPC FROM SCRATCH on the query (NO warm-start).

Inputs:
  - circle/S1_database_single_obstacle.json   (from S1_layers.py)
  - circle/s1_mpc_trajectories.npz            (from S1_all_data.py)
  - circle/benchmark_scenarios.json           (from generate_new_scenarios.py)

Output:
  - circle/S1_S2_results.json

Dependencies:
  - numpy, json, do_mpc, casadi
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import do_mpc
from casadi import *  # noqa: F401,F403


# ============================================================
# 1) Similarity metric (REAL matrices only)
# ============================================================

def dynamics_diversity(M_query: np.ndarray, M_ref: np.ndarray) -> float:
    """
    Angle between flattened real matrices. Smaller => more similar.
    """
    v1 = np.asarray(M_query, dtype=float).flatten()
    v2 = np.asarray(M_ref, dtype=float).flatten()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return float(np.pi / 2)
    cos_theta = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.arccos(abs(cos_theta)))


def select_best_cluster(M_query: np.ndarray, db: Dict[str, Any]) -> Tuple[int, float]:
    best_cluster, best_div = None, np.inf
    for cid, cd in db["consensus_nodes"].items():
        M_cd = np.array(cd["M_cd"], dtype=float)
        div = dynamics_diversity(M_query, M_cd)
        if div < best_div:
            best_div = div
            best_cluster = int(cid)
    if best_cluster is None:
        raise RuntimeError("No clusters found in DB consensus_nodes.")
    return int(best_cluster), float(best_div)


def select_best_dyn(M_query: np.ndarray, db: Dict[str, Any], cluster_id: int) -> Tuple[int, float]:
    cd = db["consensus_nodes"][str(cluster_id)]
    best_dyn, best_div = None, np.inf
    for dyn_id in cd["dyn_children"]:
        M = np.array(db["dyn_nodes"][str(dyn_id)]["M"], dtype=float)
        div = dynamics_diversity(M_query, M)
        if div < best_div:
            best_div = div
            best_dyn = int(dyn_id)
    if best_dyn is None:
        raise RuntimeError(f"No dyn children found for cluster_id={cluster_id}.")
    return int(best_dyn), float(best_div)


def select_closest_obstacle(
    query_center: Tuple[float, float],
    dyn_node: Dict[str, Any],
    obs_type: str = "car",
) -> Tuple[int, Tuple[float, float], float]:
    if "obs_types" not in dyn_node or obs_type not in dyn_node["obs_types"]:
        raise KeyError(f"dyn_node missing obs_type='{obs_type}' under 'obs_types'.")

    best_idx, best_dist = None, np.inf
    best_center: Optional[Tuple[float, float]] = None

    qc = np.array([float(query_center[0]), float(query_center[1])], dtype=float)

    for i, inst in enumerate(dyn_node["obs_types"][obs_type]):
        cx, cy = inst["center"]
        c = np.array([float(cx), float(cy)], dtype=float)
        d = float(np.linalg.norm(c - qc))
        if d < best_dist:
            best_dist = d
            best_idx = int(i)
            best_center = (float(cx), float(cy))

    if best_idx is None or best_center is None:
        raise RuntimeError(f"No obstacle instances found for obs_type='{obs_type}'.")
    return int(best_idx), best_center, float(best_dist)


# ============================================================
# 2) Trajectory retrieval from NPZ (precomputed S1 rollout)
# ============================================================

def load_retrieved_trajectory(npz_data: Dict[str, Any], dyn_id: int, inst_idx: int) -> Optional[Dict[str, Any]]:
    dyn_ids = np.asarray(npz_data["dyn_id"]).astype(int)
    insts = np.asarray(npz_data["instance_idx"]).astype(int)
    mask = (dyn_ids == int(dyn_id)) & (insts == int(inst_idx))
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    i = int(idxs[0])

    out = {
        "states": np.array(npz_data["states"][i], dtype=float),    # (T+1,2)
        "inputs": np.array(npz_data["inputs"][i], dtype=float),    # (T,1) or (T,)
        "cost": float(npz_data["cost"][i]) if "cost" in npz_data else None,
        "success_db": bool(npz_data["success"][i]) if "success" in npz_data else None,
        "runtime_sec_db": float(npz_data["runtime_sec"][i]) if "runtime_sec" in npz_data else None,
    }
    return out


# ============================================================
# 3) M-map (if scenario doesn't contain M_query)
# ============================================================

def system_to_M_real(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega0: float = 1.0) -> np.ndarray:
    """
    REAL-valued surrogate:
        M = Re( C (jωI - A)^{-1} B )
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    jwI_minus_A = 1j * omega0 * np.eye(n) - A
    M_complex = C @ np.linalg.inv(jwI_minus_A) @ B
    return np.real(M_complex)


# ============================================================
# 4) S2 MPC (do-mpc) — from scratch ONLY
# ============================================================

def create_mpc(A: np.ndarray, obstacles: Optional[List[Dict[str, Any]]], dt: float = 0.01, n_horizon: int = 6):
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
            cons = -((x1 - float(cx))**2 + (x2 - float(cy))**2) + float(r)**2
            mpc.set_nl_cons(f"obstacle_{i}", cons, ub=0.0)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(integration_tool="cvodes", t_step=float(dt))
    simulator.setup()

    return mpc, simulator


def simulate_s2_from_scratch(
    A: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]],
    dt: float,
    n_steps: int,
    start_point: Tuple[float, float],
    n_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    S2 MPC solve on QUERY, from scratch (NO warm-start).
    """
    x0 = np.array([[float(start_point[0])], [float(start_point[1])]], dtype=float)

    mpc, simulator = create_mpc(A, obstacles, dt=dt, n_horizon=n_horizon)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    state_hist = [x0.squeeze().copy()]
    input_hist: List[float] = []
    total_cost = 0.0

    t0 = time.perf_counter()

    for _ in range(int(n_steps)):
        u0 = mpc.make_step(x0)
        x_next = simulator.make_step(u0)

        x_next = np.array(x_next, dtype=float).reshape(-1, 1)
        u0     = np.array(u0,     dtype=float).reshape(-1, 1)

        state_hist.append(x_next.squeeze().copy())
        input_hist.append(float(u0.squeeze()))

        x1_val, x2_val = x_next[0, 0], x_next[1, 0]
        u_val = u0[0, 0]
        total_cost += (10.0 * x1_val**2 + x2_val**2 + u_val**2) * float(dt)

        x0 = x_next

    x1_val, x2_val = x0[0, 0], x0[1, 0]
    total_cost += (10.0 * x1_val**2 + x2_val**2) * float(dt)

    runtime = float(time.perf_counter() - t0)

    states = np.array(state_hist, dtype=float)   # (T+1,2)
    inputs = np.array(input_hist, dtype=float)   # (T,)
    return states, inputs, float(total_cost), runtime


def goal_reached(states: np.ndarray, goal: Tuple[float, float] = (0.0, 0.0), tol: float = 0.5) -> bool:
    dx = float(states[-1, 0]) - float(goal[0])
    dy = float(states[-1, 1]) - float(goal[1])
    return (dx * dx + dy * dy) <= tol * tol


def collision_free_against_query_obstacle(
    states: np.ndarray,
    obstacle_center: Tuple[float, float],
    obstacle_radius: float,
    margin: float = 0.0,
) -> bool:
    xs = np.asarray(states[:, 0], dtype=float)
    ys = np.asarray(states[:, 1], dtype=float)
    cx, cy = float(obstacle_center[0]), float(obstacle_center[1])
    r = float(obstacle_radius) + float(margin)
    return not bool(np.any((xs - cx) ** 2 + (ys - cy) ** 2 < r * r))


# ============================================================
# 5) Main: Always try S1, re-check, else S2 from scratch
# ============================================================

def run_s1_s2(
    *,
    db_json: str = "circle/S1_database_single_obstacle.json",
    traj_npz: str = "circle/s1_mpc_trajectories.npz",
    scenarios_json: str = "circle/benchmark_scenarios.json",
    out_json: str = "circle/S1_S2_results.json",
    obs_type: str = "car",
    obstacle_radius: float = 0.5,
    collision_margin: float = 0.0,
    dt: float = 0.1,
    n_steps: int = 100,
    n_horizon: int = 6,
    start_point: Tuple[float, float] = (5.0, 5.0),
    goal_point: Tuple[float, float] = (0.0, 0.0),
    goal_tol: float = 0.5,
) -> List[Dict[str, Any]]:

    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]
    dyn_info = payload.get("dynamics_info", {})

    omega0 = float(dyn_info.get("omega0", 1.0))
    B = np.array(dyn_info.get("B", [[0.0], [1.0]]), dtype=float)
    C = np.array(dyn_info.get("C", [[1.0, 0.0]]), dtype=float)

    traj_data = dict(np.load(traj_npz, allow_pickle=True))
    scenarios = json.loads(Path(scenarios_json).read_text())

    results: List[Dict[str, Any]] = []

    for k, sc in enumerate(scenarios):
        scenario_id = int(sc.get("scenario_id", k))
        A_query = np.array(sc["A_query"], dtype=float)
        query_center = (float(sc["obstacle_center"][0]), float(sc["obstacle_center"][1]))

        if "M_query" in sc:
            M_query = np.array(sc["M_query"], dtype=float)
        else:
            M_query = system_to_M_real(A_query, B=B, C=C, omega0=omega0)

        # --------- Retrieval attempt (always) ----------
        cluster_id, div_cluster = select_best_cluster(M_query, db)
        dyn_id, div_dyn = select_best_dyn(M_query, db, cluster_id)

        dyn_node = db["dyn_nodes"][str(dyn_id)]
        inst_idx, inst_center, obs_dist = select_closest_obstacle(
            query_center=query_center,
            dyn_node=dyn_node,
            obs_type=obs_type,
        )

        retrieved = load_retrieved_trajectory(traj_data, dyn_id=dyn_id, inst_idx=inst_idx)

        # --------- Evaluate S1 on THIS query (re-check) ----------
        s1_ok = False
        s1_cf_query = None
        s1_gr_query = None

        if retrieved is not None and retrieved.get("states", None) is not None:
            s1_states = retrieved["states"]

            s1_cf_query = collision_free_against_query_obstacle(
                s1_states,
                obstacle_center=query_center,
                obstacle_radius=obstacle_radius,
                margin=collision_margin,
            )
            s1_gr_query = goal_reached(s1_states, goal=goal_point, tol=goal_tol)
            s1_ok = bool(s1_cf_query and s1_gr_query)

        if s1_ok:
            results.append({
                "scenario_id": scenario_id,
                "mode": "S1_only_after_recheck",
                "success": True,
                "cost": None if retrieved.get("cost") is None else float(retrieved["cost"]),
                "runtime_sec": 0.0,
                "retrieval": {
                    "cluster_id": int(cluster_id),
                    "dyn_id": int(dyn_id),
                    "inst_idx": int(inst_idx),
                    "inst_center": [float(inst_center[0]), float(inst_center[1])],
                    "div_cluster": float(div_cluster),
                    "div_dyn": float(div_dyn),
                    "obs_dist": float(obs_dist),
                    "success_db": retrieved.get("success_db", None),
                    "runtime_sec_db": retrieved.get("runtime_sec_db", None),
                    "cost_db": retrieved.get("cost", None),
                },
                "query": {
                    "center": [float(query_center[0]), float(query_center[1])],
                    "goal": [float(goal_point[0]), float(goal_point[1])],
                    "goal_tol": float(goal_tol),
                    "obstacle_radius": float(obstacle_radius),
                    "collision_margin": float(collision_margin),
                },
                "recheck": {
                    "collision_free_wrt_query_obstacle": None if s1_cf_query is None else bool(s1_cf_query),
                    "goal_reached": None if s1_gr_query is None else bool(s1_gr_query),
                },
            })
        else:
            # --------- Fallback to S2 from scratch ----------
            obstacles_query = [{
                "type": obs_type,
                "center": (float(query_center[0]), float(query_center[1])),
                "radius": float(obstacle_radius),
            }]

            try:
                states2, inputs2, cost2, runtime2 = simulate_s2_from_scratch(
                    A=A_query,
                    obstacles=obstacles_query,
                    dt=dt,
                    n_steps=n_steps,
                    start_point=start_point,
                    n_horizon=n_horizon,
                )
                succ2 = bool(goal_reached(states2, goal=goal_point, tol=goal_tol))

                results.append({
                    "scenario_id": scenario_id,
                    "mode": "S2_from_scratch_after_S1_failed",
                    "success": succ2,
                    "cost": float(cost2),
                    "runtime_sec": float(runtime2),
                    "retrieval": {
                        "cluster_id": int(cluster_id),
                        "dyn_id": int(dyn_id),
                        "inst_idx": int(inst_idx),
                        "inst_center": [float(inst_center[0]), float(inst_center[1])] if inst_center is not None else None,
                        "div_cluster": float(div_cluster),
                        "div_dyn": float(div_dyn),
                        "obs_dist": float(obs_dist),
                        "retrieved_available": bool(retrieved is not None),
                        "retrieved_success_db": None if retrieved is None else retrieved.get("success_db", None),
                        "retrieved_cost_db": None if retrieved is None else retrieved.get("cost", None),
                    },
                    "query": {
                        "center": [float(query_center[0]), float(query_center[1])],
                        "goal": [float(goal_point[0]), float(goal_point[1])],
                        "goal_tol": float(goal_tol),
                        "obstacle_radius": float(obstacle_radius),
                        "collision_margin": float(collision_margin),
                    },
                    "recheck": {
                        "s1_ok": bool(s1_ok),
                        "collision_free_wrt_query_obstacle": None if s1_cf_query is None else bool(s1_cf_query),
                        "goal_reached": None if s1_gr_query is None else bool(s1_gr_query),
                    },
                })

            except Exception as e:
                results.append({
                    "scenario_id": scenario_id,
                    "mode": "S2_from_scratch_after_S1_failed",
                    "success": False,
                    "cost": None,
                    "runtime_sec": None,
                    "error": repr(e),
                    "retrieval": {
                        "cluster_id": int(cluster_id),
                        "dyn_id": int(dyn_id),
                        "inst_idx": int(inst_idx),
                        "div_cluster": float(div_cluster),
                        "div_dyn": float(div_dyn),
                        "obs_dist": float(obs_dist),
                    },
                })

        if (k + 1) % 10 == 0 or (k + 1) == len(scenarios):
            print(f"[S1-then-S2] {k+1}/{len(scenarios)} done")

    Path(out_json).write_text(json.dumps(results, indent=2))
    print(f"✅ Saved results → {out_json}")
    return results


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_json", type=str, default="circle/S1_database_single_obstacle.json")
    p.add_argument("--traj_npz", type=str, default="circle/s1_mpc_trajectories.npz")
    p.add_argument("--scenarios", type=str, default="circle/benchmark_scenarios.json")
    p.add_argument("--out", type=str, default="circle/S1_S2_results.json")

    p.add_argument("--obs_type", type=str, default="car")
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--collision_margin", type=float, default=0.0)

    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--n_horizon", type=int, default=6)

    p.add_argument("--start_x", type=float, default=5.0)
    p.add_argument("--start_y", type=float, default=5.0)
    p.add_argument("--goal_x", type=float, default=0.0)
    p.add_argument("--goal_y", type=float, default=0.0)
    p.add_argument("--goal_tol", type=float, default=0.5)

    args = p.parse_args()

    run_s1_s2(
        db_json=args.db_json,
        traj_npz=args.traj_npz,
        scenarios_json=args.scenarios,
        out_json=args.out,
        obs_type=args.obs_type,
        obstacle_radius=args.radius,
        collision_margin=args.collision_margin,
        dt=args.dt,
        n_steps=args.n_steps,
        n_horizon=args.n_horizon,
        start_point=(args.start_x, args.start_y),
        goal_point=(args.goal_x, args.goal_y),
        goal_tol=args.goal_tol,
    )


if __name__ == "__main__":
    main()