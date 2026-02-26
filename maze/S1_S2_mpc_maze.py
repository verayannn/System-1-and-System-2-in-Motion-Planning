"""
S1_S2_mpc_maze.py — Hybrid "S1 if works else S2" pipeline for MAZE (rectangles)

Behavior
--------
For each benchmark scenario:
  1) Run System-1 retrieval (same logic as S1_usage_maze.py):
       cluster by A-distance -> dyn by fast M-distance -> map by situation cosine
       -> retrieve stored trajectory from NPZ -> re-check collision + goal
       w.r.t. the *QUERY* rectangles/goal.
  2) If S1 success == True: output the System-1 trajectory directly.
  3) Else: run System-2 MPC (do-mpc) on the query scenario.
       IMPORTANT: In this MPC hybrid, System-1 trajectory is NOT used
       as a reference / warm-start.

Inputs
------
- DB JSON: produced by S1_layers_maze.py (see S1_usage_maze.py header)
- Trajectory NPZ: produced by your S1_all_data_maze_*.py (must contain states/inputs/dyn_id/map_idx)
- Benchmark scenarios JSON: produced by generate_benchmark_scenarios_maze.py

Output
------
Writes a JSON list (one dict per scenario) with:
  - scenario_id
  - method_used: "S1" or "S2"
  - success
  - runtime_sec
  - states, inputs
  - plus System-1 retrieval diagnostics for analysis

Run
---
python maze/S1_S2_mpc_maze.py \
  --db_json maze/S1_database_maze.json \
  --traj_npz maze/s1_sfcbf_success_trajs.npz \
  --scenarios maze/benchmark_scenarios_maze.json \
  --out maze/S1_S2_mpc_results_maze.json \
  --dt 0.05 --n_steps 800 --n_horizon 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# System-2 deps
import do_mpc
from casadi import fabs, fmax, fmin, sqrt, exp, log


Rect = Tuple[float, float, float, float]


# ============================================================
# System-1 retrieval (copied from S1_usage_maze.py; no CVX)
# ============================================================

def A_distance(A_query: np.ndarray, A_ref: np.ndarray, normalize: bool = True) -> float:
    Aq = np.asarray(A_query, dtype=float)
    Ar = np.asarray(A_ref, dtype=float)
    d = float(np.linalg.norm(Aq - Ar, ord="fro"))
    if not normalize:
        return d
    denom = float(np.linalg.norm(Ar, ord="fro")) + 1e-12
    return d / denom


def select_best_cluster_A(A_query: np.ndarray, db: Dict[str, Any]) -> Tuple[int, float]:
    best_cluster: Optional[int] = None
    best_d = float("inf")

    for cid_str, cd_node in db["consensus_nodes"].items():
        cid = int(cid_str)
        cd_dyn_id = int(cd_node["cd_dyn_id"])
        A_cd = np.array(db["dyn_nodes"][str(cd_dyn_id)]["A"], dtype=float)
        d = A_distance(A_query, A_cd, normalize=True)
        if d < best_d:
            best_d = d
            best_cluster = cid

    if best_cluster is None:
        raise RuntimeError("No clusters found in DB.")
    return best_cluster, float(best_d)


def system_to_M(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega0: float = 1.0) -> np.ndarray:
    n = A.shape[0]
    jwI_minus_A = 1j * float(omega0) * np.eye(n) - A
    M_complex = C @ np.linalg.inv(jwI_minus_A) @ B
    return np.real(M_complex)


def select_best_dyn_fast(
    A_query: np.ndarray,
    B_query: np.ndarray,
    db: Dict[str, Any],
    cluster_id: int,
    *,
    omega0: float = 1.0,
) -> Tuple[int, float]:
    """Fast approximate dynamics similarity: Frobenius distance in M-space."""
    C = np.eye(2, dtype=float)
    M_q = system_to_M(A_query, B_query, C, omega0=float(omega0))
    M_q_flat = M_q.flatten()

    cd = db["consensus_nodes"][str(cluster_id)]
    best_dyn = None
    best_dist = float("inf")

    for dyn_id in cd["dyn_children"]:
        dyn_id = int(dyn_id)
        M_dyn = np.array(db["dyn_nodes"][str(dyn_id)]["M"], dtype=float)
        d = float(np.linalg.norm(M_q_flat - M_dyn.flatten()))
        if d < best_dist:
            best_dist = d
            best_dyn = dyn_id

    if best_dyn is None:
        raise RuntimeError("No dynamics children found in selected cluster.")
    return int(best_dyn), float(best_dist)


def _dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    for _ in range(int(iters)):
        m = np.maximum.reduce(
            [
                m,
                np.roll(m, 1, axis=0),
                np.roll(m, -1, axis=0),
                np.roll(m, 1, axis=1),
                np.roll(m, -1, axis=1),
            ]
        ).astype(np.uint8)
    return m


def compute_situation_vector(
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    *,
    bounds: Tuple[float, float, float, float],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    grid_n: int,
    dt_nom: float,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
) -> np.ndarray:
    """Nominal rollout (no obstacles) -> visited cells; corridor dilation; OR with obstacles in corridor."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    def u_nominal_local(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        v = -(x - g)
        rhs = v - A @ x
        u, *_ = np.linalg.lstsq(B, rhs, rcond=None)
        return np.clip(u, -float(u_max_nom), float(u_max_nom))

    xmin, ymin, xmax, ymax = map(float, bounds)
    x = np.array(start, dtype=float).reshape(2)
    g = np.array(goal, dtype=float).reshape(2)

    visited = np.zeros((int(grid_n), int(grid_n)), dtype=np.uint8)

    for _ in range(int(n_steps_nom)):
        if float(np.sum((x - g) ** 2)) <= float(stop_tol) ** 2:
            break
        u = u_nominal_local(x, g)
        x = x + float(dt_nom) * (A @ x + B @ u)

        if not (xmin <= x[0] <= xmax and ymin <= x[1] <= ymax):
            continue

        j = int((x[0] - xmin) / (xmax - xmin) * grid_n)
        i = int((x[1] - ymin) / (ymax - ymin) * grid_n)
        i = int(np.clip(i, 0, grid_n - 1))
        j = int(np.clip(j, 0, grid_n - 1))
        visited[i, j] = 1

    corridor = _dilate4(visited, iters=int(buffer_cells))

    obs = np.zeros((int(grid_n), int(grid_n)), dtype=np.uint8)
    sx = (xmax - xmin) / float(grid_n)
    sy = (ymax - ymin) / float(grid_n)

    for (rx1, ry1, rx2, ry2) in rects:
        x1, x2 = (min(rx1, rx2), max(rx1, rx2))
        y1, y2 = (min(ry1, ry2), max(ry1, ry2))

        j1 = int((x1 - xmin) / sx)
        j2 = int((x2 - xmin) / sx)
        i1 = int((y1 - ymin) / sy)
        i2 = int((y2 - ymin) / sy)

        j1 = int(np.clip(j1, 0, grid_n - 1))
        j2 = int(np.clip(j2, 0, grid_n - 1))
        i1 = int(np.clip(i1, 0, grid_n - 1))
        i2 = int(np.clip(i2, 0, grid_n - 1))

        obs[i1 : i2 + 1, j1 : j2 + 1] = 1

    area = np.maximum(visited, (obs & corridor).astype(np.uint8))
    return area.reshape(-1).astype(np.uint8)


def select_best_map_by_situation(v_query: np.ndarray, situation_vecs: List[List[int]]) -> Tuple[int, float]:
    V_db = np.array(situation_vecs, dtype=np.uint8)
    v_q = v_query.astype(np.uint8)

    norm_q = float(np.sqrt(np.sum(v_q)))
    norm_db = np.sqrt(np.sum(V_db, axis=1)).astype(float)
    inter = np.sum(V_db & v_q, axis=1).astype(float)
    cosines = inter / (norm_db * norm_q + 1e-12)

    best_idx = int(np.argmax(cosines))
    return best_idx, float(cosines[best_idx])


def _build_index(traj_npz: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    dyn_ids = np.array(traj_npz["dyn_id"]).astype(int)
    map_idxs = np.array(traj_npz["map_idx"]).astype(int)
    idx: Dict[Tuple[int, int], int] = {}
    for i in range(len(dyn_ids)):
        idx[(int(dyn_ids[i]), int(map_idxs[i]))] = int(i)
    return idx


def load_trajectory(
    traj_npz: Dict[str, Any],
    index: Dict[Tuple[int, int], int],
    dyn_id: int,
    map_idx: int,
) -> Optional[Dict[str, Any]]:
    key = (int(dyn_id), int(map_idx))
    if key not in index:
        return None
    i = index[key]
    return {
        "states": np.array(traj_npz["states"][i], dtype=float),
        "inputs": np.array(traj_npz["inputs"][i], dtype=float),
        "runtime_sec_db": float(traj_npz["runtime_sec"][i]) if "runtime_sec" in traj_npz else None,
        "success_db": bool(traj_npz["success"][i]) if "success" in traj_npz else None,
    }


def collision_free_rectangles(states: np.ndarray, rects: List[Rect], margin: float = 0.0) -> bool:
    xs = np.asarray(states[:, 0], dtype=float)
    ys = np.asarray(states[:, 1], dtype=float)
    m = float(margin)
    for (xmin, ymin, xmax, ymax) in rects:
        if np.any((xs >= xmin - m) & (xs <= xmax + m) & (ys >= ymin - m) & (ys <= ymax + m)):
            return False
    return True


def goal_reached(states: np.ndarray, goal: Tuple[float, float], tol: float = 0.6) -> bool:
    dx = float(states[-1, 0]) - float(goal[0])
    dy = float(states[-1, 1]) - float(goal[1])
    return (dx * dx + dy * dy) <= float(tol) * float(tol)


# ============================================================
# System-2 MPC (copied from S2_mpc_maze.py)
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
    inside = fmin(fmax(dx, dy), 0)
    return outside + inside


def smooth_min(vals, kappa: float = 20.0):
    s = 0
    for v in vals:
        s = s + exp(-kappa * v)
    return -(1.0 / kappa) * log(s + 1e-12)


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
    u = model.set_variable("_u", "u", shape=(m, 1))

    model.set_rhs("x1", A[0, 0] * x1 + A[0, 1] * x2 + (B[0, :].reshape(1, -1) @ u)[0, 0])
    model.set_rhs("x2", A[1, 0] * x1 + A[1, 1] * x2 + (B[1, :].reshape(1, -1) @ u)[0, 0])
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
    mpc.bounds["lower", "_x", "x1"] = xmin
    mpc.bounds["upper", "_x", "x1"] = xmax
    mpc.bounds["lower", "_x", "x2"] = ymin
    mpc.bounds["upper", "_x", "x2"] = ymax

    for j in range(m):
        mpc.bounds["lower", "_u", "u", j] = -float(u_max)
        mpc.bounds["upper", "_u", "u", j] = float(u_max)

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


def simulate_mpc_scenario(
    A,
    B,
    rectangles,
    start,
    goal,
    *,
    dt,
    n_steps,
    n_horizon,
    u_max,
    bounds,
    wall_margin,
    smooth_kappa,
    goal_tol,
):
    mpc, simulator = create_mpc(
        A,
        B,
        rectangles,
        goal,
        dt=dt,
        n_horizon=n_horizon,
        u_max=u_max,
        bounds=bounds,
        wall_margin=wall_margin,
        smooth_kappa=smooth_kappa,
    )

    x = np.array([[start[0]], [start[1]]], dtype=float)
    mpc.x0 = x
    simulator.x0 = x
    mpc.set_initial_guess()

    states = [x.squeeze()]
    inputs = []

    t0 = time.perf_counter()
    for _ in range(int(n_steps)):
        u = mpc.make_step(x)
        x = simulator.make_step(u)

        x = np.array(x).reshape(2, 1)
        u = np.array(u).reshape(-1, 1)

        states.append(x.squeeze())
        inputs.append(u.squeeze())

        if goal_reached(np.array(states), goal, float(goal_tol)):
            break

    runtime = float(time.perf_counter() - t0)
    states = np.array(states)
    inputs = np.array(inputs)

    collision_free = collision_free_rectangles(states, [tuple(map(float, r)) for r in rectangles])
    reached = goal_reached(states, goal, float(goal_tol))
    success = bool(collision_free and reached)
    return states, inputs, runtime, success


# ============================================================
# Hybrid runner
# ============================================================

def run_hybrid(
    *,
    db_json: str,
    traj_npz_path: str,
    scenarios_json: str,
    out_json: str,
    # S1 params
    omega0: float,
    collision_margin: float,
    s1_goal_tol: float,
    grid_n: int,
    dt_nom: float,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
    # S2 MPC params
    dt: float,
    n_steps: int,
    n_horizon: int,
    wall_margin: float,
    smooth_kappa: float,
    s2_goal_tol: float,
) -> List[Dict[str, Any]]:
    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]

    traj_npz = dict(np.load(traj_npz_path, allow_pickle=True))
    traj_index = _build_index(traj_npz)

    scenarios = json.loads(Path(scenarios_json).read_text())
    results: List[Dict[str, Any]] = []

    for idx, sc in enumerate(scenarios):
        sid = int(sc.get("scenario_id", idx))
        A_query = np.array(sc["A_query"], dtype=float)
        B_query = np.array(sc["B_query"], dtype=float)

        rects = [tuple(map(float, r)) for r in sc["rectangles"]]
        bounds = tuple(map(float, sc.get("bounds", [-10.0, -10.0, 10.0, 10.0])))
        start = tuple(map(float, sc.get("start", [5.0, 5.0])))
        goal = tuple(map(float, sc.get("goal", [0.0, 0.0])))
        u_max = float(sc.get("u_max", 3.0))

        # ----------------------
        # System 1
        # ----------------------
        t_s1 = time.perf_counter()
        cluster_id, d_cluster = select_best_cluster_A(A_query, db)
        dyn_id, dyn_dist = select_best_dyn_fast(A_query, B_query, db, cluster_id, omega0=float(omega0))

        dyn_node = db["dyn_nodes"][str(dyn_id)]
        situation_vecs = dyn_node["env_types"]["maze"]["situation_vecs"]
        v_q = compute_situation_vector(
            A_query,
            B_query,
            rects,
            bounds=bounds,
            start=start,
            goal=goal,
            grid_n=int(grid_n),
            dt_nom=float(dt_nom),
            n_steps_nom=int(n_steps_nom),
            u_max_nom=float(u_max_nom),
            buffer_cells=int(buffer_cells),
            stop_tol=float(stop_tol),
        )
        map_idx, cos_sim = select_best_map_by_situation(v_q, situation_vecs)

        traj = load_trajectory(traj_npz, traj_index, dyn_id, map_idx)
        s1_runtime = float(time.perf_counter() - t_s1)

        s1_success = False
        s1_cf_query = None
        s1_gr_query = None
        s1_states = None
        s1_inputs = None
        if traj is not None and traj.get("states") is not None:
            s1_states = np.asarray(traj["states"], dtype=float)
            s1_inputs = np.asarray(traj["inputs"], dtype=float)
            s1_cf_query = collision_free_rectangles(s1_states, rects, margin=float(collision_margin))
            s1_gr_query = goal_reached(s1_states, goal, tol=float(s1_goal_tol))
            s1_success = bool(s1_cf_query and s1_gr_query)

        # ----------------------
        # Decision
        # ----------------------
        if s1_success:
            method = "S1"
            states = s1_states
            inputs = s1_inputs
            runtime = s1_runtime
            success = True
        else:
            method = "S2"
            states, inputs, runtime, success = simulate_mpc_scenario(
                A_query.tolist(),
                B_query.tolist(),
                [list(map(float, r)) for r in rects],
                start,
                goal,
                dt=float(dt),
                n_steps=int(n_steps),
                n_horizon=int(n_horizon),
                u_max=float(u_max),
                bounds=bounds,
                wall_margin=float(wall_margin),
                smooth_kappa=float(smooth_kappa),
                goal_tol=float(s2_goal_tol),
            )

        print(f"[S1->S2-MPC] sc={sid} method={method} success={success} (s1_success={s1_success})")

        results.append(
            {
                "scenario_id": sid,
                "method_used": method,
                "success": bool(success),
                "runtime_sec": float(runtime),
                "states": states.tolist() if states is not None else None,
                "inputs": inputs.tolist() if inputs is not None else None,

                # --- S1 diagnostics ---
                "s1_success": bool(s1_success),
                "s1_runtime_sec": float(s1_runtime),
                "retrieved_cluster_id": int(cluster_id),
                "retrieved_dyn_id": int(dyn_id),
                "retrieved_map_idx": int(map_idx),
                "A_dist_cluster": float(d_cluster),
                "dyn_distance": float(dyn_dist),
                "cosine_situation": float(cos_sim),
                "collision_free_wrt_query_rects": None if s1_cf_query is None else bool(s1_cf_query),
                "goal_reached_wrt_query_goal": None if s1_gr_query is None else bool(s1_gr_query),
                "success_db": None if traj is None else traj.get("success_db", None),
                "runtime_sec_db": None if traj is None else traj.get("runtime_sec_db", None),
            }
        )

    Path(out_json).write_text(json.dumps(results, indent=2))
    print(f"\n✅ wrote: {out_json} (N={len(results)})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_json", type=str, required=True)
    ap.add_argument("--traj_npz", type=str, required=True)
    ap.add_argument("--scenarios", type=str, required=True)
    ap.add_argument("--out", type=str, default="maze/S1_S2_mpc_results_maze.json")

    # ---- S1 params ----
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--collision_margin", type=float, default=0.0)
    ap.add_argument("--s1_goal_tol", type=float, default=0.6)

    ap.add_argument("--grid_n", type=int, default=25)
    ap.add_argument("--dt_nom", type=float, default=0.05)
    ap.add_argument("--n_steps_nom", type=int, default=200)
    ap.add_argument("--u_max_nom", type=float, default=3.0)
    ap.add_argument("--buffer_cells", type=int, default=2)
    ap.add_argument("--stop_tol", type=float, default=0.6)

    # ---- S2 MPC params ----
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=800)
    ap.add_argument("--n_horizon", type=int, default=20)
    ap.add_argument("--wall_margin", type=float, default=0.2)
    ap.add_argument("--smooth_kappa", type=float, default=20.0)
    ap.add_argument("--s2_goal_tol", type=float, default=0.5)

    args = ap.parse_args()

    run_hybrid(
        db_json=args.db_json,
        traj_npz_path=args.traj_npz,
        scenarios_json=args.scenarios,
        out_json=args.out,
        omega0=args.omega0,
        collision_margin=args.collision_margin,
        s1_goal_tol=args.s1_goal_tol,
        grid_n=args.grid_n,
        dt_nom=args.dt_nom,
        n_steps_nom=args.n_steps_nom,
        u_max_nom=args.u_max_nom,
        buffer_cells=args.buffer_cells,
        stop_tol=args.stop_tol,
        dt=args.dt,
        n_steps=args.n_steps,
        n_horizon=args.n_horizon,
        wall_margin=args.wall_margin,
        smooth_kappa=args.smooth_kappa,
        s2_goal_tol=args.s2_goal_tol,
    )


if __name__ == "__main__":
    main()
