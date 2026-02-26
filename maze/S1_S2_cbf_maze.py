"""
S1_S2_cbf_maze.py — Hybrid "S1 if works else S2" pipeline for MAZE (rectangles)

What it does
------------
For each benchmark scenario:
  1) Run System-1 retrieval (same logic as S1_usage_maze.py):
       cluster by A-distance -> dyn by fast M-distance -> map by situation cosine
       then load the stored trajectory (states, inputs) from NPZ and re-check
       collision + goal w.r.t. the *QUERY* rectangles/goal.
  2) If S1 success == True: output S1 trajectory directly.
  3) Else: run System-2 SFCBF safety filter (QP CBF) BUT using the S1 trajectory
       as a *reference trajectory* for the nominal controller.

Notes on the reference-tracking safety filter
--------------------------------------------
System-2 here is still a safety filter:
  - You give a nominal input u_nom (now tracking the S1 reference path),
  - then project it onto CBF-safe controls via a QP.

Nominal tracking policy used here:
  - Let x_ref[k] be the k-th reference state (if available).
  - Build u_ls that tries to move toward x_ref[k] via least-squares:
        v = -(x - x_ref[k]), solve B u ≈ v - A x
  - If reference inputs u_ref[k] exist, blend them:
        u_nom = (1-w)*u_ls + w*u_ref[k]   with w=0.7
  - Then run the same CBF QP as in S2_cbf_maze.py.

Run
---
python maze/S1_S2_cbf_maze.py \
  --db_json maze/S1_database_maze.json \
  --traj_npz maze/s1_sfcbf_success_trajs.npz \
  --scenarios maze/benchmark_scenarios_maze.json \
  --out maze/S1_S2_cbf_results_maze.json \
  --dt 0.05 --n_steps 800 --margin 0.35 --gamma 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

Rect = Tuple[float, float, float, float]


# ============================================================
# System-1 retrieval (copied/trimmed from S1_usage_maze.py)
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
    """
    Nominal rollout (no obstacles) -> visited cells.
    Corridor = dilated visited.
    Obstacle occupancy grid.
    Area = visited OR (obstacles & corridor).
    Returns flattened binary vector (uint8).
    """
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


# ============================================================
# Shared metrics (kept identical to S1/S2 files)
# ============================================================

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
# System-2 SFCBF safety filter (adapted from S2_cbf_maze.py)
# ============================================================

def smooth_abs(z: float, eps: float = 1e-6) -> float:
    return float(math.sqrt(z * z + eps))


def smooth_max(z: float, eps: float = 1e-6) -> float:
    return float(0.5 * (z + math.sqrt(z * z + eps)))


def dist_outside_rect_and_grad(x: np.ndarray, rect: Rect, eps: float = 1e-6) -> Tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(2)
    xmin, ymin, xmax, ymax = map(float, rect)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)

    zx = x[0] - cx
    zy = x[1] - cy

    ax = smooth_abs(zx, eps)
    ay = smooth_abs(zy, eps)

    dx_raw = ax - hx
    dy_raw = ay - hy

    dx = smooth_max(dx_raw, eps)
    dy = smooth_max(dy_raw, eps)

    d = math.sqrt(dx * dx + dy * dy + eps)

    dax_dzx = zx / max(ax, 1e-12)
    day_dzy = zy / max(ay, 1e-12)

    sdx = math.sqrt(dx_raw * dx_raw + eps)
    sdy = math.sqrt(dy_raw * dy_raw + eps)
    ddx_ddxraw = 0.5 * (1.0 + dx_raw / max(sdx, 1e-12))
    ddy_ddyraw = 0.5 * (1.0 + dy_raw / max(sdy, 1e-12))

    dd_ddx = dx / max(d, 1e-12)
    dd_ddy = dy / max(d, 1e-12)

    grad = np.zeros(2, dtype=float)
    grad[0] = dd_ddx * ddx_ddxraw * dax_dzx
    grad[1] = dd_ddy * ddy_ddyraw * day_dzy
    return float(d), grad


def u_ls_to_target(x: np.ndarray, target: np.ndarray, A: np.ndarray, B: np.ndarray, u_max: float) -> np.ndarray:
    """Least-squares nominal (same structure as S2 u_nominal), but toward an arbitrary target."""
    x = np.asarray(x, dtype=float).reshape(2)
    target = np.asarray(target, dtype=float).reshape(2)
    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)

    v = -(x - target)
    rhs = v - A @ x
    u, *_ = np.linalg.lstsq(B, rhs, rcond=None)
    u = np.clip(u, -float(u_max), float(u_max))
    return u.reshape(-1)


def cbf_filter_qp(
    x: np.ndarray,
    u_nom: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    *,
    margin: float,
    gamma: float,
    u_max: float,
    eps_geom: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=float).reshape(2)
    u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)
    m = int(B.shape[1])

    u = cp.Variable(m)
    obj = cp.Minimize(cp.sum_squares(u - u_nom))
    cons = [u <= float(u_max), u >= -float(u_max)]

    Ax = A @ x
    for rect in rects:
        d, grad_d = dist_outside_rect_and_grad(x, rect, eps=eps_geom)
        h = float(d - margin)
        cons.append(grad_d @ (Ax + B @ u) >= -float(gamma) * h)

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return u_nom.copy(), False
        return np.array(u.value).reshape(m), True
    except Exception:
        return u_nom.copy(), False


def simulate_sfcbf_with_reference(
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    *,
    ref_states: Optional[np.ndarray],
    ref_inputs: Optional[np.ndarray],
    ref_blend_w: float,
    dt: float,
    n_steps: int,
    u_max: float,
    margin: float,
    gamma: float,
    goal_tol: float,
    collision_margin: float,
) -> Dict[str, Any]:
    """SFCBF simulation where the nominal input tracks an S1 reference trajectory."""
    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)
    m = int(B.shape[1])

    x = np.array(start, dtype=float).reshape(2)
    g = np.array(goal, dtype=float).reshape(2)

    X = [x.copy()]
    U: List[np.ndarray] = []

    ok = True
    t0 = time.perf_counter()

    # sanitize ref
    rs = None if ref_states is None else np.asarray(ref_states, dtype=float)
    ru = None if ref_inputs is None else np.asarray(ref_inputs, dtype=float)
    if rs is not None and rs.ndim != 2:
        rs = None
    if ru is not None and ru.ndim != 2:
        ru = None

    for k in range(int(n_steps)):
        # early stop if reached actual goal
        if float(np.sum((x - g) ** 2)) <= float(goal_tol) ** 2:
            break

        # choose tracking target
        if rs is not None and len(rs) > 0:
            tgt = rs[min(k, len(rs) - 1), :2]
        else:
            tgt = g

        u_ls = u_ls_to_target(x, tgt, A, B, u_max=u_max)

        if ru is not None and len(ru) > 0:
            u_ref = ru[min(k, len(ru) - 1), :]
            u_ref = np.clip(u_ref.reshape(-1), -float(u_max), float(u_max))
            w = float(np.clip(ref_blend_w, 0.0, 1.0))
            u0 = (1.0 - w) * u_ls + w * u_ref
        else:
            u0 = u_ls

        u1, feas = cbf_filter_qp(x, u0, A, B, rects, margin=margin, gamma=gamma, u_max=u_max)
        if not feas:
            ok = False

        xdot = A @ x + B @ u1
        x = x + float(dt) * xdot

        X.append(x.copy())
        U.append(u1.copy())

    runtime = float(time.perf_counter() - t0)
    Xn = np.array(X, dtype=float)
    Un = np.array(U, dtype=float).reshape(-1, m)

    cf = bool(ok) and collision_free_rectangles(Xn, rects, margin=collision_margin)
    gr = goal_reached(Xn, goal, tol=goal_tol)
    succ = bool(cf and gr)

    return {
        "states": Xn.tolist(),
        "inputs": Un.tolist(),
        "runtime_sec": runtime,
        "success": succ,
        "collision_free": bool(cf),
        "goal_reached": bool(gr),
        "ok_qp_all_steps": bool(ok),
        "T_steps_executed": int(len(Xn) - 1),
    }


# ============================================================
# Hybrid runner
# ============================================================

def run_hybrid(
    *,
    db_json: str,
    traj_npz_path: str,
    scenarios_json: str,
    out_json: str,
    # S1 knobs
    omega0: float = 1.0,
    collision_margin_s1: float = 0.0,
    goal_tol: float = 0.6,
    grid_n: int = 25,
    dt_nom: float = 0.05,
    n_steps_nom: int = 200,
    u_max_nom: float = 3.0,
    buffer_cells: int = 2,
    stop_tol: float = 0.6,
    # S2 knobs
    dt: float = 0.05,
    n_steps: int = 800,
    u_max_default: float = 3.0,
    margin: float = 0.35,
    gamma: float = 2.0,
    collision_margin_s2: float = 0.05,
    ref_blend_w: float = 0.7,
) -> List[Dict[str, Any]]:
    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]

    traj_npz = dict(np.load(traj_npz_path, allow_pickle=True))
    traj_index = _build_index(traj_npz)

    scenarios = json.loads(Path(scenarios_json).read_text())
    if not isinstance(scenarios, list):
        raise TypeError("Expected scenarios JSON to be a list of scenario dicts.")

    results: List[Dict[str, Any]] = []
    total_t0 = time.perf_counter()

    for k, sc in enumerate(scenarios):
        scenario_id = int(sc.get("scenario_id", k))

        A_query = np.array(sc.get("A_query", sc.get("A")), dtype=float)
        B_query = np.array(sc.get("B_query", sc.get("B")), dtype=float)
        if A_query is None or B_query is None:
            raise KeyError(f"Scenario {scenario_id} missing A_query/B_query (or A/B).")

        rects = [tuple(map(float, r)) for r in sc["rectangles"]]
        bounds = tuple(map(float, sc.get("bounds", [-10.0, -10.0, 10.0, 10.0])))
        start = tuple(map(float, sc.get("start", (5.0, 5.0))))
        goal = tuple(map(float, sc.get("goal", (0.0, 0.0))))

        # scenario may override u_max
        u_max = float(sc.get("u_max", u_max_default))

        # ---------------- S1 retrieval ----------------
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
        s1_cf = None
        s1_gr = None
        s1_states = None
        s1_inputs = None

        if traj is not None and traj.get("states") is not None:
            s1_states = np.asarray(traj["states"], dtype=float)
            s1_inputs = np.asarray(traj["inputs"], dtype=float)
            s1_cf = collision_free_rectangles(s1_states, rects, margin=float(collision_margin_s1))
            s1_gr = goal_reached(s1_states, goal, tol=float(goal_tol))
            s1_success = bool(s1_cf and s1_gr)

        # ---------------- S2 fallback if needed ----------------
        if s1_success:
            mode = "S1"
            out = {
                "states": s1_states.tolist() if s1_states is not None else None,
                "inputs": s1_inputs.tolist() if s1_inputs is not None else None,
                "runtime_sec": s1_runtime,
                "success": True,
                "collision_free": bool(s1_cf),
                "goal_reached": bool(s1_gr),
                "ok_qp_all_steps": None,
                "T_steps_executed": int(len(s1_states) - 1) if s1_states is not None else None,
            }
            s2_runtime = 0.0
        else:
            mode = "S2"
            t_s2 = time.perf_counter()
            out = simulate_sfcbf_with_reference(
                A=A_query,
                B=B_query,
                rects=rects,
                start=start,
                goal=goal,
                ref_states=s1_states,
                ref_inputs=s1_inputs,
                ref_blend_w=float(ref_blend_w),
                dt=float(dt),
                n_steps=int(n_steps),
                u_max=float(u_max),
                margin=float(margin),
                gamma=float(gamma),
                goal_tol=float(goal_tol),
                collision_margin=float(collision_margin_s2),
            )
            s2_runtime = float(time.perf_counter() - t_s2)

        results.append(
            {
                "scenario_id": scenario_id,
                "mode": mode,
                # retrieval ids
                "retrieved_cluster_id": int(cluster_id),
                "retrieved_dyn_id": int(dyn_id),
                "retrieved_map_idx": int(map_idx),
                # retrieval diagnostics
                "A_dist_cluster": float(d_cluster),
                "dyn_distance": float(dyn_dist),
                "cosine_situation": float(cos_sim),
                # S1 check
                "s1_success": bool(s1_success),
                "collision_free_wrt_query_rects": None if s1_cf is None else bool(s1_cf),
                "goal_reached_wrt_query_goal": None if s1_gr is None else bool(s1_gr),
                "success_db": None if traj is None else traj.get("success_db", None),
                "runtime_sec_db": None if traj is None else traj.get("runtime_sec_db", None),
                # timings
                "runtime_s1_sec": float(s1_runtime),
                "runtime_s2_sec": float(s2_runtime),
                "runtime_total_sec": float(s1_runtime + s2_runtime),
                # final output
                **out,
            }
        )

        if (k + 1) % 10 == 0 or (k + 1) == len(scenarios):
            succ = sum(1 for r in results if r["success"])
            print(
                f"[S1->S2-SFCBF] {k+1}/{len(scenarios)} done | success={succ} "
                f"({100.0*succ/max(1,len(results)):.1f}%)"
            )

    total_rt = float(time.perf_counter() - total_t0)
    Path(out_json).write_text(json.dumps(results, indent=2))
    print("[ok] wrote:", out_json)
    print(f"[summary] N={len(results)} total_wall_time_sec={total_rt:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_json", type=str, required=True)
    ap.add_argument("--traj_npz", type=str, required=True)
    ap.add_argument("--scenarios", type=str, required=True)
    ap.add_argument("--out", type=str, default="maze/S1_S2_cbf_results_maze.json")

    # S1 knobs
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--collision_margin_s1", type=float, default=0.0)
    ap.add_argument("--goal_tol", type=float, default=0.6)
    ap.add_argument("--grid_n", type=int, default=25)
    ap.add_argument("--dt_nom", type=float, default=0.05)
    ap.add_argument("--n_steps_nom", type=int, default=200)
    ap.add_argument("--u_max_nom", type=float, default=3.0)
    ap.add_argument("--buffer_cells", type=int, default=2)
    ap.add_argument("--stop_tol", type=float, default=0.6)

    # S2 knobs
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=800)
    ap.add_argument("--u_max", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=0.35)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--collision_margin_s2", type=float, default=0.05)
    ap.add_argument("--ref_blend_w", type=float, default=0.7)

    args = ap.parse_args()

    run_hybrid(
        db_json=args.db_json,
        traj_npz_path=args.traj_npz,
        scenarios_json=args.scenarios,
        out_json=args.out,
        omega0=args.omega0,
        collision_margin_s1=args.collision_margin_s1,
        goal_tol=args.goal_tol,
        grid_n=args.grid_n,
        dt_nom=args.dt_nom,
        n_steps_nom=args.n_steps_nom,
        u_max_nom=args.u_max_nom,
        buffer_cells=args.buffer_cells,
        stop_tol=args.stop_tol,
        dt=args.dt,
        n_steps=args.n_steps,
        u_max_default=args.u_max,
        margin=args.margin,
        gamma=args.gamma,
        collision_margin_s2=args.collision_margin_s2,
        ref_blend_w=args.ref_blend_w,
    )


if __name__ == "__main__":
    main()
