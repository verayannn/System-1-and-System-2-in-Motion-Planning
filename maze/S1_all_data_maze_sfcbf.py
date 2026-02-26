"""
S1_all_data_maze_sfcbf.py — Run SFCBF for ALL (dyn_id × map_idx) in S1_database_maze.json,
and SAVE ONLY SUCCESS trajectories as the System-1 trajectory DB.

Input DB JSON from S1_layers_maze.py:
- payload["db"]["dyn_nodes"][dyn_id]["A"], ["B"]
- payload["shared_maps"][map_idx]["rectangles"], start, goal, bounds

Output NPZ keys (ONLY successful cases):
  - states: (Ns, T+1, 2) float32
  - inputs: (Ns, T,   m) float32
  - dyn_id: (Ns,) int32
  - map_idx: (Ns,) int32
  - runtime_sec: (Ns,) float64
  - success: (Ns,) int8  (all 1)
  - collision_free: (Ns,) int8
  - goal_reached: (Ns,) int8

Run:
  python maze/S1_all_data_maze_sfcbf.py \
    --db_json maze/S1_database_maze.json \
    --out_npz maze/s1_sfcbf_success_trajs.npz \
    --dt 0.05 --n_steps 800 --margin 0.35 --gamma 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import cvxpy as cp

Rect = Tuple[float, float, float, float]


# ----------------------------
# Smooth geometry primitives
# ----------------------------

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


# ----------------------------
# Nominal controller (simple LS to goal)
# ----------------------------

def u_nominal(x: np.ndarray, goal: np.ndarray, A: np.ndarray, B: np.ndarray, u_max: float) -> np.ndarray:
    x = x.reshape(2)
    goal = goal.reshape(2)
    v = -(x - goal)
    rhs = v - A @ x
    u, *_ = np.linalg.lstsq(B, rhs, rcond=None)
    u = np.clip(u, -u_max, u_max)
    return u.reshape(-1)


# ----------------------------
# CBF-QP Safety filter
# ----------------------------

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
    m = B.shape[1]

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


# ----------------------------
# Metrics
# ----------------------------

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


# ----------------------------
# Simulation (STOP when goal reached)
# ----------------------------

def simulate_sfcbf(
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    *,
    dt: float,
    n_steps: int,
    u_max: float,
    margin: float,
    gamma: float,
    goal_tol: float,
    collision_margin: float,
) -> Dict[str, Any]:
    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)
    m = B.shape[1]

    x = np.array(start, dtype=float).reshape(2)
    g = np.array(goal, dtype=float).reshape(2)

    X = [x.copy()]
    U: List[np.ndarray] = []

    t0 = time.perf_counter()
    ok = True

    for _ in range(int(n_steps)):
        # stop early if reached
        if float(np.sum((x - g) ** 2)) <= float(goal_tol) ** 2:
            break

        u0 = u_nominal(x, g, A, B, u_max=u_max)
        u1, feas = cbf_filter_qp(
            x, u0, A, B, rects,
            margin=margin, gamma=gamma, u_max=u_max
        )
        if not feas:
            # still simulate nominal, but mark not-ok (so we can filter)
            ok = False

        xdot = A @ x + B @ u1
        x = x + dt * xdot

        X.append(x.copy())
        U.append(u1.copy())

    rt = float(time.perf_counter() - t0)
    Xn = np.array(X, dtype=float)
    Un = np.array(U, dtype=float).reshape(-1, m)

    cf = bool(ok) and collision_free_rectangles(Xn, rects, margin=collision_margin)
    gr = goal_reached(Xn, goal, tol=goal_tol)
    succ = bool(cf and gr)

    return {
        "states": Xn,
        "inputs": Un,
        "runtime_sec": rt,
        "success": succ,
        "collision_free": bool(cf),
        "goal_reached": bool(gr),
    }


# ----------------------------
# Runner over DB (keep only success)
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_json", type=str, default="maze/S1_database_maze.json")
    ap.add_argument("--out_npz", type=str, default="maze/s1_sfcbf_success_trajs.npz")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=800)
    ap.add_argument("--u_max", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=0.35)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--goal_tol", type=float, default=0.6)
    ap.add_argument("--collision_margin", type=float, default=0.05)
    ap.add_argument("--max_dyn", type=int, default=None)
    ap.add_argument("--max_maps", type=int, default=None)
    args = ap.parse_args()

    payload = json.loads(Path(args.db_json).read_text())
    dyn_nodes = payload["db"]["dyn_nodes"]
    shared_maps = payload["shared_maps"]

    dyn_ids = sorted([int(k) for k in dyn_nodes.keys()])
    if args.max_dyn is not None:
        dyn_ids = dyn_ids[: int(args.max_dyn)]

    map_idxs = list(range(len(shared_maps)))
    if args.max_maps is not None:
        map_idxs = map_idxs[: int(args.max_maps)]

    # collect ONLY successes
    states_list: List[np.ndarray] = []
    inputs_list: List[np.ndarray] = []
    dyn_list: List[int] = []
    map_list: List[int] = []
    rt_list: List[float] = []
    cf_list: List[int] = []
    gr_list: List[int] = []

    # to store variable-length trajectories, we pad to max length after we know it
    tmp_states: List[np.ndarray] = []
    tmp_inputs: List[np.ndarray] = []
    tmp_meta: List[Tuple[int, int, float, int, int]] = []  # did, mi, rt, cf, gr

    total = len(dyn_ids) * len(map_idxs)
    done = 0
    succ_count = 0

    for did in dyn_ids:
        node = dyn_nodes[str(did)]
        A = np.array(node["A"], dtype=float)
        B = np.array(node["B"], dtype=float)

        for mi in map_idxs:
            mp = shared_maps[mi]
            rects = [tuple(map(float, r)) for r in mp["rectangles"]]
            start = tuple(map(float, mp["start"]))
            goal = tuple(map(float, mp["goal"]))
            u_max = float(node.get("u_max", args.u_max))

            out = simulate_sfcbf(
                A=A, B=B, rects=rects, start=start, goal=goal,
                dt=args.dt, n_steps=args.n_steps,
                u_max=u_max, margin=args.margin, gamma=args.gamma,
                goal_tol=args.goal_tol, collision_margin=args.collision_margin,
            )

            done += 1
            if out["success"]:
                succ_count += 1
                tmp_states.append(out["states"])
                tmp_inputs.append(out["inputs"])
                tmp_meta.append((did, mi, out["runtime_sec"], int(out["collision_free"]), int(out["goal_reached"])))

            if done % 50 == 0 or done == total:
                print(f"[sfcbf] {done}/{total} | success_kept={succ_count}")

    if succ_count == 0:
        raise RuntimeError("No successful trajectories found. Loosen margin/gamma/goal_tol or reduce obstacle density.")

    # pad to common length
    m = tmp_inputs[0].shape[1] if tmp_inputs[0].ndim == 2 else 2
    max_Tp1 = max(s.shape[0] for s in tmp_states)
    max_T = max_Tp1 - 1

    Ns = succ_count
    states_arr = np.zeros((Ns, max_Tp1, 2), dtype=np.float32)
    inputs_arr = np.zeros((Ns, max_T, m), dtype=np.float32)

    for k in range(Ns):
        S = tmp_states[k].astype(np.float32)
        U = tmp_inputs[k].astype(np.float32)
        Tp1 = S.shape[0]
        T = U.shape[0]
        states_arr[k, :Tp1, :] = S
        inputs_arr[k, :T, :] = U

        did, mi, rt, cf, gr = tmp_meta[k]
        dyn_list.append(int(did))
        map_list.append(int(mi))
        rt_list.append(float(rt))
        cf_list.append(int(cf))
        gr_list.append(int(gr))

    np.savez_compressed(
        args.out_npz,
        states=states_arr,
        inputs=inputs_arr,
        dyn_id=np.array(dyn_list, dtype=np.int32),
        map_idx=np.array(map_list, dtype=np.int32),
        runtime_sec=np.array(rt_list, dtype=np.float64),
        success=np.ones((Ns,), dtype=np.int8),
        collision_free=np.array(cf_list, dtype=np.int8),
        goal_reached=np.array(gr_list, dtype=np.int8),
    )
    print("[ok] wrote:", args.out_npz)
    print(f"[summary] kept {Ns} successful trajectories out of {total} ({100.0*Ns/total:.1f}%)")


if __name__ == "__main__":
    main()
