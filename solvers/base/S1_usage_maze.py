"""
S1_usage_maze.py — System-1 retrieval for SCATTER-RECTANGLE ("maze") benchmark

Pipeline (mirrors your random-circles S1_usage.py structure):

For each benchmark scenario:
  1) Select best cluster by A-distance to cluster representative (cd_dyn_id)
  2) Select best dyn within cluster by DIVERSITY metric:
       - compute minimal alpha (radians) such that {M_query, M_dyn} is alpha-alignable
       - pick dyn with smallest alpha  (more "similar" in the diversity sense)
  3) Select best map_idx within that dyn by situation similarity:
       - compute binary grid "area that matters" vector for query
       - cosine similarity with stored per-map vectors in DB
       - pick best map_idx (top-1)
  4) Retrieve stored trajectory from NPZ (dyn_id, map_idx)
  5) IMPORTANT: re-check collision against QUERY rectangles (not stored map),
     and optionally goal reached vs QUERY goal.

Inputs:
- DB JSON from your S1_layers_maze.py (must contain):
    payload["db"]["consensus_nodes"][cid]["cd_dyn_id"]
    payload["db"]["dyn_nodes"][dyn_id]["A"], ["B"], ["M"]
    payload["db"]["dyn_nodes"][dyn_id]["env_types"]["maze"]["situation_vecs"]  (list of vectors)
- Trajectory NPZ from your S1_all_data_maze_sfcbf.py (keys):
    states, inputs, dyn_id, map_idx, (optional) success, collision_free, goal_reached, runtime_sec
- Benchmark scenarios JSON from generate_benchmark_scenarios_maze.py:
    list of {scenario_id, A_query, B_query, rectangles, bounds, start, goal}

Run:
  python maze/S1_usage_maze.py \
    --db_json maze/S1_database_maze.json \
    --traj_npz maze/s1_sfcbf_success_trajs.npz \
    --scenarios maze/benchmark_scenarios_maze.json \
    --out maze/S1_results_maze.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cvxpy as cp


Rect = Tuple[float, float, float, float]


# ============================================================
# (A) A-distance for cluster selection (same spirit as your S1_usage.py)
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


# ============================================================
# (B) Diversity / alpha-alignment metric for dynamics selection
# ============================================================

# ============================================================
# (B) FAST approximate dynamics similarity (NO CVX)
# ============================================================

def system_to_M(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega0: float = 1.0) -> np.ndarray:
    n = A.shape[0]
    jwI_minus_A = 1j * omega0 * np.eye(n) - A
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
    """
    FAST approximate dynamics similarity:
        use Frobenius norm distance on M matrices.
    Smaller distance => more similar.
    """
    C = np.eye(2, dtype=float)
    M_q = system_to_M(A_query, B_query, C, omega0=float(omega0))
    M_q_flat = M_q.flatten()

    cd = db["consensus_nodes"][str(cluster_id)]

    best_dyn = None
    best_dist = float("inf")

    for dyn_id in cd["dyn_children"]:
        dyn_id = int(dyn_id)
        M_dyn = np.array(db["dyn_nodes"][str(dyn_id)]["M"], dtype=float)
        d = np.linalg.norm(M_q_flat - M_dyn.flatten())
        if d < best_dist:
            best_dist = d
            best_dyn = dyn_id

    return int(best_dyn), float(best_dist)

# ============================================================
# (C) Situation similarity (your cosine binary grid vector)
# ============================================================

def cosine_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = int(np.sum(a))
    bb = int(np.sum(b))
    if aa == 0 or bb == 0:
        return 0.0
    inter = int(np.sum((a.astype(bool) & b.astype(bool)).astype(np.uint8)))
    return float(inter) / float(np.sqrt(aa * bb))


def _dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    for _ in range(int(iters)):
        m = np.maximum.reduce([
            m,
            np.roll(m,  1, axis=0),
            np.roll(m, -1, axis=0),
            np.roll(m,  1, axis=1),
            np.roll(m, -1, axis=1),
        ]).astype(np.uint8)
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
    nominal rollout (no obstacles) -> visited cells
    corridor = dilated visited
    obstacle occupancy grid
    area = visited OR (obstacles & corridor)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    def u_nominal(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        v = -(x - g)
        rhs = v - A @ x
        u, *_ = np.linalg.lstsq(B, rhs, rcond=None)
        return np.clip(u, -u_max_nom, u_max_nom)

    xmin, ymin, xmax, ymax = map(float, bounds)
    x = np.array(start, dtype=float).reshape(2)
    g = np.array(goal, dtype=float).reshape(2)

    visited = np.zeros((grid_n, grid_n), dtype=np.uint8)

    for _ in range(int(n_steps_nom)):
        if float(np.sum((x - g) ** 2)) <= float(stop_tol) ** 2:
            break

        u = u_nominal(x, g)
        x = x + float(dt_nom) * (A @ x + B @ u)

        if not (xmin <= x[0] <= xmax and ymin <= x[1] <= ymax):
            continue

        j = int((x[0] - xmin) / (xmax - xmin) * grid_n)
        i = int((x[1] - ymin) / (ymax - ymin) * grid_n)
        i = int(np.clip(i, 0, grid_n - 1))
        j = int(np.clip(j, 0, grid_n - 1))
        visited[i, j] = 1

    corridor = _dilate4(visited, iters=int(buffer_cells))

    obs = np.zeros((grid_n, grid_n), dtype=np.uint8)
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

        obs[i1:i2 + 1, j1:j2 + 1] = 1

    area = np.maximum(visited, (obs & corridor).astype(np.uint8))
    return area.reshape(-1).astype(np.uint8)


def select_best_map_by_situation(
    v_query: np.ndarray,
    situation_vecs: List[List[int]],
) -> Tuple[int, float]:
    """
    FAST vectorized cosine similarity.
    """
    V_db = np.array(situation_vecs, dtype=np.uint8)   # (64, D)
    v_q = v_query.astype(np.uint8)

    norm_q = np.sqrt(np.sum(v_q))
    norm_db = np.sqrt(np.sum(V_db, axis=1))

    inter = np.sum(V_db & v_q, axis=1)

    cosines = inter / (norm_db * norm_q + 1e-12)

    best_idx = int(np.argmax(cosines))
    best_s = float(cosines[best_idx])

    return best_idx, best_s


# ============================================================
# (D) Trajectory retrieval (NPZ) + collision recheck
# ============================================================

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
    out = {
        "states": np.array(traj_npz["states"][i], dtype=float),
        "inputs": np.array(traj_npz["inputs"][i], dtype=float),
        "runtime_sec_db": float(traj_npz["runtime_sec"][i]) if "runtime_sec" in traj_npz else None,
        "success_db": bool(traj_npz["success"][i]) if "success" in traj_npz else None,
    }
    return out


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
# Run S1 on ALL scenarios
# ============================================================

def run_s1_on_scenarios(
    *,
    db_json: str,
    traj_npz_path: str,
    scenarios_json: str,
    out_json: str,
    omega0: float = 1.0,
    align_solver: str = "SCS",
    collision_margin: float = 0.0,
    goal_tol: float = 0.6,
    # situation-vector params (keep them explicit & controllable)
    grid_n: int = 25,
    dt_nom: float = 0.05,
    n_steps_nom: int = 200,
    u_max_nom: float = 3.0,
    buffer_cells: int = 2,
    stop_tol: float = 0.6,
) -> List[Dict[str, Any]]:
    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]

    traj_npz = dict(np.load(traj_npz_path, allow_pickle=True))
    traj_index = _build_index(traj_npz)

    scenarios = json.loads(Path(scenarios_json).read_text())
    results: List[Dict[str, Any]] = []

    for sc in scenarios:
        scenario_id = int(sc["scenario_id"])
        A_query = np.array(sc["A_query"], dtype=float)
        B_query = np.array(sc["B_query"], dtype=float)

        rects = [tuple(map(float, r)) for r in sc["rectangles"]]
        bounds = tuple(map(float, sc.get("bounds", [-10.0, -10.0, 10.0, 10.0])))
        start = tuple(map(float, sc.get("start", [5.0, 5.0])))
        goal = tuple(map(float, sc.get("goal", [0.0, 0.0])))

        t0 = time.perf_counter()

        # 1) cluster by A-distance
        cluster_id, d_cluster = select_best_cluster_A(A_query, db)

        # 2) dyn by diversity (minimal alpha) -- now by distance
        dyn_id, dyn_dist = select_best_dyn_fast(
            A_query, B_query, db, cluster_id,
            omega0=float(omega0)
        )


        # 3) best map by situation cosine
        dyn_node = db["dyn_nodes"][str(dyn_id)]
        situation_vecs = dyn_node["env_types"]["maze"]["situation_vecs"]

        v_q = compute_situation_vector(
            A_query, B_query, rects,
            bounds=bounds, start=start, goal=goal,
            grid_n=int(grid_n),
            dt_nom=float(dt_nom),
            n_steps_nom=int(n_steps_nom),
            u_max_nom=float(u_max_nom),
            buffer_cells=int(buffer_cells),
            stop_tol=float(stop_tol),
        )

        map_idx, cos_sim = select_best_map_by_situation(v_q, situation_vecs)

        # 4) retrieve traj
        traj = load_trajectory(traj_npz, traj_index, dyn_id, map_idx)

        runtime_sec = float(time.perf_counter() - t0)

        if traj is None or traj.get("states", None) is None:
            success = False
            cf_query = None
            gr_query = None
        else:
            states = traj["states"]
            cf_query = collision_free_rectangles(states, rects, margin=float(collision_margin))
            gr_query = goal_reached(states, goal, tol=float(goal_tol))
            success = bool(cf_query and gr_query)

###OUTPUT PROCESSING FROM HERE####
        results.append(
            {
                "scenario_id": scenario_id,

                # retrieved ids
                "retrieved_cluster_id": int(cluster_id),
                "retrieved_dyn_id": int(dyn_id),
                "retrieved_map_idx": int(map_idx),

                # debug distances/scores
                "A_dist_cluster": float(d_cluster),
                "dyn_distance": float(dyn_dist),
                "cosine_situation": float(cos_sim),

                # required outputs (like benchmark_report style)
                "success": bool(success),
                "runtime_sec": float(runtime_sec),

                # extra debug flags
                "collision_free_wrt_query_rects": None if cf_query is None else bool(cf_query),
                "goal_reached_wrt_query_goal": None if gr_query is None else bool(gr_query),
                "success_db": None if traj is None else traj.get("success_db", None),
                "runtime_sec_db": None if traj is None else traj.get("runtime_sec_db", None),
            }
        )

        print(
            f"[S1] sc={scenario_id} " 
            f"cluster={cluster_id} " #Solution ID from which we retrieved the traj - Step 1
            f"dyn={dyn_id} " #Solution ID from which we retrieved the traj - Step 2
            f"dyn_dist={dyn_dist:.4f} " #Confidence 3
            f"map={map_idx} " #Solution ID from which we retrieved the traj - Step 3
            f"cos={cos_sim:.3f} " #Confidence 4
            f"success={success}"
        )

        '''To retrive a solution:
            Retrieve stored trajectory from NPZ (dyn_id, map_idx)
        '''


    Path(out_json).write_text(json.dumps(results, indent=2))
    print(f"\nwrote: {out_json} (N={len(results)})")
    return results


def run_s1_single(
    *,
    db_json,
    traj_npz_path,
    scenario
):
    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]

    traj_npz = dict(np.load(traj_npz_path, allow_pickle=True))
    traj_index = _build_index(traj_npz)

    A_query = np.array(scenario["A_query"], dtype=float)
    B_query = np.array(scenario["B_query"], dtype=float)

    rects = [tuple(map(float, r)) for r in scenario["rectangles"]]
    bounds = tuple(map(float, scenario["bounds"]))
    start = tuple(map(float, scenario["start"]))
    goal = tuple(map(float, scenario["goal"]))

    # --- SAME pipeline ---
    cluster_id, _ = select_best_cluster_A(A_query, db)
    dyn_id, _ = select_best_dyn_fast(A_query, B_query, db, cluster_id)

    dyn_node = db["dyn_nodes"][str(dyn_id)]
    situation_vecs = dyn_node["env_types"]["maze"]["situation_vecs"]

    v_q = compute_situation_vector(
        A_query, B_query, rects,
        bounds=bounds, start=start, goal=goal,
        grid_n=25, dt_nom=0.05, n_steps_nom=200,
        u_max_nom=3.0, buffer_cells=2, stop_tol=0.6
    )

    map_idx, cos_sim = select_best_map_by_situation(v_q, situation_vecs)

    traj = load_trajectory(traj_npz, traj_index, dyn_id, map_idx)

    if traj is None:
        return None, 0.0

    states = traj["states"]

    success = (
        collision_free_rectangles(states, rects)
        and goal_reached(states, goal)
    )

    return states.tolist(), float(cos_sim), success

# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_json", type=str, required=True)
    p.add_argument("--traj_npz", type=str, required=True)
    p.add_argument("--scenarios", type=str, required=True)
    p.add_argument("--out", type=str, default="maze/S1_results_maze.json")

    # diversity metric options
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--align_solver", type=str, default="SCS")

    # recheck options
    p.add_argument("--collision_margin", type=float, default=0.0)
    p.add_argument("--goal_tol", type=float, default=0.6)

    # situation vector options
    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument("--dt_nom", type=float, default=0.05)
    p.add_argument("--n_steps_nom", type=int, default=200)
    p.add_argument("--u_max_nom", type=float, default=3.0)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)

    args = p.parse_args()

    ###SOLVING FROM HERE####
    run_s1_on_scenarios(
        db_json=args.db_json,
        traj_npz_path=args.traj_npz,
        scenarios_json=args.scenarios,
        out_json=args.out,
        omega0=args.omega0,
        align_solver=args.align_solver,
        collision_margin=args.collision_margin,
        goal_tol=args.goal_tol,
        grid_n=args.grid_n,
        dt_nom=args.dt_nom,
        n_steps_nom=args.n_steps_nom,
        u_max_nom=args.u_max_nom,
        buffer_cells=args.buffer_cells,
        stop_tol=args.stop_tol,
    )


if __name__ == "__main__":
    main()
