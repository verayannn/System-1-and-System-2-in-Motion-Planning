import json
import numpy as np
from pathlib import Path

from base.S1_usage_maze import (
    select_best_cluster_A,
    select_best_dyn_fast,
    compute_situation_vector,
    select_best_map_by_situation,
    load_trajectory,
    _build_index
)

# --------------------------------------------------
# CACHE (loaded once)
# --------------------------------------------------

_db = None
_traj_npz = None
_traj_index = None


def _goal_reached(states, goal, tol):
    states = np.asarray(states, dtype=float)
    goal = np.asarray(goal, dtype=float).reshape(2)
    if states.ndim != 2 or states.shape[0] == 0:
        return False
    return float(np.linalg.norm(states[-1, :2] - goal)) <= float(tol)


def _point_collision_free(x, rects, margin=0.0):
    px, py = float(x[0]), float(x[1])
    m = float(margin)
    for xmin, ymin, xmax, ymax in rects:
        if xmin - m <= px <= xmax + m and ymin - m <= py <= ymax + m:
            return False
    return True


def _extend_to_goal_if_needed(
    states,
    A,
    B,
    rects,
    bounds,
    goal,
    *,
    goal_tol,
    dt=0.05,
    u_max=3.0,
    max_extra_steps=120,
):
    """Tighten a retrieved primitive to the query goal tolerance.

    Older primitive databases were generated with a looser stop tolerance
    (0.6), while the benchmark success criterion is often 0.5. This appends a
    short nominal tail under the query dynamics instead of marking an otherwise
    good retrieved primitive as failed solely because it stopped just outside
    the benchmark tolerance.
    """
    out = np.asarray(states, dtype=float)
    if _goal_reached(out, goal, goal_tol):
        return out

    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)
    goal = np.asarray(goal, dtype=float).reshape(2)
    x = out[-1, :2].astype(float).copy()
    xmin, ymin, xmax, ymax = map(float, bounds)

    appended = []
    for _ in range(int(max_extra_steps)):
        rhs = -(x - goal) - A @ x
        try:
            u, *_ = np.linalg.lstsq(B, rhs, rcond=None)
        except np.linalg.LinAlgError:
            break
        u = np.clip(u.reshape(-1), -float(u_max), float(u_max))
        x_next = x + float(dt) * (A @ x + B @ u)

        if not (xmin <= x_next[0] <= xmax and ymin <= x_next[1] <= ymax):
            break
        if not _point_collision_free(x_next, rects, margin=0.0):
            break

        appended.append(x_next.copy())
        x = x_next

        if float(np.linalg.norm(x - goal)) <= float(goal_tol):
            break

    if appended:
        out = np.vstack([out[:, :2], np.asarray(appended, dtype=float)])
    return out


def _init():
    global _db, _traj_npz, _traj_index

    if _db is None:
        _db = json.loads(
            Path("db/S1_database_maze.json").read_text()
        )["db"]

        _traj_npz = dict(
            np.load("db/s1_sfcbf_success_trajs.npz", allow_pickle=True)
        )

        _traj_index = _build_index(_traj_npz)


# --------------------------------------------------
# MAIN ENTRYPOINT (SINGLE SCENARIO)
# --------------------------------------------------

def solveMotionPrimitives(scenario):
    """
    Input:
        scenario (dict)

    Output:
        states (list) or None
        confidence (float)
    """
    _init()

    if isinstance(scenario, dict):
        A_query = np.array(scenario["A_query"], dtype=float)
        B_query = np.array(scenario["B_query"], dtype=float)
        rects = scenario["rectangles"]
        bounds = tuple(scenario["bounds"])
        start = tuple(scenario["start"])
        goal = tuple(scenario["goal"])
        u_max = float(scenario.get("u_max", 3.0))
        goal_tol = float(scenario.get("goal_tol", 0.5))
    else:
        A_query = np.array(scenario.A, dtype=float)
        B_query = np.array(scenario.B, dtype=float)
        rects = scenario.rects
        bounds = tuple(scenario.bounds)
        start = tuple(scenario.start)
        goal = tuple(scenario.goal)
        u_max = float(getattr(scenario, "u_max", 3.0))
        goal_tol = float(getattr(scenario, "goal_tol", 0.5))

    # --- pipeline ---
    cluster_id, _ = select_best_cluster_A(A_query, _db)
    dyn_id, _ = select_best_dyn_fast(A_query, B_query, _db, cluster_id)

    dyn_node = _db["dyn_nodes"][str(dyn_id)]
    situation_vecs = dyn_node["env_types"]["maze"]["situation_vecs"]

    v_q = compute_situation_vector(
        A_query, B_query, rects,
        bounds=bounds, start=start, goal=goal,
        grid_n=25,
        dt_nom=0.05,
        n_steps_nom=200,
        u_max_nom=3.0,
        buffer_cells=2,
        stop_tol=0.6
    )

    map_idx, cos_sim = select_best_map_by_situation(v_q, situation_vecs)

    traj = load_trajectory(_traj_npz, _traj_index, dyn_id, map_idx)

    if traj is None:
        return None, 0.0

    states = _extend_to_goal_if_needed(
        traj["states"],
        A_query,
        B_query,
        rects,
        bounds,
        goal,
        goal_tol=goal_tol,
        u_max=u_max,
    )

    return states, float(cos_sim)
