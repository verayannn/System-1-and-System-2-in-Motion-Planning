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
    else:
        A_query = np.array(scenario.A, dtype=float)
        B_query = np.array(scenario.B, dtype=float)
        rects = scenario.rects
        bounds = tuple(scenario.bounds)
        start = tuple(scenario.start)
        goal = tuple(scenario.goal)

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

    return traj["states"], float(cos_sim)