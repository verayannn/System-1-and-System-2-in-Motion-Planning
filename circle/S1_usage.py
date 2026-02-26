"""
S1_usage.py — System-1 retrieval on a list of benchmark scenarios (OPTION A)

OPTION A = do similarity search using REAL dynamics matrix A (not M).

What it does:
- Loads S1 DB JSON (must contain per-dyn "A" and per-cluster "cd_dyn_id")
- Loads precomputed trajectories NPZ from S1_all_data.py
- Loads benchmark scenarios JSON (from generate_new_scenarios.py; must contain "A_query" and "obstacle_center")
- For each scenario:
    1) Select best cluster by distance(A_query, A_cd) where A_cd comes from cd_dyn_id
    2) Select best dyn within cluster by distance(A_query, A_dyn)
    3) Select closest obstacle instance (by Euclidean distance to scenario obstacle_center)
    4) Retrieve stored trajectory (states/inputs/cost) from NPZ
    5) IMPORTANT: Re-check collision against the *scenario (query) obstacle center*
       because the retrieved trajectory was generated for a *stored* obstacle position.

Outputs:
- JSON list compatible with benchmark_report.py (scenario_id, success, cost, runtime_sec)
- plus extra debug fields (retrieved ids, distances, collision flags, etc.)

Run:
  python circle/S1_usage.py \
    --db_json circle/S1_database_single_obstacle.json \
    --traj_npz circle/s1_mpc_trajectories.npz \
    --scenarios circle/benchmark_scenarios.json \
    --out circle/S1_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# (A) Similarity metric on A (REAL)
# ============================================================

def A_distance(A_query: np.ndarray, A_ref: np.ndarray, normalize: bool = True) -> float:
    """
    Frobenius distance between A matrices.
    If normalize=True: ||Aq - Ar||_F / (||Ar||_F + eps)
    """
    Aq = np.asarray(A_query, dtype=float)
    Ar = np.asarray(A_ref, dtype=float)
    d = float(np.linalg.norm(Aq - Ar, ord="fro"))
    if not normalize:
        return d
    denom = float(np.linalg.norm(Ar, ord="fro")) + 1e-12
    return d / denom


# ============================================================
# Selection using A
# ============================================================

def select_best_cluster_A(A_query: np.ndarray, db: Dict[str, Any]) -> Tuple[int, float]:
    """
    Pick cluster whose consensus representative A (via cd_dyn_id) is closest to A_query.
    """
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


def select_best_dyn_A(A_query: np.ndarray, db: Dict[str, Any], cluster_id: int) -> Tuple[int, float]:
    """
    Within cluster, pick dyn_id whose A is closest to A_query.
    """
    cd = db["consensus_nodes"][str(cluster_id)]
    best_dyn: Optional[int] = None
    best_d = float("inf")

    for dyn_id in cd["dyn_children"]:
        dyn_id = int(dyn_id)
        A_dyn = np.array(db["dyn_nodes"][str(dyn_id)]["A"], dtype=float)
        d = A_distance(A_query, A_dyn, normalize=True)
        if d < best_d:
            best_d = d
            best_dyn = dyn_id

    if best_dyn is None:
        raise RuntimeError(f"No dyn children found for cluster_id={cluster_id}.")
    return best_dyn, float(best_d)


def select_closest_obstacle(
    query_center: Tuple[float, float],
    dyn_node: Dict[str, Any],
    obs_type: str = "car",
) -> Tuple[int, Tuple[float, float], float]:
    """
    Pick the stored obstacle instance closest to query_center.
    Returns: (inst_idx, stored_center, dist)
    """
    if "obs_types" not in dyn_node or obs_type not in dyn_node["obs_types"]:
        raise KeyError(f"dyn_node missing obs_type='{obs_type}'")

    qc = np.array([float(query_center[0]), float(query_center[1])], dtype=float)

    best_idx: Optional[int] = None
    best_center: Optional[Tuple[float, float]] = None
    best_dist = float("inf")

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
    return best_idx, best_center, float(best_dist)


# ============================================================
# Trajectory retrieval (from NPZ)
# ============================================================

def _build_index(traj_npz: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    dyn_ids = np.array(traj_npz["dyn_id"]).astype(int)
    insts = np.array(traj_npz["instance_idx"]).astype(int)
    idx: Dict[Tuple[int, int], int] = {}
    for i in range(len(dyn_ids)):
        idx[(int(dyn_ids[i]), int(insts[i]))] = int(i)
    return idx


def load_trajectory(
    traj_npz: Dict[str, Any],
    index: Dict[Tuple[int, int], int],
    dyn_id: int,
    inst_idx: int,
) -> Optional[Dict[str, Any]]:
    key = (int(dyn_id), int(inst_idx))
    if key not in index:
        return None
    i = index[key]
    out = {
        "states": np.array(traj_npz["states"][i], dtype=float),
        "inputs": np.array(traj_npz["inputs"][i], dtype=float),
        "cost": float(traj_npz["cost"][i]) if "cost" in traj_npz else None,
        "runtime_sec": float(traj_npz["runtime_sec"][i]) if "runtime_sec" in traj_npz else None,
        "success_db": bool(traj_npz["success"][i]) if "success" in traj_npz else None,
    }
    return out


# ============================================================
# Collision re-check against *query* obstacle
# ============================================================

def collision_free_against_query_obstacle(
    states: np.ndarray,
    obstacle_center: Tuple[float, float],
    obstacle_radius: float,
    margin: float = 0.0,
) -> bool:
    """
    True if trajectory never enters the disk around the QUERY obstacle center.
    """
    xs = np.asarray(states[:, 0], dtype=float)
    ys = np.asarray(states[:, 1], dtype=float)
    cx, cy = float(obstacle_center[0]), float(obstacle_center[1])
    r = float(obstacle_radius) + float(margin)
    return not bool(np.any((xs - cx) ** 2 + (ys - cy) ** 2 < r * r))


# ============================================================
# System-1 over scenarios (OPTION A)
# ============================================================

def run_s1_on_scenarios(
    *,
    db_json: str,
    traj_npz_path: str,
    scenarios_json: str,
    obs_type: str = "car",
    base_radius: float = 0.5,
    collision_margin: float = 0.0,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]

    traj_npz = dict(np.load(traj_npz_path, allow_pickle=True))
    traj_index = _build_index(traj_npz)

    scenarios = json.loads(Path(scenarios_json).read_text())
    results: List[Dict[str, Any]] = []

    for sc in scenarios:
        scenario_id = int(sc["scenario_id"])

        if "A_query" not in sc:
            raise KeyError("Scenario missing 'A_query'. For OPTION A you must include A_query in scenarios JSON.")
        A_query = np.array(sc["A_query"], dtype=float)

        if "obstacle_center" not in sc:
            raise KeyError("Scenario missing 'obstacle_center'.")
        query_center = (float(sc["obstacle_center"][0]), float(sc["obstacle_center"][1]))

        t0 = time.perf_counter()

        # --- S1 retrieval by A ---
        cluster_id, d_cluster = select_best_cluster_A(A_query, db)
        dyn_id, d_dyn = select_best_dyn_A(A_query, db, cluster_id)

        dyn_node = db["dyn_nodes"][str(dyn_id)]
        inst_idx, stored_center, d_obs = select_closest_obstacle(query_center, dyn_node, obs_type)

        traj = load_trajectory(traj_npz, traj_index, dyn_id, inst_idx)

        runtime_sec = float(time.perf_counter() - t0)

        # --- Decide success for THIS benchmark scenario ---
        # Use: (trajectory exists) AND (collision-free w.r.t QUERY obstacle)
        if traj is None or traj.get("states", None) is None:
            success = False
            cost = None
            cf_query = None
        else:
            states = traj["states"]
            cf_query = collision_free_against_query_obstacle(
                states,
                obstacle_center=query_center,
                obstacle_radius=base_radius,
                margin=collision_margin,
            )

            # NOTE: We keep the precomputed trajectory cost from S1_all_data NPZ.
            # This cost corresponds to the STORED obstacle center used in that run,
            # not the query obstacle center. If you want "query-consistent cost",
            # we can recompute it separately.
            cost = traj.get("cost", None)
            success = bool(cf_query)

        results.append(
            {
                "scenario_id": scenario_id,

                # retrieved ids
                "retrieved_cluster_id": int(cluster_id),
                "retrieved_dyn_id": int(dyn_id),
                "retrieved_inst_idx": int(inst_idx),

                # scenario info (debug)
                "query_obstacle_center": [float(query_center[0]), float(query_center[1])],
                "stored_obstacle_center": [float(stored_center[0]), float(stored_center[1])],

                # distances (debug)
                "A_dist_cluster": float(d_cluster),
                "A_dist_dyn": float(d_dyn),
                "dist_obstacle": float(d_obs),

                # metrics required by benchmark_report.py
                "success": bool(success),
                "cost": None if cost is None else float(cost),
                "runtime_sec": float(runtime_sec),

                # extra flags
                "collision_free_wrt_query_obstacle": None if cf_query is None else bool(cf_query),
                "success_db": None if traj is None else traj.get("success_db", None),
                "runtime_sec_db": None if traj is None else traj.get("runtime_sec", None),
            }
        )

    if save_path is not None:
        Path(save_path).write_text(json.dumps(results, indent=2))
        print(f"✅ S1 benchmark results saved to {save_path}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_json", type=str, default="S1_database_single_obstacle.json")
    p.add_argument("--traj_npz", type=str, default="s1_mpc_trajectories.npz")
    p.add_argument("--scenarios", type=str, default="benchmark_scenarios.json")
    p.add_argument("--obs_type", type=str, default="car")
    p.add_argument("--base_radius", type=float, default=0.5)
    p.add_argument("--collision_margin", type=float, default=0.0)
    p.add_argument("--out", type=str, default="S1_results.json")
    args = p.parse_args()

    run_s1_on_scenarios(
        db_json=args.db_json,
        traj_npz_path=args.traj_npz,
        scenarios_json=args.scenarios,
        obs_type=args.obs_type,
        base_radius=args.base_radius,
        collision_margin=args.collision_margin,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
