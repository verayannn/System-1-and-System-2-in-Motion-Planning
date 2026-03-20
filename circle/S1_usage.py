"""
S1_usage.py — System-1 Retrieval-Based Benchmark

This script performs "System-1" reasoning: instead of solving an MPC 
optimization from scratch, it retrieves the best-matching "expert" 
trajectory from a precomputed database.

Retrieval Hierarchy:
1. Cluster Selection: Match A_query to Cluster Consensus A.
2. Dynamics Selection: Match A_query to specific Node A within Cluster.
3. Instance Selection: Match Query Obstacle Center to nearest Stored Obstacle.
4. Safety Verification: Re-check retrieved path against query obstacle.

python circle/S1_usage.py \
    --db_json circle/S1_database_single_obstacle.json \
    --traj_npz circle/s1_mpc_trajectories.npz \
    --scenarios circle/benchmark_scenarios.json \
    --out circle/S1_results.json \
    --base_radius 1.5
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ============================================================
# 1) Similarity Metrics
# ============================================================

def A_distance(A_query: np.ndarray, A_ref: np.ndarray) -> float:
    """
    Computes the normalized Frobenius distance between two dynamics matrices.
    """
    Aq = np.asarray(A_query, dtype=float)
    Ar = np.asarray(A_ref, dtype=float)
    dist = float(np.linalg.norm(Aq - Ar, ord="fro"))
    # Normalize by the magnitude of the reference to handle scale differences
    denom = float(np.linalg.norm(Ar, ord="fro")) + 1e-12
    return dist / denom

# ============================================================
# 2) Retrieval Logic
# ============================================================

def select_best_cluster(A_query: np.ndarray, db: Dict[str, Any]) -> int:
    """Finds the cluster ID with the closest consensus dynamics."""
    best_cid = None
    min_d = float("inf")
    for cid_str, node in db["consensus_nodes"].items():
        cd_dyn_id = str(node["cd_dyn_id"])
        A_cd = np.array(db["dyn_nodes"][cd_dyn_id]["A"])
        d = A_distance(A_query, A_cd)
        if d < min_d:
            min_d, best_cid = d, int(cid_str)
    return best_cid

def select_best_dyn(A_query: np.ndarray, db: Dict[str, Any], cluster_id: int) -> int:
    """Finds the best dynamics node within a specific cluster."""
    cluster = db["consensus_nodes"][str(cluster_id)]
    best_did = None
    min_d = float("inf")
    for did in cluster["dyn_children"]:
        A_dyn = np.array(db["dyn_nodes"][str(did)]["A"])
        d = A_distance(A_query, A_dyn)
        if d < min_d:
            min_d, best_did = d, int(did)
    return best_did

def select_closest_obstacle(query_center: List[float], dyn_node: Dict[str, Any]) -> Tuple[int, float]:
    """Finds the stored trajectory index that had the closest obstacle center."""
    best_idx = None
    min_d = float("inf")
    qc = np.array(query_center)
    # Ensure the dynamics node actually has 'car' type obstacles stored
    instances = dyn_node["obs_types"].get("car", [])
    for i, inst in enumerate(instances):
        stored_c = np.array(inst["center"])
        d = float(np.linalg.norm(qc - stored_c))
        if d < min_d:
            min_d, best_idx = d, i
    return best_idx, min_d

# ============================================================
# 3) Safety Re-Verification
# ============================================================

def is_safe_against_query(states: np.ndarray, obs_center: List[float], radius: float) -> bool:
    """
    Retrieval is only 'Successful' if the expert's path doesn't hit 
    the obstacle in its NEW position.
    """
    cx, cy = obs_center
    # Compute Euclidean distance to obstacle at every time step
    dists = np.sqrt((states[:, 0] - cx)**2 + (states[:, 1] - cy)**2)
    return bool(np.all(dists >= radius))

# ============================================================
# 4) Benchmark Execution
# ============================================================

def run_s1_benchmark(args):
    # 1. Load Database and Scenarios
    print(f"📂 Loading Database: {args.db_json}")
    db_payload = json.loads(Path(args.db_json).read_text())
    db = db_payload["db"]
    
    scenarios = json.loads(Path(args.scenarios).read_text())
    
    print(f"📦 Loading Trajectories: {args.traj_npz}")
    traj_data = np.load(args.traj_npz, allow_pickle=True)
    
    # 2. Build Index for fast NPZ lookup: (dyn_id, inst_idx) -> index
    traj_index = {}
    for i in range(len(traj_data["dyn_id"])):
        key = (int(traj_data["dyn_id"][i]), int(traj_data["instance_idx"][i]))
        traj_index[key] = i

    results = []
    print(f"🚀 Starting Retrieval for {len(scenarios)} scenarios...")

    

    for sc in scenarios:
        sid = sc["scenario_id"]
        A_query = np.array(sc["A_query"])
        q_center = sc["obstacle_center"]

        t_start = time.perf_counter()

        # Step A: Hierarchy Search
        cid = select_best_cluster(A_query, db)
        did = select_best_dyn(A_query, db, cid)
        inst_idx, obs_dist = select_closest_obstacle(q_center, db["dyn_nodes"][str(did)])

        # Step B: Retrieval
        traj_key = (did, inst_idx)
        success = False
        cost = None
        
        if traj_key in traj_index:
            idx = traj_index[traj_key]
            states = traj_data["states"][idx]
            
            # Step C: Safety Re-check (Crucial step!)
            safe = is_safe_against_query(states, q_center, args.base_radius)
            
            if safe:
                success = True
                cost = float(traj_data["cost"][idx])
        
        runtime = time.perf_counter() - t_start

        results.append({
            "scenario_id": sid,
            "success": bool(success),
            "cost": cost,
            "runtime_sec": float(runtime),
            "retrieved_dyn_id": did,
            "retrieved_inst_idx": inst_idx,
            "dist_to_stored_obs": obs_dist,
            "method": "S1_Retrieval"
        })

    # 3. Save Results
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"✅ S1 Benchmark Complete. Results saved to {args.out}")

# ============================================================
# 5) Main CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Input files
    p.add_argument("--db_json", type=str, default="circle/S1_database_single_obstacle.json")
    p.add_argument("--traj_npz", type=str, default="circle/s1_mpc_trajectories.npz")
    p.add_argument("--scenarios", type=str, default="circle/benchmark_scenarios.json")
    p.add_argument("--out", type=str, default="circle/S1_results.json")
    
    # Benchmarking constraints
    p.add_argument("--base_radius", type=float, default=1.0)  # Zeng et al. standard
    p.add_argument("--goal_tol", type=float, default=0.5)
    
    args = p.parse_args()
    run_s1_benchmark(args)