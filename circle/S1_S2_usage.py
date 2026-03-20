"""
S1_S2_usage.py — Hybrid Retrieval-then-Expert Pipeline

Logic:
  1) S1: Retrieve best expert trajectory from DB using Frobenius distance on A.
  2) Re-check S1: Verify if the retrieved path is safe against the NEW obstacle.
  3) S2 Fallback: If S1 is unsafe/fails, solve from scratch using the MATLAB MPC-CBF expert.

S2: MPCCBF

python circle/S1_S2_usage.py \
    --db_json circle/S1_database_single_obstacle.json \
    --traj_npz circle/s1_mpc_trajectories.npz \
    --scenarios circle/benchmark_scenarios.json \
    --out circle/S1_S2_results.json \
    --radius 1.0 \
    --start_x 5.0 \
    --start_y 5.0 \
    --goal_x 0.0 \
    --goal_y 0.0
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# Import our unified solvers from S1_all_data
from S1_all_data import MATLAB_S2_Solver

# ============================================================
# 1) S1 Retrieval Helpers
# ============================================================

def A_distance(A_query: np.ndarray, A_ref: np.ndarray) -> float:
    """Normalized Frobenius distance for dynamics matching."""
    Aq = np.asarray(A_query, dtype=float)
    Ar = np.asarray(A_ref, dtype=float)
    dist = float(np.linalg.norm(Aq - Ar, ord="fro"))
    denom = float(np.linalg.norm(Ar, ord="fro")) + 1e-12
    return dist / denom

def select_best_cluster(A_query: np.ndarray, db: Dict[str, Any]) -> int:
    best_cid, min_d = None, float("inf")
    for cid, node in db["consensus_nodes"].items():
        A_cd = np.array(db["dyn_nodes"][str(node["cd_dyn_id"])]["A"])
        d = A_distance(A_query, A_cd)
        if d < min_d:
            min_d, best_cid = d, int(cid)
    return best_cid

def select_best_dyn(A_query: np.ndarray, db: Dict[str, Any], cluster_id: int) -> int:
    cluster = db["consensus_nodes"][str(cluster_id)]
    best_did, min_d = None, float("inf")
    for did in cluster["dyn_children"]:
        A_dyn = np.array(db["dyn_nodes"][str(did)]["A"])
        d = A_distance(A_query, A_dyn)
        if d < min_d:
            min_d, best_did = d, int(did)
    return best_did

def select_closest_obstacle(query_center: List[float], dyn_node: Dict[str, Any]) -> Tuple[int, float]:
    best_idx, min_d = None, float("inf")
    qc = np.array(query_center)
    for i, inst in enumerate(dyn_node["obs_types"].get("car", [])):
        stored_c = np.array(inst["center"])
        d = float(np.linalg.norm(qc - stored_c))
        if d < min_d:
            min_d, best_idx = d, i
    return best_idx, min_d

def is_safe(states: np.ndarray, obs_center: List[float], radius: float) -> bool:
    """Safety re-check for S1 path."""
    cx, cy = obs_center
    dists = np.sqrt((states[:, 0] - cx)**2 + (states[:, 1] - cy)**2)
    return bool(np.all(dists >= radius))

# ============================================================
# 2) Main Hybrid Runner
# ============================================================

def run_hybrid_benchmark(args):
    # Load DB and Scenarios
    db_payload = json.loads(Path(args.db_json).read_text())
    db = db_payload["db"]
    scenarios = json.loads(Path(args.scenarios).read_text())
    
    # Load S1 Trajectories and Build Index
    traj_data = np.load(args.traj_npz, allow_pickle=True)
    traj_index = {(int(traj_data["dyn_id"][i]), int(traj_data["instance_idx"][i])): i 
                  for i in range(len(traj_data["dyn_id"]))}

    # Initialize S2 MATLAB Solver
    repo_path = str((Path(__file__).parent / "MPC-CBF-master").absolute())
    s2_solver = MATLAB_S2_Solver(repo_path)

    results = []
    print(f"🔄 Running Hybrid S1+S2 on {len(scenarios)} scenarios...")

    

    for sc in scenarios:
        sid = sc["scenario_id"]
        A_query = np.array(sc["A_query"])
        q_center = sc["obstacle_center"]
        x0 = np.array([[args.start_x], [args.start_y]])
        goal = np.array([[args.goal_x], [args.goal_y]])

        t_start = time.perf_counter()
        mode = "S1"
        
        # --- PHASE 1: Try System-1 Retrieval ---
        cid = select_best_cluster(A_query, db)
        did = select_best_dyn(A_query, db, cid)
        inst_idx, _ = select_closest_obstacle(q_center, db["dyn_nodes"][str(did)])
        
        s1_key = (did, inst_idx)
        success = False
        cost = None
        
        if s1_key in traj_index:
            idx = traj_index[s1_key]
            s1_states = traj_data["states"][idx]
            if is_safe(s1_states, q_center, args.radius):
                success = True
                cost = float(traj_data["cost"][idx])
                mode = "S1_Retrieval"

        # --- PHASE 2: Fallback to System-2 Expert ---
        if not success:
            mode = "S2_Fallback_MATLAB"
            try:
                states, inputs, avg_rt = s2_solver.solve_trajectory(
                    A_mat=A_query, x0=x0, goal=goal,
                    obs_center=q_center, radius=args.radius,
                    dt=args.dt, n_steps=args.n_steps
                )
                dist_to_goal = np.linalg.norm(states[-1, :2] - goal.flatten())
                success = dist_to_goal <= args.goal_tol
                cost = np.sum(np.square(inputs)) * args.dt
            except Exception as e:
                print(f"❌ S2 Failed for scenario {sid}: {e}")
                success = False

        runtime = time.perf_counter() - t_start

        results.append({
            "scenario_id": sid,
            "mode": mode,
            "success": bool(success),
            "cost": float(cost) if cost is not None else None,
            "runtime_sec": float(runtime)
        })

    # Save Results
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"✅ Hybrid results saved to {args.out}")

# ============================================================
# 3) CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db_json", type=str, default="circle/S1_database_single_obstacle.json")
    p.add_argument("--traj_npz", type=str, default="circle/s1_mpc_trajectories.npz")
    p.add_argument("--scenarios", type=str, default="circle/benchmark_scenarios.json")
    p.add_argument("--out", type=str, default="circle/S1_S2_results.json")

    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--start_x", type=float, default=5.0)
    p.add_argument("--start_y", type=float, default=5.0)
    p.add_argument("--goal_x", type=float, default=0.0)
    p.add_argument("--goal_y", type=float, default=0.0)
    p.add_argument("--goal_tol", type=float, default=0.5)

    args = p.parse_args()
    run_hybrid_benchmark(args)