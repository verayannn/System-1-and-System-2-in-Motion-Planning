"""
S2_usage.py — System-2 Only Benchmark (from scratch)
Matches S1_usage.py structure and supports solver switching (MPC-DC vs MPC-CBF).

Reads:
  - circle/benchmark_scenarios.json
Runs:
  - Selected Solver (do-mpc or MATLAB MPC-CBF) for each query dynamics/obstacle.
Writes:
  - circle/S2_results.json

python circle/S2_usage.py --solver_type mpc_cbf_matlab --out circle/S2_MPCCBF_results.json

python circle/S2_usage.py --solver_type mpc_dc --out circle/S2_MPCDC_results.json

"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

# Import the Solver Classes from your S1_all_data.py 
# (Assuming they are in the same directory or accessible)
from S1_all_data import MPCDCSolver, MATLAB_S2_Solver

# ============================================================
# 1) Benchmark Runner
# ============================================================

def run_s2_benchmark(args):
    # Load scenarios
    scenarios = json.loads(Path(args.scenarios).read_text())
    
    # Initialize the chosen Solver
    if args.solver_type == "mpc_cbf_matlab":
        # Point to the Zeng et al. repo
        repo = str((Path(__file__).parent / "MPC-CBF-master").absolute())
        solver = MATLAB_S2_Solver(repo)
    else:
        solver = MPCDCSolver(n_horizon=args.n_horizon, dt=args.dt)

    results = []

    for sc in scenarios:
        scenario_id = sc["scenario_id"]
        A_query = np.array(sc["A_query"])
        obs_center = sc["obstacle_center"]
        
        print(f"[S2-Baseline] Scenario {scenario_id} | Solver: {args.solver_type}")
        
        x0 = np.array([[args.start_x], [args.start_y]])
        goal = np.array([[args.goal_x], [args.goal_y]])
        
        try:
            # Run the solve from scratch (no warm start)

            states, inputs, avg_rt = solver.solve_trajectory(
                A_mat=A_query,
                x0=x0,
                goal=goal,
                obs_center=obs_center,
                radius=args.base_radius,
                dt=args.dt,        # <--- ADD THIS LINE
                n_steps=args.n_steps
            )
            
            # Metrics
            # 1. Goal Success
            dist_to_goal = np.linalg.norm(states[-1, :2] - goal.flatten())
            success = dist_to_goal <= args.goal_tol
            
            # 2. Safety (Collision Free)
            dists_to_obs = np.linalg.norm(states[:, :2] - np.array(obs_center), axis=1)
            collision_free = np.all(dists_to_obs > args.base_radius)
            
            # 3. Cost (Control Effort)
            cost = np.sum(np.square(inputs)) * args.dt

            results.append({
                "scenario_id": scenario_id,
                "success": bool(success and collision_free),
                "collision_free": bool(collision_free),
                "goal_reached": bool(success),
                "cost": float(cost),
                "runtime_sec": float(avg_rt * args.n_steps), # Total trajectory time
                "solver_used": args.solver_type
            })

        except Exception as e:
            print(f"❌ Error in Scenario {scenario_id}: {e}")
            results.append({
                "scenario_id": scenario_id,
                "success": False,
                "error": str(e)
            })

    # Save results
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"✅ S2 Benchmark Complete. Results saved to {args.out}")

# ============================================================
# 2) Main / CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Logic Choice
    p.add_argument("--solver_type", choices=["mpc_dc", "mpc_cbf_matlab"], default="mpc_dc")
    
    # Files
    p.add_argument("--scenarios", type=str, default="circle/benchmark_scenarios.json")
    p.add_argument("--out", type=str, default="circle/S2_results.json")
    
    # Parameters (should match S1_all_data / S1_usage for fair comparison)
    p.add_argument("--n_horizon", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--start_x", type=float, default=5.0)
    p.add_argument("--start_y", type=float, default=5.0)
    p.add_argument("--goal_x", type=float, default=0.0)
    p.add_argument("--goal_y", type=float, default=0.0)
    p.add_argument("--base_radius", type=float, default=1.0)
    p.add_argument("--goal_tol", type=float, default=0.5)

    args = p.parse_args()
    run_s2_benchmark(args)