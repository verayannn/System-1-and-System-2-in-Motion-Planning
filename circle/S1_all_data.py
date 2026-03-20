"""
S1_all_data.py — Fully Integrated Flexible Database Generation
Supports: MPC-DC (via do-mpc) and MPC-CBF (via MATLAB Engine)

python circle/S1_all_data.py --db_json circle/S1_database_single_obstacle.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import casadi as ca
import do_mpc

try:
    import matlab.engine
except ImportError:
    print("⚠️ MATLAB Engine not found. Use 'pip install matlabengine' to use MPC-CBF.")


# ============================================================
# 1) Solver Interfaces
# ============================================================

class BaseS2Solver:
    def solve_trajectory(self, A_mat: np.ndarray, x0: np.ndarray, goal: np.ndarray,
                         obs_center: List[float], radius: float, dt: float, n_steps: int):
        raise NotImplementedError
    

class MPCDCSolver:
    """RESTORED: Your original MPC-DC implementation using do-mpc"""
    def __init__(self, n_horizon=6, dt=0.1):
        # This is the part that was likely missing!
        self.n_horizon = n_horizon
        self.dt = dt

    def solve_trajectory(self, A_mat, x0, goal, obs_center, radius, dt, n_steps):
        # We use the 'dt' passed in during the call to ensure consistency 
        # across different solvers in S2_usage.py
        
        model = do_mpc.model.Model('discrete')
        _x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
        _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))
        
        # Dynamics: x_next = x + (Ax + Bu)*dt
        B = np.array([[0.0], [1.0]])
        model.set_rhs('x', _x + (A_mat @ _x + B @ _u) * dt)
        model.setup()

        # MPC Setup
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': dt,
            'n_robust': 0,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        # Objective (5,5) -> (0,0)
        mterm = ca.sumsqr(_x - goal)
        lterm = ca.sumsqr(_x - goal) + 0.1 * ca.sumsqr(_u) #####
        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Safety
        dist = ca.norm_2(_x - np.array(obs_center).reshape(2,1))
        mpc.set_nl_cons('obstacle', -dist + radius, soft_constraint=True)

        mpc.setup()
        mpc.x0 = x0
        mpc.set_initial_guess()

        # Simulation Loop
        states, inputs = [x0.flatten()], []
        curr_x = x0
        start_t = time.time()
        for _ in range(n_steps):
            u0 = mpc.make_step(curr_x)
            # Record the move
            curr_x = curr_x + (A_mat @ curr_x + B @ u0) * dt
            states.append(curr_x.flatten())
            inputs.append(u0.flatten())

        return np.array(states), np.array(inputs), (time.time() - start_t)/n_steps



class MATLAB_S2_Solver(BaseS2Solver):
    """Zeng et al. MPC-CBF via MATLAB"""


    def __init__(self, repo_path):
        import matlab.engine
        print("🔗 Searching for shared MATLAB session...")
        
        # Look for the session you named in Step 1
        sessions = matlab.engine.find_matlab()
        
        if 'MATLAB_FOR_PYTHON' in sessions:
            print("✅ Found 'MATLAB_FOR_PYTHON'. Connecting...")
            self.eng = matlab.engine.connect_matlab('MATLAB_FOR_PYTHON')
        elif len(sessions) > 0:
            print(f"✅ Found unnamed session ({sessions[0]}). Connecting...")
            self.eng = matlab.engine.connect_matlab(sessions[0])
        else:
            print("⚠️ No open MATLAB found. Attempting to start new engine (may fail license)...")
            self.eng = matlab.engine.start_matlab()

        # Add the Zeng et al. repo to the MATLAB path
        self.eng.addpath(self.eng.genpath(repo_path), nargout=0)
    

    def solve_trajectory(self, A_mat, x0, goal, obs_center, radius, dt, n_steps):
        states, inputs = [x0.flatten()], []
        curr_x = x0
        start_time = time.time()

        for _ in range(n_steps):
            res = self.eng.run_mpc_cbf_step(
                matlab.double(curr_x.tolist()),
                matlab.double(goal.tolist()),
                matlab.double(obs_center),
                float(dt),
                nargout=2
            )

            u = np.array(res[0]).flatten()

            B = np.array([[0], [1]])
            curr_x = curr_x + (A_mat @ curr_x + B @ u.reshape(-1, 1)) * dt

            states.append(curr_x.flatten())
            inputs.append(u)

        runtime = (time.time() - start_time) / n_steps
        return np.array(states), np.array(inputs), runtime


# ============================================================
# 2) Helper Functions
# ============================================================

def check_metrics(states, obs_center, radius, goal):
    """Calculates success, collision, and cost."""
    dists = np.linalg.norm(states[:, :2] - np.array(obs_center), axis=1)
    collision_free = np.all(dists > radius)
    goal_reached = np.linalg.norm(states[-1, :2] - goal) < 0.5
    return bool(collision_free and goal_reached), bool(collision_free)


# ============================================================
# 3) Main Loop
# ============================================================

def run_all_flexible(args):
    db_path = Path(args.db_json)
    payload = json.loads(db_path.read_text())
    db = payload["db"]

    if args.solver_type == "mpc_cbf_matlab":
        repo = str((Path(__file__).parent / "MPC-CBF-master").absolute())
        solver = MATLAB_S2_Solver(repo)
    else:
        solver = MPCDCSolver()

    all_states, all_inputs, all_metrics = [], [], []
    dyn_ids = sorted(db["dyn_nodes"].keys(), key=lambda x: int(x))

    for d_id in dyn_ids:
        A = np.array(db["dyn_nodes"][d_id]["A"])
        instances = db["dyn_nodes"][d_id]["obs_types"][args.obs_type]

        for idx, inst in enumerate(instances):
            print(f"Solving: Dyn {d_id} | Instance {idx} via {args.solver_type}")

            x0 = np.array([[5.0], [5.0]])
            goal = np.array([0.0, 0.0])

            states, inputs, runtime = solver.solve_trajectory(
                A, x0, goal, inst["center"], args.base_radius, args.dt, args.n_steps
            )

            success, cf = check_metrics(states, inst["center"], args.base_radius, goal)

            all_states.append(states)
            all_inputs.append(inputs)
            all_metrics.append({
                "dyn_id": int(d_id),
                "inst_idx": idx,
                "center": inst["center"],
                "success": success,
                "collision_free": cf,
                "runtime": runtime
            })

    np.savez(
        args.out_npz,
        states=np.array(all_states, dtype=object),
        inputs=np.array(all_inputs, dtype=object),
        metrics=all_metrics
    )

    print(f"✅ DB Saved to {args.out_npz}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solver_type", choices=["mpc_dc", "mpc_cbf_matlab"], default="mpc_dc")
    p.add_argument("--db_json", type=str, required=True)
    p.add_argument("--out_npz", type=str, default="s1_trajectories.npz")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--base_radius", type=float, default=1.0) #### 1.5
    p.add_argument("--obs_type", type=str, default="car")
    args = p.parse_args()

    run_all_flexible(args)