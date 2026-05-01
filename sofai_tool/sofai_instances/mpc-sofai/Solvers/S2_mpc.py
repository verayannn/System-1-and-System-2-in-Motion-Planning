import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from Solvers.Base.S2_mpc_maze import simulate_scenario
from Solvers.Base.S2_mpc_maze import collision_free_rectangles, goal_reached


def solve_MPC(scenario):
    """
    External S2 solver (MPC)

    Input:
        scenario (MazeProblem)

    Output:
        states (list) or None
        confidence (float)
    """


    try:
        states, inputs, runtime = simulate_scenario(
            scenario.A,
            scenario.B,
            scenario.rects,
            scenario.start,
            scenario.goal,
            dt=0.05,
            n_steps=800,
            n_horizon=20,
            u_max=scenario.u_max,
            bounds=scenario.bounds,
            wall_margin=0.2,
            smooth_kappa=20.0,
            goal_tol=0.5,
            calculate_correctness=False
        )

        return states

    except Exception as e:
        print(f"[solve_MPC] Exception occurred: {e}", flush=True)
        return None