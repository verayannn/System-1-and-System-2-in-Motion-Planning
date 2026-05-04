import os

import numpy as np

from solvers.base.S2_cbf_maze import simulate_sfcbf
from solvers.base.S2_cbf_maze import collision_free_rectangles, goal_reached


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


def solve_CBF(scenario):
    """
    External System-2 solver using the SFCBF safety-filter controller.

    Input:
        scenario (MazeProblem)

    Output:
        states (np.ndarray) or None
    """
    try:
        out = simulate_sfcbf(
            A=np.asarray(scenario.A, dtype=float),
            B=np.asarray(scenario.B, dtype=float),
            rects=scenario.rects,
            start=scenario.start,
            goal=scenario.goal,
            dt=_env_float("SOFAI_CBF_DT", 0.05),
            n_steps=_env_int("SOFAI_CBF_STEPS", 800),
            u_max=float(scenario.u_max),
            margin=_env_float("SOFAI_CBF_MARGIN", 0.35),
            gamma=_env_float("SOFAI_CBF_GAMMA", 2.0),
            goal_tol=_env_float("SOFAI_CBF_GOAL_TOL", 0.6),
            collision_margin=_env_float("SOFAI_CBF_COLLISION_MARGIN", 0.05),
        )
        return np.asarray(out["states"], dtype=float)

    except Exception as e:
        print(f"[solve_CBF] Exception occurred: {e}", flush=True)
        return None
