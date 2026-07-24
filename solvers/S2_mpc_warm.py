"""S2 MPC variant that warm-starts from a failed S1 trajectory."""

from __future__ import annotations

import numpy as np

from solvers.S2_mpc import solve_MPC_with_info as _solve_mpc_with_info


def solve_MPC_warm_with_info(
    scenario,
    *,
    s1_states: np.ndarray | None = None,
    s1_inputs: np.ndarray | None = None,
    s1_dt: float | None = None,
):
    """Solve MPC with the failed S1 trajectory as its primal warm start."""
    return _solve_mpc_with_info(
        scenario,
        s1_states=s1_states,
        s1_inputs=s1_inputs,
        s1_dt=s1_dt,
        use_s1_warm_start=True,
    )


def solve_MPC_warm(scenario):
    out = solve_MPC_warm_with_info(scenario)
    return None if out is None else out["states"]
