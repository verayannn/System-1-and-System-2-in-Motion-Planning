"""Experimental nonlinear System-2 MPC using do-mpc and IPOPT.

This backend intentionally remains separate from the native acados solver in
``S2_mpc.py``. Both use the same scenario contract and executed dynamics.
"""

from __future__ import annotations

import time
import traceback
import warnings
from typing import Any

import numpy as np

try:
    # do-mpc 5.1.x emits this warning from an internal identity comparison.
    # It is unrelated to the MPC model and otherwise repeats in each worker.
    warnings.filterwarnings(
        "ignore",
        message=r'"is" with a literal.*',
        category=SyntaxWarning,
    )
    import casadi as ca
    import do_mpc
except ImportError:  # pragma: no cover - depends on optional installation
    ca = None
    do_mpc = None

from solvers.S2_cbf import build_global_reference_path
from solvers.S2_mpc import _nonlinear_drift_expr, _point_on_polyline, _step_state
from solvers._s2_common import (
    collision_free_rectangles,
    env_float,
    env_int,
    goal_reached,
    maybe_patch_goal_trajectory,
    rect_to_superellipse,
    scenario_bounds,
    scenario_goal,
    scenario_goal_tol,
    scenario_rects,
    scenario_start,
    scenario_u_max,
)


def _build_controller(start, goal, rects, bounds, dt: float, horizon: int, u_max: float, scenario):
    if do_mpc is None or ca is None:
        raise RuntimeError("S2 MPC do-mpc requires the optional dependency 'do-mpc[full]'.")

    q_pos = env_float("SOFAI_MPC_DO_Q_POS", 75.0)
    r_u = env_float("SOFAI_MPC_DO_R_U", 0.75)
    r_du = env_float("SOFAI_MPC_DO_R_DU", 0.50)
    du_max = env_float("SOFAI_MPC_DO_DU_MAX", 12.0)
    radius = env_float("SOFAI_MPC_DO_RADIUS", 0.22)
    margin = env_float("SOFAI_MPC_DO_MARGIN", 0.10)
    exponent = env_float("SOFAI_MPC_DO_RECT_EXPONENT", 10.0)
    slack_penalty = env_float("SOFAI_MPC_DO_OBSTACLE_SLACK_PENALTY", 1e5)
    max_slack = env_float("SOFAI_MPC_DO_OBSTACLE_MAX_SLACK", 0.25)

    model = do_mpc.model.Model("continuous")
    z = model.set_variable(var_type="_x", var_name="z", shape=(4, 1))
    du = model.set_variable(var_type="_u", var_name="du", shape=(2, 1))
    target = model.set_variable(var_type="_tvp", var_name="target", shape=(2, 1))
    drift = _nonlinear_drift_expr(z[:2], scenario)
    model.set_rhs("z", ca.vertcat(drift[0] + z[2], drift[1] + z[3], du[0], du[1]))

    position_error = z[:2] - target
    stage_cost = q_pos * ca.dot(position_error, position_error) + r_u * ca.dot(z[2:], z[2:])
    terminal_cost = 2.0 * q_pos * ca.dot(position_error, position_error)
    model.set_expression("stage_cost", stage_cost)
    model.set_expression("terminal_cost", terminal_cost)
    model.setup()

    controller = do_mpc.controller.MPC(model)
    controller.settings.supress_ipopt_output()
    controller.set_param(
        n_horizon=int(horizon),
        t_step=float(dt),
        n_robust=0,
        state_discretization="collocation",
        collocation_type="radau",
        collocation_deg=2,
        collocation_ni=1,
        store_full_solution=False,
    )
    controller.settings.nlpsol_opts.update(
        {
            "ipopt.max_iter": int(env_int("SOFAI_MPC_DO_MAX_ITER", 400)),
            "ipopt.tol": float(env_float("SOFAI_MPC_DO_TOL", 1e-5)),
            "ipopt.acceptable_tol": float(env_float("SOFAI_MPC_DO_ACCEPTABLE_TOL", 1e-4)),
            "ipopt.print_level": 0,
            "print_time": 0,
        }
    )
    controller.set_objective(mterm=model.aux["terminal_cost"], lterm=model.aux["stage_cost"])
    controller.set_rterm(du=np.full(2, r_du, dtype=float))

    xmin, ymin, xmax, ymax = bounds
    controller.bounds["lower", "_x", "z"] = np.array([xmin, ymin, -u_max, -u_max], dtype=float)
    controller.bounds["upper", "_x", "z"] = np.array([xmax, ymax, u_max, u_max], dtype=float)
    controller.bounds["lower", "_u", "du"] = np.full(2, -du_max, dtype=float)
    controller.bounds["upper", "_u", "du"] = np.full(2, du_max, dtype=float)

    for index, rect in enumerate(rects):
        cx, cy, ax, ay, e, _, _ = rect_to_superellipse(
            rect,
            robot_radius=radius,
            margin=margin,
            exponent=exponent,
        )
        level = ((z[0] - cx) / ax) ** e + ((z[1] - cy) / ay) ** e
        controller.set_nl_cons(
            f"obstacle_{index}",
            1.0 - level,
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=slack_penalty,
            maximum_violation=max_slack,
        )

    reference = {"target": np.asarray(goal, dtype=float).reshape(2, 1)}

    def tvp_fun(_):
        template = controller.get_tvp_template()
        template["_tvp", :, "target"] = reference["target"]
        return template

    controller.set_tvp_fun(tvp_fun)
    controller.setup()
    controller.x0 = np.r_[np.asarray(start, dtype=float).reshape(2), 0.0, 0.0].reshape(4, 1)
    controller.u0 = np.zeros((2, 1))
    controller.set_initial_guess()
    return controller, reference, radius + margin


def _reference_target(position: np.ndarray, path: np.ndarray | None, path_index: int, goal: np.ndarray, lookahead: float) -> tuple[np.ndarray, int]:
    if path is None or len(path) == 0:
        return np.asarray(goal, dtype=float).reshape(2), 0
    path_index += int(np.argmin(np.linalg.norm(path[path_index:] - position, axis=1)))
    route = np.vstack([position, path[path_index:]])
    return _point_on_polyline(route, lookahead), path_index


def _solver_status(controller) -> dict[str, Any]:
    stats = getattr(controller, "solver_stats", {}) or {}
    if not isinstance(stats, dict):
        stats = {}
    return {
        "backend": "do_mpc_ipopt",
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "unknown")),
        "iterations": int(stats.get("iter_count", 0) or 0),
    }


def solve_MPC_DO_with_info(
    scenario,
    *,
    s1_states: np.ndarray | None = None,
    s1_inputs: np.ndarray | None = None,
    s1_dt: float | None = None,
):
    """Solve one nonlinear scenario using a do-mpc/IPOPT receding horizon."""
    del s1_states, s1_inputs, s1_dt  # Experimental backend currently uses only the safe global reference.
    try:
        rects = scenario_rects(scenario)
        start = scenario_start(scenario)
        goal = scenario_goal(scenario)
        bounds = scenario_bounds(scenario)
        dt = env_float("SOFAI_MPC_DO_DT", 0.075)
        horizon = env_int("SOFAI_MPC_DO_HORIZON", 20)
        n_steps = env_int("SOFAI_MPC_DO_STEPS", 900)
        goal_tol = scenario_goal_tol(scenario, env_float("SOFAI_MPC_DO_GOAL_TOL", 0.5))
        u_max = scenario_u_max(scenario)
        du_max = env_float("SOFAI_MPC_DO_DU_MAX", 12.0)
        clearance = env_float("SOFAI_MPC_DO_RADIUS", 0.22) + env_float("SOFAI_MPC_DO_MARGIN", 0.10)
        reference_path = build_global_reference_path(
            start,
            goal,
            rects,
            bounds,
            clearance=clearance,
            resolution=env_float("SOFAI_MPC_DO_REFERENCE_GRID", 0.20),
        )

        controller, reference, _ = _build_controller(start, goal, rects, bounds, dt, horizon, u_max, scenario)
        z = np.r_[start, 0.0, 0.0].astype(float)
        states = [z[:2].copy()]
        inputs = []
        status_history = []
        reference_index = 0
        termination = "step_limit"
        t0 = time.perf_counter()

        for _ in range(int(n_steps)):
            if goal_reached(np.asarray(states, dtype=float), goal, goal_tol):
                termination = "goal_reached"
                break

            target, reference_index = _reference_target(
                z[:2],
                reference_path,
                reference_index,
                goal,
                lookahead=max(0.8 * u_max * dt * horizon, 1.0),
            )
            reference["target"] = target.reshape(2, 1)
            controller.x0 = z.reshape(4, 1)
            du = np.asarray(controller.make_step(controller.x0), dtype=float).reshape(-1)
            status = _solver_status(controller)
            status_history.append(status)
            if not status["success"] or du.shape[0] != 2 or not np.isfinite(du).all():
                termination = "ipopt_failure"
                break

            du = np.clip(du, -du_max, du_max)
            u_mid = z[2:] + 0.5 * dt * du
            next_position = _step_state(z[:2], u_mid, dt, scenario)
            if not collision_free_rectangles(next_position.reshape(1, 2), rects):
                termination = "predicted_collision"
                break
            z[:2] = next_position
            z[2:] = np.clip(z[2:] + dt * du, -u_max, u_max)
            states.append(z[:2].copy())
            inputs.append(u_mid.copy())

        runtime = float(time.perf_counter() - t0)
        trajectory = maybe_patch_goal_trajectory(
            np.asarray(states, dtype=float),
            goal,
            goal_tol,
            patch_tol=env_float("SOFAI_MPC_DO_GOAL_PATCH_TOL", max(goal_tol, 0.75)),
        )
        controls = np.asarray(inputs, dtype=float).reshape(-1, 2) if inputs else np.zeros((0, 2), dtype=float)
        return {
            "states": trajectory,
            "inputs": controls,
            "dt": float(dt),
            "runtime_sec": runtime,
            "success": bool(collision_free_rectangles(trajectory, rects) and goal_reached(trajectory, goal, goal_tol)),
            "collision_free": bool(collision_free_rectangles(trajectory, rects)),
            "goal_reached": bool(goal_reached(trajectory, goal, goal_tol)),
            "reference_path": reference_path,
            "solver_status": {
                "backend": "do_mpc_ipopt",
                "solve_calls": len(status_history),
                "last": status_history[-1] if status_history else {},
                "termination": termination,
                "soft_obstacles_enabled": True,
            },
            "s1_warm_start": {"enabled": False, "used": False, "status": "unsupported_backend"},
        }
    except Exception as exc:
        print(f"[solve_MPC_DO] Exception occurred: {exc}", flush=True)
        traceback.print_exc()
        return None


def solve_MPC_DO(scenario):
    result = solve_MPC_DO_with_info(scenario)
    return None if result is None else result["states"]
