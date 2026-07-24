from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover - exercised in this env
    ca = None

from solvers._s2_common import (
    collision_free_rectangles,
    detect_acados_root,
    ensure_acados_template_path,
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
from solvers.S2_cbf import build_global_reference_path


AcadosModel = AcadosOcp = AcadosOcpSolver = None
_LIBC = ctypes.CDLL(None)
try:
    _LIBC.fflush.argtypes = [ctypes.c_void_p]
except Exception:
    pass


def _load_acados_template():
    global AcadosModel, AcadosOcp, AcadosOcpSolver
    if AcadosModel is None or AcadosOcp is None or AcadosOcpSolver is None:
        ensure_acados_template_path()
        try:
            from acados_template import AcadosModel as _AcadosModel, AcadosOcp as _AcadosOcp, AcadosOcpSolver as _AcadosOcpSolver
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "acados_template is missing a dependency in the active environment. "
                "Install 'deprecated' or activate the env that already has acados_template."
            ) from exc
        AcadosModel, AcadosOcp, AcadosOcpSolver = _AcadosModel, _AcadosOcp, _AcadosOcpSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver


@contextlib.contextmanager
def _suppress_fd_output(enabled: bool = True):
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            _LIBC.fflush(None)
        except Exception:
            pass
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            _LIBC.fflush(None)
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def _acados_backend_ready() -> bool:
    return detect_acados_root() is not None


def _dynamics_payload(scenario):
    payload = getattr(scenario, "nonlinear_dynamics", None)
    return payload if isinstance(payload, dict) else {}


def _nonlinear_drift(x, scenario):
    payload = _dynamics_payload(scenario)
    model = str(payload.get("model", "")).strip()
    regime = str(payload.get("regime", "")).strip()
    params = dict(payload.get("parameters", {}))
    x1, x2 = float(x[0]), float(x[1])

    if model == "control_affine_tanh_trig_2d":
        if regime == "sink":
            a = float(params.get("a", 0.5))
            b = float(params.get("b", 0.5))
            shear = float(params.get("shear", 0.0))
            return np.array(
                [-a * np.tanh(x1) + shear * np.sin(x2), -b * np.tanh(x2) - 0.5 * shear * np.sin(x1)],
                dtype=float,
            )
        if regime == "rotate_cw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return np.array(
                [-damp * np.tanh(x1) + omega * np.sin(x2), -damp * np.tanh(x2) - omega * np.sin(x1)],
                dtype=float,
            )
        if regime == "rotate_ccw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return np.array(
                [-damp * np.tanh(x1) - omega * np.sin(x2), -damp * np.tanh(x2) + omega * np.sin(x1)],
                dtype=float,
            )
        a = float(params.get("a", 0.5))
        b = float(params.get("b", 0.5))
        shear = float(params.get("shear", 0.0))
        return np.array(
            [-a * np.tanh(x1) + 2.0 * shear * x2, -b * np.tanh(x2) + 0.25 * shear * x1],
            dtype=float,
        )

    A = getattr(scenario, "A", None)
    if A is None:
        A = getattr(scenario, "A_query", None)
    if A is not None:
        A = np.asarray(A, dtype=float).reshape(2, 2)
        return (A @ np.asarray(x, dtype=float).reshape(2)).astype(float)
    return np.zeros(2, dtype=float)


def _nonlinear_drift_expr(x, scenario):
    payload = _dynamics_payload(scenario)
    model = str(payload.get("model", "")).strip()
    regime = str(payload.get("regime", "")).strip()
    params = dict(payload.get("parameters", {}))
    x1, x2 = x[0], x[1]

    if model == "control_affine_tanh_trig_2d":
        if regime == "sink":
            a = float(params.get("a", 0.5))
            b = float(params.get("b", 0.5))
            shear = float(params.get("shear", 0.0))
            return ca.vertcat(-a * ca.tanh(x1) + shear * ca.sin(x2), -b * ca.tanh(x2) - 0.5 * shear * ca.sin(x1))
        if regime == "rotate_cw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return ca.vertcat(-damp * ca.tanh(x1) + omega * ca.sin(x2), -damp * ca.tanh(x2) - omega * ca.sin(x1))
        if regime == "rotate_ccw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return ca.vertcat(-damp * ca.tanh(x1) - omega * ca.sin(x2), -damp * ca.tanh(x2) + omega * ca.sin(x1))
        a = float(params.get("a", 0.5))
        b = float(params.get("b", 0.5))
        shear = float(params.get("shear", 0.0))
        return ca.vertcat(-a * ca.tanh(x1) + 2.0 * shear * x2, -b * ca.tanh(x2) + 0.25 * shear * x1)

    A = getattr(scenario, "A", None)
    if A is None:
        A = getattr(scenario, "A_query", None)
    if A is not None:
        A = np.asarray(A, dtype=float).reshape(2, 2)
        return ca.vertcat(*(A[0, 0] * x1 + A[0, 1] * x2, A[1, 0] * x1 + A[1, 1] * x2))
    return ca.vertcat(0, 0)


def _step_state(x, u, dt, scenario):
    return np.asarray(x, dtype=float).reshape(2) + float(dt) * (
        _nonlinear_drift(x, scenario) + np.asarray(u, dtype=float).reshape(2)
    )


def _point_on_polyline(points: np.ndarray, distance: float) -> np.ndarray:
    remaining = float(distance)
    for index in range(len(points) - 1):
        start, end = points[index], points[index + 1]
        segment = float(np.linalg.norm(end - start))
        if segment <= 1e-9:
            continue
        if remaining <= segment:
            return start + (remaining / segment) * (end - start)
        remaining -= segment
    return np.asarray(points[-1], dtype=float)


def _reference_warm_start(
    z: np.ndarray,
    path: np.ndarray | None,
    path_index: int,
    goal: np.ndarray,
    scenario,
    dt: float,
    horizon: int,
    u_max: float,
    du_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Warm-start the augmented state ``[position, control]`` along a route."""
    z = np.asarray(z, dtype=float).reshape(4)
    x = z[:2]
    u_current = z[2:]
    if path is None or len(path) == 0:
        path = np.asarray([x, goal], dtype=float)
        path_index = 0

    path_index += int(np.argmin(np.linalg.norm(path[path_index:] - x, axis=1)))
    route = np.vstack([x, path[path_index:]])
    spacing = max(0.8 * float(u_max) * float(dt), 0.05)
    refs = np.vstack([x, *[_point_on_polyline(route, k * spacing) for k in range(1, horizon + 1)]])

    z_guess = [z.copy()]
    du_guess = []
    x_rollout = x.copy()
    u_rollout = u_current.copy()
    for target in refs[1:]:
        desired_u = np.clip((target - x_rollout) / float(dt) - _nonlinear_drift(x_rollout, scenario), -float(u_max), float(u_max))
        du = np.clip((desired_u - u_rollout) / float(dt), -float(du_max), float(du_max))
        # The augmented continuous model ramps the control over the interval.
        u_mid = u_rollout + 0.5 * float(dt) * du
        x_rollout = _step_state(x_rollout, u_mid, dt, scenario)
        u_rollout = np.clip(u_rollout + float(dt) * du, -float(u_max), float(u_max))
        du_guess.append(du)
        z_guess.append(np.r_[x_rollout, u_rollout])
    return np.asarray(z_guess, dtype=float), np.asarray(du_guess, dtype=float), refs, path_index


def _s1_primal_warm_start(
    z: np.ndarray,
    states: np.ndarray,
    inputs: np.ndarray | None,
    path_index: int,
    scenario,
    dt: float,
    source_dt: float,
    horizon: int,
    u_max: float,
    du_max: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build an acados primal iterate from an S1 rollout without changing yref."""
    z = np.asarray(z, dtype=float).reshape(4)
    path = np.asarray(states, dtype=float)[:, :2]
    path_index += int(np.argmin(np.linalg.norm(path[path_index:] - z[:2], axis=1)))
    source_dt = max(float(source_dt), 1e-6)
    sample_index = path_index + np.arange(horizon + 1, dtype=float) * float(dt) / source_dt
    sample_index = np.clip(sample_index, 0.0, float(len(path) - 1))
    grid = np.arange(len(path), dtype=float)
    positions = np.column_stack([np.interp(sample_index, grid, path[:, axis]) for axis in range(2)])
    positions[0] = z[:2]

    if inputs is None or np.asarray(inputs).ndim != 2 or len(inputs) == 0:
        controls = np.zeros((horizon + 1, 2), dtype=float)
        controls[0] = z[2:]
        for k in range(horizon):
            desired = (positions[k + 1] - positions[k]) / float(dt) - _nonlinear_drift(positions[k], scenario)
            controls[k + 1] = np.clip(desired, -float(u_max), float(u_max))
    else:
        source_u = np.asarray(inputs, dtype=float)[:, :2]
        input_index = np.clip(sample_index, 0.0, float(len(source_u) - 1))
        input_grid = np.arange(len(source_u), dtype=float)
        controls = np.column_stack([np.interp(input_index, input_grid, source_u[:, axis]) for axis in range(2)])
        controls = np.clip(controls, -float(u_max), float(u_max))
        controls[0] = z[2:]

    # Keep the primal state dynamically consistent with the nonlinear model.
    # The S1 route selects the local controls, but it is never treated as an
    # acados state reference or a feasibility constraint.
    z_guess = [z.copy()]
    x_rollout = z[:2].copy()
    for k in range(horizon):
        u_mid = 0.5 * (controls[k] + controls[k + 1])
        x_rollout = _step_state(x_rollout, u_mid, dt, scenario)
        z_guess.append(np.r_[x_rollout, controls[k + 1]])
    z_guess = np.asarray(z_guess, dtype=float)
    du_guess = np.clip(np.diff(controls, axis=0) / float(dt), -float(du_max), float(du_max))
    return z_guess, du_guess, path_index


def _usable_s1_warm_start(scenario, states, *, start: np.ndarray, rects, margin: float) -> tuple[np.ndarray | None, str]:
    """Reject malformed or unsafe S1 paths before passing them to acados."""
    if states is None:
        return None, "missing_states"
    path = np.asarray(states, dtype=float)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2 or not np.isfinite(path[:, :2]).all():
        return None, "invalid_states"
    path = path[:, :2]
    if np.linalg.norm(path[0] - np.asarray(start, dtype=float)[:2]) > 1.0:
        return None, "start_mismatch"
    if not collision_free_rectangles(path, rects, margin=float(margin)):
        return None, "collision"
    return path, "accepted"


def _build_solver(start, goal, rects, bounds, dt, horizon, u_max, scenario):
    if ca is None or not _acados_backend_ready():
        return None, None, False

    AcadosModel, AcadosOcp, AcadosOcpSolver = _load_acados_template()
    nx = 4
    nu = 2
    x = ca.MX.sym("x", nx)
    du = ca.MX.sym("du", nu)
    xdot = ca.MX.sym("xdot", nx)

    model = AcadosModel()
    model.name = "sofai_s2_mpc_nl_2d"
    model.x = x
    model.u = du
    model.xdot = xdot
    drift = _nonlinear_drift_expr(x[:2], scenario)
    model.f_expl_expr = ca.vertcat(drift[0] + x[2], drift[1] + x[3], du[0], du[1])
    model.f_impl_expr = xdot - model.f_expl_expr

    q_pos = float(env_float("SOFAI_MPC_Q_POS", 75.0))
    r_u = float(env_float("SOFAI_MPC_R_U", 0.75))
    r_du = float(env_float("SOFAI_MPC_R_DU", 0.50))
    # Slightly less conservative default safety geometry for accuracy-first MPC.
    robot_radius = float(env_float("SOFAI_MPC_RADIUS", 0.22))
    obstacle_margin = float(env_float("SOFAI_MPC_MARGIN", 0.10))
    exponent = float(env_float("SOFAI_MPC_RECT_EXPONENT", 10.0))
    obstacle_buffer = float(env_float("SOFAI_MPC_CONSTRAINT_BUFFER", 0.0))
    soften_obstacles = bool(env_int("SOFAI_MPC_SOFT_OBSTACLES", 1))
    slack_linear = float(env_float("SOFAI_MPC_OBSTACLE_SLACK_LINEAR", 1e4))
    slack_quadratic = float(env_float("SOFAI_MPC_OBSTACLE_SLACK_QUADRATIC", 1e5))

    model.cost_y_expr = ca.vertcat(x[0], x[1], x[2], x[3], du[0], du[1])
    model.cost_y_expr_e = ca.vertcat(x[0], x[1])

    if rects:
        h_expr = []
        for rect in rects:
            obs = rect_to_superellipse(
                rect,
                robot_radius=robot_radius,
                margin=obstacle_margin,
                exponent=exponent,
            )
            cx, cy, ax, ay, e, _, _ = obs
            level = ((x[0] - cx) / ax) ** e + ((x[1] - cy) / ay) ** e
            # Keep the same safe set (level >= 1) without feeding enormous
            # e=10 values from distant obstacles into the acados QP.
            h_expr.append((level - 1.0) / (level + 1.0))
        model.con_h_expr = ca.vertcat(*h_expr)
        model.con_h_expr_e = model.con_h_expr

    goal_x, goal_y = float(goal[0]), float(goal[1])
    z0 = np.array([float(start[0]), float(start[1]), 0.0, 0.0], dtype=float)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_export_directory = tempfile.mkdtemp(prefix="sofai_s2_mpc_")
    ocp.solver_options.N_horizon = int(horizon)
    ocp.solver_options.tf = float(dt) * int(horizon)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = int(env_int("SOFAI_MPC_MAX_ITER", 200))
    ocp.solver_options.print_level = 0
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 2

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag([q_pos, q_pos, r_u, r_u, r_du, r_du])
    ocp.cost.W_e = np.diag([2.0 * q_pos, 2.0 * q_pos])
    ocp.cost.yref = np.array([goal_x, goal_y, 0.0, 0.0, 0.0, 0.0], dtype=float)
    ocp.cost.yref_e = np.array([goal_x, goal_y], dtype=float)

    xmin, ymin, xmax, ymax = bounds
    du_max = float(env_float("SOFAI_MPC_DU_MAX", 12.0))
    ocp.constraints.x0 = z0
    ocp.constraints.lbx = np.array([xmin, ymin, -u_max, -u_max], dtype=float)
    ocp.constraints.ubx = np.array([xmax, ymax, u_max, u_max], dtype=float)
    ocp.constraints.idxbx = np.array([0, 1, 2, 3], dtype=int)
    ocp.constraints.lbu = np.array([-du_max, -du_max], dtype=float)
    ocp.constraints.ubu = np.array([du_max, du_max], dtype=float)
    ocp.constraints.idxbu = np.array([0, 1], dtype=int)

    if rects:
        n_obs = len(rects)
        ocp.constraints.lh = np.full(n_obs, obstacle_buffer, dtype=float)
        ocp.constraints.uh = np.full(n_obs, 1e8, dtype=float)
        ocp.constraints.lh_e = np.full(n_obs, obstacle_buffer, dtype=float)
        ocp.constraints.uh_e = np.full(n_obs, 1e8, dtype=float)
        if soften_obstacles:
            # Slack keeps the QP feasible during an SQP linearization failure.
            # The high penalty still makes obstacle violation a last resort.
            slack_indices = np.arange(n_obs, dtype=int)
            ocp.constraints.idxsh = slack_indices
            ocp.constraints.idxsh_e = slack_indices
            linear_penalty = np.full(n_obs, slack_linear, dtype=float)
            quadratic_penalty = np.full(n_obs, slack_quadratic, dtype=float)
            ocp.cost.zl = linear_penalty
            ocp.cost.zu = linear_penalty
            ocp.cost.Zl = quadratic_penalty
            ocp.cost.Zu = quadratic_penalty
            ocp.cost.zl_e = linear_penalty
            ocp.cost.zu_e = linear_penalty
            ocp.cost.Zl_e = quadratic_penalty
            ocp.cost.Zu_e = quadratic_penalty

    solver = AcadosOcpSolver(ocp, json_file=str(Path(ocp.code_export_directory) / "acados_ocp.json"), verbose=False)
    return solver, z0, soften_obstacles


def _set_solver_iterate(solver, z, z_guess, du_guess, reference_states, horizon: int) -> None:
    """Load a dynamically consistent primal iterate and tracking references."""
    for k in range(horizon + 1):
        solver.set(k, "x", z_guess[k])
    for k in range(horizon):
        solver.set(k, "u", du_guess[k])
        solver.set(k, "yref", np.r_[reference_states[k + 1], 0.0, 0.0, 0.0, 0.0])
    solver.set(horizon, "yref", reference_states[-1])
    solver.set(0, "lbx", z)
    solver.set(0, "ubx", z)


def _solve_mpc_fallback(*args, **kwargs):
    raise RuntimeError("S2 MPC is acados-only in this repo. Set ACADOS_SOURCE_DIR to a built acados install.")


def solve_MPC_with_info(
    scenario,
    *,
    s1_states: np.ndarray | None = None,
    s1_inputs: np.ndarray | None = None,
    s1_dt: float | None = None,
):
    try:
        if ca is None or not _acados_backend_ready():
            raise RuntimeError("S2 MPC requires a working acados backend. Set ACADOS_SOURCE_DIR to the built acados install.")

        rects = scenario_rects(scenario)
        start = scenario_start(scenario)
        goal = scenario_goal(scenario)
        bounds = scenario_bounds(scenario)
        dt = env_float("SOFAI_MPC_DT", 0.075)
        horizon = env_int("SOFAI_MPC_HORIZON", 20)
        n_steps = env_int("SOFAI_MPC_STEPS", 900)
        goal_tol = scenario_goal_tol(scenario, env_float("SOFAI_MPC_GOAL_TOL", 0.5))
        u_max = scenario_u_max(scenario)
        du_max = float(env_float("SOFAI_MPC_DU_MAX", 12.0))
        robot_radius = float(env_float("SOFAI_MPC_RADIUS", 0.22))
        obstacle_margin = float(env_float("SOFAI_MPC_MARGIN", 0.10))
        use_s1_warm_start = os.environ.get("SOFAI_MPC_USE_S1_WARM_START", "0").strip().lower() in {"1", "true", "yes", "on"}
        warm_states, warm_start_status = _usable_s1_warm_start(
            scenario,
            s1_states,
            start=start,
            rects=rects,
            margin=robot_radius + obstacle_margin,
        ) if use_s1_warm_start else (None, "disabled")
        warm_inputs = None if warm_states is None or s1_inputs is None else np.asarray(s1_inputs, dtype=float)
        warm_dt = float(s1_dt) if s1_dt is not None else dt
        reference_path = None
        if os.environ.get("SOFAI_MPC_GLOBAL_REFERENCE", "1").strip().lower() not in {"0", "false", "no", "off"}:
            reference_path = build_global_reference_path(
                start,
                goal,
                rects,
                bounds,
                clearance=robot_radius + obstacle_margin,
                resolution=env_float("SOFAI_MPC_REFERENCE_GRID", 0.20),
            )
        reference_index = 0

        with _suppress_fd_output(enabled=bool(env_int("SOFAI_MPC_SILENCE_ACADOS", 1))):
            solver, z, soft_obstacles = _build_solver(start, goal, rects, bounds, dt, horizon, u_max, scenario)
            if solver is None:
                raise RuntimeError("S2 MPC could not initialize acados.")

            states = [z[:2].copy()]
            inputs = []
            t0 = time.perf_counter()

            z_guess, du_guess, reference_states, reference_index = _reference_warm_start(
                z, reference_path, reference_index, goal, scenario, dt, horizon, u_max, du_max
            )
            warm_index = 0
            if warm_states is not None:
                z_guess, du_guess, warm_index = _s1_primal_warm_start(
                    z, warm_states, warm_inputs, warm_index, scenario, dt, warm_dt, horizon, u_max, du_max
                )

            solve_calls = 0
            nonzero_statuses: list[int] = []
            last_status = 0
            recovery_attempts = 0
            recovery_successes = 0
            max_recoveries = max(0, env_int("SOFAI_MPC_QP_RECOVERY_RETRIES", 1))
            termination = "step_limit"

            for _ in range(int(n_steps)):
                if goal_reached(np.asarray(states, dtype=float), goal, goal_tol):
                    termination = "goal_reached"
                    break

                _set_solver_iterate(solver, z, z_guess, du_guess, reference_states, horizon)
                status = int(solver.solve())
                solve_calls += 1
                last_status = status
                if status != 0:
                    nonzero_statuses.append(status)

                if status == 4 and recovery_attempts < max_recoveries:
                    # A fresh code-generated solver clears the failed QP state.
                    # Reinitialize strictly from the safe global reference, not S1.
                    recovery_attempts += 1
                    solver, _, _ = _build_solver(start, goal, rects, bounds, dt, horizon, u_max, scenario)
                    if solver is None:
                        termination = "recovery_initialization_failed"
                        break
                    z_guess, du_guess, reference_states, reference_index = _reference_warm_start(
                        z, reference_path, reference_index, goal, scenario, dt, horizon, u_max, du_max
                    )
                    _set_solver_iterate(solver, z, z_guess, du_guess, reference_states, horizon)
                    status = int(solver.solve())
                    solve_calls += 1
                    last_status = status
                    if status != 0:
                        nonzero_statuses.append(status)
                    if status == 0:
                        recovery_successes += 1

                if status != 0:
                    termination = f"acados_status_{status}"
                    print(f"[solve_MPC] acados failed with status={status}; stopping rollout before applying a stale control.", flush=True)
                    break
                try:
                    du = np.asarray(solver.get(0, "u"), dtype=float).reshape(-1)
                except Exception:
                    termination = "invalid_control"
                    break
                if not np.all(np.isfinite(du)):
                    termination = "nonfinite_control"
                    break

                u_mid = z[2:] + 0.5 * float(dt) * du
                next_position = _step_state(z[:2], u_mid, dt, scenario)
                # Slacks are an optimization recovery device, not permission to
                # execute a colliding action. Keep the benchmark's point checks.
                if not collision_free_rectangles(next_position.reshape(1, 2), rects):
                    termination = "predicted_collision"
                    break
                z[:2] = next_position
                z[2:] = np.clip(z[2:] + float(dt) * du, -float(u_max), float(u_max))
                states.append(z[:2].copy())
                inputs.append(u_mid.copy())

                z_guess, du_guess, reference_states, reference_index = _reference_warm_start(
                    z, reference_path, reference_index, goal, scenario, dt, horizon, u_max, du_max
                )
                if warm_states is not None:
                    z_guess, du_guess, warm_index = _s1_primal_warm_start(
                        z, warm_states, warm_inputs, warm_index, scenario, dt, warm_dt, horizon, u_max, du_max
                    )

            runtime = float(time.perf_counter() - t0)
            X = maybe_patch_goal_trajectory(
                np.asarray(states, dtype=float),
                goal,
                goal_tol,
                patch_tol=env_float("SOFAI_MPC_GOAL_PATCH_TOL", max(goal_tol, 0.75)),
            )
            U = np.asarray(inputs, dtype=float).reshape(-1, 2) if inputs else np.zeros((0, 2), dtype=float)

            return {
                "states": X,
                "inputs": U,
                "dt": float(dt),
                "runtime_sec": runtime,
                "success": bool(collision_free_rectangles(X, rects) and goal_reached(X, goal, goal_tol)),
                "collision_free": bool(collision_free_rectangles(X, rects)),
                "goal_reached": bool(goal_reached(X, goal, goal_tol)),
                "reference_path": reference_path,
                "solver_status": {
                    "solve_calls": int(solve_calls),
                    "last_status": int(last_status),
                    "nonzero_statuses": nonzero_statuses,
                    "termination": termination,
                    "obstacle_slacks_enabled": bool(soft_obstacles),
                    "qp_recovery_attempts": int(recovery_attempts),
                    "qp_recovery_successes": int(recovery_successes),
                },
                "s1_warm_start": {
                    "enabled": bool(use_s1_warm_start),
                    "used": bool(warm_states is not None),
                    "status": warm_start_status,
                },
            }
    except Exception as e:
        print(f"[solve_MPC] Exception occurred: {e}", flush=True)
        traceback.print_exc()
        return None


def solve_MPC(scenario):
    out = solve_MPC_with_info(scenario)
    return None if out is None else out["states"]
