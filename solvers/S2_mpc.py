from __future__ import annotations

import contextlib
import ctypes
import os
import tempfile
import time
import traceback
import sys
from pathlib import Path

from scipy.optimize import minimize
import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover - exercised in this env
    ca = None

from solvers._s2_common import (
    detect_acados_root,
    collision_free_rectangles,
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
    acados_root = detect_acados_root()
    return acados_root is not None


def _build_solver(start, goal, rects, bounds, dt, horizon, u_max):
    if ca is None or not _acados_backend_ready():
        return None, None

    AcadosModel, AcadosOcp, AcadosOcpSolver = _load_acados_template()
    nx = 4
    nu = 2
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    xdot = ca.MX.sym("xdot", nx)

    px, py, theta, v = x[0], x[1], x[2], x[3]
    accel, omega = u[0], u[1]

    model = AcadosModel()
    model.name = "sofai_s2_mpc_du"
    model.x = x
    model.u = u
    model.xdot = xdot
    model.f_expl_expr = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
        accel,
    )
    model.f_impl_expr = xdot - model.f_expl_expr

    q_pos = float(env_float("SOFAI_MPC_Q_POS", 25.0))
    q_vel = float(env_float("SOFAI_MPC_Q_VEL", 2.0))
    r_accel = float(env_float("SOFAI_MPC_R_ACCEL", 0.2))
    r_omega = float(env_float("SOFAI_MPC_R_OMEGA", 0.2))
    robot_radius = float(env_float("SOFAI_MPC_RADIUS", 0.25))
    obstacle_margin = float(env_float("SOFAI_MPC_MARGIN", 0.10))
    exponent = float(env_float("SOFAI_MPC_RECT_EXPONENT", 8.0))

    model.cost_y_expr = ca.vertcat(px, py, v, accel, omega)
    model.cost_y_expr_e = ca.vertcat(px, py, v)

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
            h_expr.append(((px - cx) / ax) ** e + ((py - cy) / ay) ** e - 1.0)
        model.con_h_expr = ca.vertcat(*h_expr)
        model.con_h_expr_e = model.con_h_expr

    goal_x, goal_y = float(goal[0]), float(goal[1])
    x0 = np.array([start[0], start[1], 0.0, 0.0], dtype=float)
    goal_state = np.array([goal_x, goal_y, 0.0], dtype=float)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_export_directory = str(Path(tempfile.gettempdir()) / "sofai_s2_mpc_acados")
    Path(ocp.code_export_directory).mkdir(parents=True, exist_ok=True)
    ocp.solver_options.N_horizon = int(horizon)
    ocp.solver_options.tf = float(dt) * int(horizon)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = int(env_int("SOFAI_MPC_MAX_ITER", 20))
    ocp.solver_options.print_level = 0
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag([q_pos, q_pos, q_vel, r_accel, r_omega])
    ocp.cost.W_e = np.diag([q_pos, q_pos, q_vel])
    ocp.cost.yref = np.array([goal_x, goal_y, 0.0, 0.0, 0.0], dtype=float)
    ocp.cost.yref_e = goal_state

    xmin, ymin, xmax, ymax = bounds
    ocp.constraints.x0 = x0
    ocp.constraints.lbx = np.array([xmin, ymin, -np.pi, 0.0], dtype=float)
    ocp.constraints.ubx = np.array([xmax, ymax, np.pi, max(2.0 * float(u_max), 1.0)], dtype=float)
    ocp.constraints.idxbx = np.array([0, 1, 2, 3], dtype=int)
    ocp.constraints.lbu = np.array([-u_max, -u_max], dtype=float)
    ocp.constraints.ubu = np.array([u_max, u_max], dtype=float)
    ocp.constraints.idxbu = np.array([0, 1], dtype=int)

    if rects:
        n_obs = len(rects)
        ocp.constraints.lh = np.zeros(n_obs, dtype=float)
        ocp.constraints.uh = np.full(n_obs, 1e8, dtype=float)
        ocp.constraints.lh_e = np.zeros(n_obs, dtype=float)
        ocp.constraints.uh_e = np.full(n_obs, 1e8, dtype=float)

    solver = AcadosOcpSolver(ocp, json_file=str(Path(ocp.code_export_directory) / "acados_ocp.json"), verbose=False)
    return solver, x0


def _step_state(x, u, dt):
    px, py, theta, v = map(float, x)
    accel, omega = map(float, u)
    return np.array(
        [
            px + dt * v * np.cos(theta),
            py + dt * v * np.sin(theta),
            ((theta + dt * omega + np.pi) % (2.0 * np.pi)) - np.pi,
            max(v + dt * accel, 0.0),
        ],
        dtype=float,
    )


def _nominal_control(x, goal, u_max):
    dx = float(goal[0]) - float(x[0])
    dy = float(goal[1]) - float(x[1])
    theta_des = np.arctan2(dy, dx)
    theta_err = ((theta_des - float(x[2]) + np.pi) % (2.0 * np.pi)) - np.pi
    omega = np.clip(2.0 * theta_err, -float(u_max), float(u_max))

    dist = max(np.hypot(dx, dy) - 0.1, 0.0)
    v_des = min(dist, max(0.5 * float(u_max), 0.0))
    accel = np.clip(1.5 * (v_des - float(x[3])), -float(u_max), float(u_max))
    return np.array([accel, omega], dtype=float)


def _barrier_value(x, obs, robot_radius):
    if float(obs[-1]) == 0.0:
        ox, oy, r_obs = float(obs[0]), float(obs[1]), float(obs[2])
        d_min = robot_radius + r_obs
        return (float(x[0]) - ox) ** 2 + (float(x[1]) - oy) ** 2 - d_min**2
    ox, oy, a, b, e, theta = map(float, obs[:6])
    ct, st = np.cos(theta), np.sin(theta)
    px = ct * (float(x[0]) - ox) + st * (float(x[1]) - oy)
    py = -st * (float(x[0]) - ox) + ct * (float(x[1]) - oy)
    ax = max(a + robot_radius, 1e-3)
    ay = max(b + robot_radius, 1e-3)
    return (px / ax) ** e + (py / ay) ** e - 1.0


def _solve_mpc_fallback(start, goal, rects, bounds, dt, horizon, n_steps, goal_tol, u_max):
    robot_radius = float(env_float("SOFAI_MPC_RADIUS", 0.25))
    obstacle_margin = float(env_float("SOFAI_MPC_MARGIN", 0.10))
    exponent = float(env_float("SOFAI_MPC_RECT_EXPONENT", 8.0))
    q_pos = float(env_float("SOFAI_MPC_Q_POS", 25.0))
    q_vel = float(env_float("SOFAI_MPC_Q_VEL", 2.0))
    r_accel = float(env_float("SOFAI_MPC_R_ACCEL", 0.2))
    r_omega = float(env_float("SOFAI_MPC_R_OMEGA", 0.2))

    obs = [rect_to_superellipse(r, robot_radius=robot_radius, margin=obstacle_margin, exponent=exponent) for r in rects]

    x = np.array([start[0], start[1], 0.0, 0.0], dtype=float)
    states = [x.copy()]
    inputs = []
    u_seed = _nominal_control(x, goal, u_max)
    u_guess = np.tile(u_seed[None, :], (horizon, 1)).astype(float)
    t0 = time.perf_counter()

    def rollout(z):
        U = z.reshape(horizon, 2)
        X = [x.copy()]
        cur = x.copy()
        for uk in U:
            cur = _step_state(cur, uk, dt)
            X.append(cur.copy())
        return np.asarray(X, dtype=float), U

    def objective(z):
        X, U = rollout(z)
        cost = 0.0
        for k in range(horizon):
            dx = X[k, 0] - float(goal[0])
            dy = X[k, 1] - float(goal[1])
            cost += q_pos * (dx * dx + dy * dy) + q_vel * (X[k, 3] ** 2)
            cost += r_accel * (U[k, 0] ** 2) + r_omega * (U[k, 1] ** 2)
        dx = X[-1, 0] - float(goal[0])
        dy = X[-1, 1] - float(goal[1])
        cost += 2.0 * q_pos * (dx * dx + dy * dy) + 2.0 * q_vel * (X[-1, 3] ** 2)
        return float(cost)

    def obstacle_constraint(z):
        X, _ = rollout(z)
        vals = []
        for k in range(1, len(X)):
            for ob in obs:
                vals.append(_barrier_value(X[k], ob, robot_radius))
        return np.asarray(vals, dtype=float)

    bnds = [(-float(u_max), float(u_max))] * (2 * horizon)
    for _ in range(int(n_steps)):
        if goal_reached(np.asarray(states, dtype=float), goal, goal_tol):
            break

        z0 = u_guess.reshape(-1)
        constraints = []
        if obs:
            constraints.append({"type": "ineq", "fun": obstacle_constraint})

        res = minimize(
            objective,
            z0,
            method="SLSQP",
            bounds=bnds,
            constraints=constraints,
            options={"maxiter": 40, "ftol": 1e-3, "disp": False},
        )

        if res.success and np.all(np.isfinite(res.x)):
            u = np.asarray(res.x, dtype=float).reshape(horizon, 2)[0]
            u = np.clip(u, -float(u_max), float(u_max))
        else:
            u = _nominal_control(x, goal, u_max)

        if np.linalg.norm(u) < float(env_float("SOFAI_MPC_MIN_CONTROL_NORM", 1e-3)) and np.linalg.norm(x[:2] - np.asarray(goal[:2], dtype=float)) > float(goal_tol):
            u = _nominal_control(x, goal, u_max)

        x = _step_state(x, u, dt)
        states.append(x.copy())
        inputs.append(u.copy())
        u_guess = np.vstack([u[None, :], u_guess[:-1]])

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
        "runtime_sec": runtime,
        "success": bool(collision_free_rectangles(X, rects) and goal_reached(X, goal, goal_tol)),
        "collision_free": bool(collision_free_rectangles(X, rects)),
        "goal_reached": bool(goal_reached(X, goal, goal_tol)),
    }


def solve_MPC_with_info(scenario):
    try:
        rects = scenario_rects(scenario)
        start = scenario_start(scenario)
        goal = scenario_goal(scenario)
        bounds = scenario_bounds(scenario)
        dt = env_float("SOFAI_MPC_DT", 0.03)
        horizon = env_int("SOFAI_MPC_HORIZON", 12)
        n_steps = env_int("SOFAI_MPC_STEPS", 600)
        goal_tol = scenario_goal_tol(scenario, env_float("SOFAI_MPC_GOAL_TOL", 0.5))
        u_max = scenario_u_max(scenario)

        with _suppress_fd_output(enabled=bool(env_int("SOFAI_MPC_SILENCE_ACADOS", 1))):
            solver, x = _build_solver(start, goal, rects, bounds, dt, horizon, u_max)
            if solver is None:
                return _solve_mpc_fallback(start, goal, rects, bounds, dt, horizon, n_steps, goal_tol, u_max)

            states = [x.copy()]
            inputs = []
            t0 = time.perf_counter()

            x_guess = np.tile(x.reshape(1, -1), (horizon + 1, 1))
            u_seed = _nominal_control(x, goal, u_max)
            u_guess = np.tile(u_seed[None, :], (horizon, 1)).astype(float)

            for _ in range(int(n_steps)):
                if goal_reached(np.asarray(states, dtype=float), goal, goal_tol):
                    break

                for k in range(horizon + 1):
                    solver.set(k, "x", x_guess[k])
                for k in range(horizon):
                    solver.set(k, "u", u_guess[k])

                solver.set(0, "lbx", x)
                solver.set(0, "ubx", x)
                status = solver.solve()

                if status != 0:
                    u = _nominal_control(x, goal, u_max)
                else:
                    u = np.asarray(solver.get(0, "u"), dtype=float).reshape(-1)
                    if not np.all(np.isfinite(u)):
                        u = _nominal_control(x, goal, u_max)
                    elif np.linalg.norm(u) < float(env_float("SOFAI_MPC_MIN_CONTROL_NORM", 1e-3)) and np.linalg.norm(x[:2] - np.asarray(goal[:2], dtype=float)) > float(goal_tol):
                        u = _nominal_control(x, goal, u_max)

                x = _step_state(x, u, dt)
                states.append(x.copy())
                inputs.append(u.copy())

                x_guess = np.vstack([x[None, :], x_guess[:-1]])
                u_guess = np.vstack([u[None, :], u_guess[:-1]])

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
                "runtime_sec": runtime,
                "success": bool(collision_free_rectangles(X, rects) and goal_reached(X, goal, goal_tol)),
                "collision_free": bool(collision_free_rectangles(X, rects)),
                "goal_reached": bool(goal_reached(X, goal, goal_tol)),
            }
    except Exception as e:
        print(f"[solve_MPC] Exception occurred: {e}", flush=True)
        traceback.print_exc()
        return None


def solve_MPC(scenario):
    out = solve_MPC_with_info(scenario)
    return None if out is None else out["states"]
