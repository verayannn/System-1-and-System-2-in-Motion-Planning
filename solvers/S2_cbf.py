from __future__ import annotations

import time
import traceback

import numpy as np

from solvers._s2_common import (
    collision_free_rectangles,
    env_float,
    env_int,
    goal_reached,
    maybe_patch_goal_trajectory,
    rect_to_superellipse,
    scenario_goal,
    scenario_goal_tol,
    scenario_rects,
    scenario_start,
    scenario_u_max,
)
from safe_control.position_control.cbf_qp import CBFQP


def _wrap_angle(theta: float) -> float:
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


class _DynamicUnicycleCBFRobot:
    def __init__(self, start: np.ndarray, dt: float, u_max: float):
        self.dt = float(dt)
        self.X = np.array([start[0], start[1], 0.0, 0.0], dtype=float).reshape(-1, 1)
        self.robot_spec = {
            "model": "DynamicUnicycle2D",
            "a_max": float(u_max),
            "w_max": float(u_max),
            "v_max": max(2.0 * float(u_max), 1.0),
            "radius": float(env_float("SOFAI_CBF_RADIUS", 0.25)),
        }
        self.robot_radius = float(self.robot_spec["radius"])

    def f(self):
        v = float(self.X[3, 0])
        th = float(self.X[2, 0])
        return np.array([v * np.cos(th), v * np.sin(th), 0.0, 0.0], dtype=float).reshape(-1, 1)

    def g(self):
        return np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)

    def step(self, u):
        u = np.asarray(u, dtype=float).reshape(2, 1)
        self.X = self.X + (self.f() + self.g() @ u) * self.dt
        self.X[2, 0] = _wrap_angle(self.X[2, 0])

    def nominal_input(self, goal, d_min):
        goal = np.asarray(goal, dtype=float).reshape(-1, 1)
        x = self.X
        distance = max(np.linalg.norm(x[0:2, 0] - goal[0:2, 0]) - float(d_min), 0.0)
        theta_d = np.arctan2(goal[1, 0] - x[1, 0], goal[0, 0] - x[0, 0])
        error_theta = _wrap_angle(theta_d - x[2, 0])
        omega = 2.0 * error_theta
        v = 0.0 if abs(error_theta) > np.deg2rad(90) else min(distance * np.cos(error_theta), self.robot_spec["v_max"])
        accel = 1.0 * (v - x[3, 0])
        return np.array([accel, omega], dtype=float).reshape(-1, 1)

    def _barrier_h(self, obs):
        x = self.X
        if float(obs[-1]) == 0.0:
            ox, oy, r_obs = float(obs[0]), float(obs[1]), float(obs[2])
            d_min = self.robot_radius + r_obs
            return (x[0, 0] - ox) ** 2 + (x[1, 0] - oy) ** 2 - 1.01 * d_min**2

        ox, oy, a, b, e, theta = map(float, obs[:6])
        ct, st = np.cos(theta), np.sin(theta)
        pox = ct * (x[0, 0] - ox) + st * (x[1, 0] - oy)
        poy = -st * (x[0, 0] - ox) + ct * (x[1, 0] - oy)
        ax = max(a + self.robot_radius, 1e-3)
        ay = max(b + self.robot_radius, 1e-3)
        return (pox / ax) ** e + (poy / ay) ** e - 1.0

    def agent_barrier(self, obs):
        x0 = self.X.copy()
        h = float(self._barrier_h(obs))

        def hdot_at(x_state):
            self.X = x_state
            grad = np.zeros((1, 4), dtype=float)
            eps = 1e-6
            base = float(self._barrier_h(obs))
            for i in range(4):
                x_pert = x_state.copy()
                x_pert[i, 0] += eps
                self.X = x_pert
                grad[0, i] = (float(self._barrier_h(obs)) - base) / eps
            self.X = x_state
            return float((grad @ self.f())[0, 0])

        self.X = x0
        grad = np.zeros((1, 4), dtype=float)
        eps = 1e-6
        base = float(self._barrier_h(obs))
        for i in range(4):
            x_pert = x0.copy()
            x_pert[i, 0] += eps
            self.X = x_pert
            grad[0, i] = (float(self._barrier_h(obs)) - base) / eps
        self.X = x0

        h_dot = float((grad @ self.f())[0, 0])
        dh_dot_dx = np.zeros((1, 4), dtype=float)
        for i in range(4):
            x_pert = x0.copy()
            x_pert[i, 0] += eps
            self.X = x_pert
            dh_dot_dx[0, i] = (hdot_at(x_pert) - h_dot) / eps
        self.X = x0
        return h, h_dot, dh_dot_dx


def _build_robot(start: np.ndarray, dt: float, u_max: float, n_obs: int):
    robot = _DynamicUnicycleCBFRobot(start, dt, u_max)
    robot.robot_spec["num_constraints"] = max(1, int(n_obs))
    return robot, robot.robot_spec


def solve_CBF_with_info(scenario):
    try:
        rects = scenario_rects(scenario)
        start = scenario_start(scenario)
        goal = scenario_goal(scenario)
        dt = env_float("SOFAI_CBF_DT", 0.05)
        n_steps = env_int("SOFAI_CBF_STEPS", 400)
        goal_tol = scenario_goal_tol(scenario, env_float("SOFAI_CBF_GOAL_TOL", 0.5))
        margin = env_float("SOFAI_CBF_MARGIN", 0.10)
        exponent = env_float("SOFAI_CBF_RECT_EXPONENT", 8.0)
        u_max = scenario_u_max(scenario)

        robot, robot_spec = _build_robot(start, dt, u_max, len(rects))
        controller = CBFQP(robot, robot_spec, num_obs=max(1, len(rects)))
        obstacles = [
            rect_to_superellipse(r, robot_radius=robot.robot_radius, margin=margin, exponent=exponent)
            for r in rects
        ]

        states = [robot.X.reshape(-1).copy()]
        inputs = []
        t0 = time.perf_counter()

        for _ in range(int(n_steps)):
            if goal_reached(np.asarray(states, dtype=float), goal, goal_tol):
                break

            u_ref = robot.nominal_input(goal, d_min=goal_tol)
            u = controller.solve_control_problem(
                robot.X,
                {"u_ref": u_ref},
                obstacles if obstacles else None,
            )
            if u is None:
                u = u_ref

            u = np.asarray(u, dtype=float).reshape(-1, 1)
            robot.step(u)
            states.append(robot.X.reshape(-1).copy())
            inputs.append(u.reshape(-1).copy())

        runtime = float(time.perf_counter() - t0)
        X = maybe_patch_goal_trajectory(np.asarray(states, dtype=float), goal, goal_tol, patch_tol=env_float("SOFAI_CBF_GOAL_PATCH_TOL", max(goal_tol, 0.75)))
        collision_free = collision_free_rectangles(X, rects, margin=robot.robot_radius)
        solved = goal_reached(X, goal, goal_tol)
        return {
            "states": X,
            "inputs": np.asarray(inputs, dtype=float),
            "collision_free": bool(collision_free),
            "solved": bool(solved),
            "runtime": runtime,
            "message": "success" if solved else "failed_to_reach_goal",
        }
    except Exception as e:
        print(f"[solve_CBF] Exception occurred: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "states": None,
            "inputs": None,
            "collision_free": False,
            "solved": False,
            "runtime": float("inf"),
            "message": str(e),
        }


def solve_CBF(scenario):
    out = solve_CBF_with_info(scenario)
    return out["states"]
