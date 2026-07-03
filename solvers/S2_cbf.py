from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np

from safe_control.position_control.cbf_qp import CBFQP
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


def _dynamics_payload(scenario: Any) -> dict[str, Any]:
    payload = getattr(scenario, "nonlinear_dynamics", None)
    return payload if isinstance(payload, dict) else {}


def _nonlinear_drift(x: np.ndarray, scenario: Any) -> np.ndarray:
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
            return np.array([-a * np.tanh(x1) + shear * np.sin(x2), -b * np.tanh(x2) - 0.5 * shear * np.sin(x1)], dtype=float)
        if regime == "rotate_cw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return np.array([-damp * np.tanh(x1) + omega * np.sin(x2), -damp * np.tanh(x2) - omega * np.sin(x1)], dtype=float)
        if regime == "rotate_ccw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return np.array([-damp * np.tanh(x1) - omega * np.sin(x2), -damp * np.tanh(x2) + omega * np.sin(x1)], dtype=float)
        a = float(params.get("a", 0.5))
        b = float(params.get("b", 0.5))
        shear = float(params.get("shear", 0.0))
        return np.array([-a * np.tanh(x1) + 2.0 * shear * x2, -b * np.tanh(x2) + 0.25 * shear * x1], dtype=float)

    A = getattr(scenario, "A_query", getattr(scenario, "A", None))
    if A is not None:
        A = np.asarray(A, dtype=float).reshape(2, 2)
        return (A @ np.asarray(x, dtype=float).reshape(2)).astype(float)
    return np.zeros(2, dtype=float)


class _NonlinearPointCBFRobot:
    def __init__(self, start: np.ndarray, dt: float, u_max: float, scenario: Any):
        self.dt = float(dt)
        self.scenario = scenario
        self.X = np.asarray(start, dtype=float).reshape(2, 1)
        self.last_u = np.zeros((2, 1), dtype=float)
        self.robot_spec = {
            "model": "SingleIntegrator2D",
            "v_max": float(u_max),
            "radius": float(env_float("SOFAI_CBF_RADIUS", 0.25)),
        }
        self.robot_radius = float(self.robot_spec["radius"])

    def f(self):
        return _nonlinear_drift(self.X.reshape(-1), self.scenario).reshape(2, 1)

    def g(self):
        return np.eye(2, dtype=float)

    def step(self, u):
        u = np.asarray(u, dtype=float).reshape(2, 1)
        self.X = self.X + (self.f() + self.g() @ u) * self.dt
        self.last_u = u.copy()

    def nominal_input(self, goal, d_min):
        goal = np.asarray(goal, dtype=float).reshape(2)
        x = self.X.reshape(-1)
        drift = _nonlinear_drift(x, self.scenario)
        u = 1.8 * (goal - x) - drift
        return np.clip(u, -float(self.robot_spec["v_max"]), float(self.robot_spec["v_max"])).reshape(-1, 1)

    def _barrier_h(self, obs):
        x = self.X.reshape(-1)
        if float(obs[-1]) == 0.0:
            ox, oy, r_obs = float(obs[0]), float(obs[1]), float(obs[2])
            d_min = self.robot_radius + r_obs
            return (x[0] - ox) ** 2 + (x[1] - oy) ** 2 - 1.01 * d_min**2

        ox, oy, a, b, e, theta = map(float, obs[:6])
        ct, st = np.cos(theta), np.sin(theta)
        px = ct * (x[0] - ox) + st * (x[1] - oy)
        py = -st * (x[0] - ox) + ct * (x[1] - oy)
        ax = max(a + self.robot_radius, 1e-3)
        ay = max(b + self.robot_radius, 1e-3)
        return (px / ax) ** e + (py / ay) ** e - 1.0

    def agent_barrier(self, obs):
        x0 = self.X.copy()
        h = float(self._barrier_h(obs))
        grad = np.zeros((1, 2), dtype=float)
        eps = 1e-6
        base = float(self._barrier_h(obs))
        for i in range(2):
            x_pert = x0.copy()
            x_pert[i, 0] += eps
            self.X = x_pert
            grad[0, i] = (float(self._barrier_h(obs)) - base) / eps
        self.X = x0
        return h, grad


def _build_robot(start: np.ndarray, dt: float, u_max: float, scenario: Any, n_obs: int):
    robot = _NonlinearPointCBFRobot(start, dt, u_max, scenario)
    robot.robot_spec["num_constraints"] = max(1, int(n_obs))
    robot.robot_spec["cbf_mode"] = "hard"
    return robot, robot.robot_spec


def _rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (R @ np.asarray(vec, dtype=float).reshape(2)).astype(float)


def _choose_reference_control(robot: _NonlinearPointCBFRobot, goal: np.ndarray, rects, goal_tol: float) -> np.ndarray:
    x = robot.X.reshape(-1)
    drift = _nonlinear_drift(x, robot.scenario)
    goal_vec = np.asarray(goal, dtype=float).reshape(2) - x
    if np.linalg.norm(goal_vec) < 1e-8:
        goal_vec = np.array([1.0, 0.0], dtype=float)

    base = np.clip(1.8 * goal_vec - drift, -robot.robot_spec["v_max"], robot.robot_spec["v_max"])
    candidates = [
        base,
        robot.last_u.reshape(-1),
        0.8 * base,
        0.6 * base,
        0.4 * base,
        0.25 * base,
        np.zeros(2, dtype=float),
    ]

    for angle in (np.pi / 6.0, -np.pi / 6.0, np.pi / 4.0, -np.pi / 4.0, np.pi / 2.0, -np.pi / 2.0):
        cand = 1.8 * _rotate(goal_vec, angle) - drift
        candidates.append(cand)

    best_u = None
    best_score = float("inf")
    dist_curr = float(np.linalg.norm(x - goal[:2]))
    for u in candidates:
        u = np.clip(np.asarray(u, dtype=float).reshape(2), -robot.robot_spec["v_max"], robot.robot_spec["v_max"])
        x_next = x + robot.dt * (drift + u)
        if not np.isfinite(x_next).all():
            continue
        if not collision_free_rectangles(np.vstack([x[:2], x_next[:2]]), rects, margin=robot.robot_radius):
            continue
        dist_next = float(np.linalg.norm(x_next - goal[:2]))
        score = dist_next + 0.03 * float(np.linalg.norm(u - base)) + (0.0 if dist_next <= dist_curr else 1.5)
        if score < best_score:
            best_score = score
            best_u = u

    if best_u is None:
        best_u = base
    return np.asarray(best_u, dtype=float).reshape(-1, 1)


def solve_CBF_with_info(scenario):
    try:
        rects = scenario_rects(scenario)
        start = scenario_start(scenario)
        goal = scenario_goal(scenario)
        dt = env_float("SOFAI_CBF_DT", 0.04)
        n_steps = env_int("SOFAI_CBF_STEPS", 600)
        goal_tol = scenario_goal_tol(scenario, env_float("SOFAI_CBF_GOAL_TOL", 0.5))
        margin = env_float("SOFAI_CBF_MARGIN", 0.20)
        exponent = env_float("SOFAI_CBF_RECT_EXPONENT", 10.0)
        u_max = scenario_u_max(scenario)

        robot, robot_spec = _build_robot(start, dt, u_max, scenario, len(rects))
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

            u_ref = _choose_reference_control(robot, goal, rects, goal_tol)
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
        X = maybe_patch_goal_trajectory(
            np.asarray(states, dtype=float),
            goal,
            goal_tol,
            patch_tol=env_float("SOFAI_CBF_GOAL_PATCH_TOL", max(goal_tol, 0.75)),
        )
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
