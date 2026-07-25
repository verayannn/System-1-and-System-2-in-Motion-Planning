from __future__ import annotations

import heapq
import time
import traceback
from typing import Any

import numpy as np

from safe_control.position_control.cbf_qp import CBFQP
from solvers._s2_common import (
    collision_free_rectangles,
    env_bool,
    env_float,
    env_int,
    goal_reached,
    maybe_patch_goal_trajectory,
    scenario_bounds,
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


def _rect_to_cbf_obstacle(rect, *, margin: float, exponent: float) -> np.ndarray:
    """Convert a rectangle to a conservatively inflated superellipse."""
    xmin, ymin, xmax, ymax = map(float, rect)
    return np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            max(0.5 * (xmax - xmin) + float(margin), 1e-3),
            max(0.5 * (ymax - ymin) + float(margin), 1e-3),
            float(exponent),
            0.0,
            1.0,
        ],
        dtype=float,
    )


def _nearest_free_cell(blocked: np.ndarray, row: int, col: int) -> tuple[int, int] | None:
    rows, cols = blocked.shape
    if 0 <= row < rows and 0 <= col < cols and not blocked[row, col]:
        return row, col
    for radius in range(1, max(rows, cols)):
        r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
        c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
        candidates = np.argwhere(~blocked[r0:r1, c0:c1])
        if candidates.size:
            distances = np.sum(np.square(candidates - np.array([row - r0, col - c0])), axis=1)
            best = candidates[int(np.argmin(distances))]
            return int(best[0] + r0), int(best[1] + c0)
    return None


def build_global_reference_path(start: np.ndarray, goal: np.ndarray, rects, bounds, *, clearance: float, resolution: float) -> np.ndarray | None:
    """Plan a static 8-connected reference path around inflated rectangles."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    resolution = max(float(resolution), 0.1)
    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)
    blocked = np.zeros_like(xx, dtype=bool)
    for rx1, ry1, rx2, ry2 in rects:
        blocked |= (
            (xx >= float(rx1) - clearance)
            & (xx <= float(rx2) + clearance)
            & (yy >= float(ry1) - clearance)
            & (yy <= float(ry2) + clearance)
        )

    start_cell = _nearest_free_cell(blocked, int(np.argmin(np.abs(ys - start[1]))), int(np.argmin(np.abs(xs - start[0]))))
    goal_cell = _nearest_free_cell(blocked, int(np.argmin(np.abs(ys - goal[1]))), int(np.argmin(np.abs(xs - goal[0]))))
    if start_cell is None or goal_cell is None:
        return None

    rows, cols = blocked.shape
    costs = np.full((rows, cols), np.inf, dtype=float)
    parent = np.full((rows, cols, 2), -1, dtype=np.int32)
    costs[start_cell] = 0.0
    queue: list[tuple[float, float, int, int]] = [(0.0, 0.0, *start_cell)]
    neighbors = [(dr, dc, float(np.hypot(dr, dc))) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    while queue:
        _, current_cost, row, col = heapq.heappop(queue)
        if current_cost != costs[row, col]:
            continue
        if (row, col) == goal_cell:
            break
        for dr, dc, step_cost in neighbors:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[row + dr, col] or blocked[row, col + dc]):
                continue
            next_cost = current_cost + step_cost
            if next_cost >= costs[nr, nc]:
                continue
            costs[nr, nc] = next_cost
            parent[nr, nc] = (row, col)
            heuristic = float(np.hypot(goal_cell[0] - nr, goal_cell[1] - nc))
            heapq.heappush(queue, (next_cost + heuristic, next_cost, nr, nc))

    if not np.isfinite(costs[goal_cell]):
        return None
    cells = [goal_cell]
    while cells[-1] != start_cell:
        row, col = cells[-1]
        previous = tuple(parent[row, col])
        if previous[0] < 0:
            return None
        cells.append(previous)
    cells.reverse()
    raw = np.asarray([[xs[col], ys[row]] for row, col in cells], dtype=float)
    raw[0] = np.asarray(start, dtype=float)[:2]
    raw[-1] = np.asarray(goal, dtype=float)[:2]

    # Keep the full grid route. Long line-of-sight shortcuts can cut too close
    # to a rectangle and leave a local CBF at a barrier boundary.
    return raw


def _reference_target(path: np.ndarray, index: int, x: np.ndarray, *, waypoint_tol: float, lookahead: float) -> tuple[np.ndarray, int]:
    # Look-ahead can intentionally skip grid points. Re-anchor progress on the
    # closest remaining path point so the target never stays behind the robot.
    remaining_distances = np.linalg.norm(path[index:] - x[:2], axis=1)
    index += int(np.argmin(remaining_distances))
    while index < len(path) - 1 and np.linalg.norm(x[:2] - path[index]) <= waypoint_tol:
        index += 1
    target = index
    remaining = 0.0
    while target < len(path) - 1 and remaining < lookahead:
        remaining += float(np.linalg.norm(path[target + 1] - path[target]))
        target += 1
    return path[target], index


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

    gain = env_float("SOFAI_CBF_REFERENCE_GAIN", 2.5)
    base = np.clip(gain * goal_vec - drift, -robot.robot_spec["v_max"], robot.robot_spec["v_max"])
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
        cand = gain * _rotate(goal_vec, angle) - drift
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
        dt = env_float("SOFAI_CBF_DT", 0.075)
        n_steps = env_int("SOFAI_CBF_STEPS", 900)
        goal_tol = scenario_goal_tol(scenario, env_float("SOFAI_CBF_GOAL_TOL", 0.5))
        margin = env_float("SOFAI_CBF_MARGIN", 0.20)
        exponent = env_float("SOFAI_CBF_RECT_EXPONENT", 10.0)
        u_max = scenario_u_max(scenario)

        robot, robot_spec = _build_robot(start, dt, u_max, scenario, len(rects))
        controller = CBFQP(robot, robot_spec, num_obs=max(1, len(rects)))
        obstacles = [
            _rect_to_cbf_obstacle(r, margin=margin, exponent=exponent)
            for r in rects
        ]
        reference_path = None
        if env_bool("SOFAI_CBF_GLOBAL_REFERENCE", True):
            reference_path = build_global_reference_path(
                start,
                goal,
                rects,
                scenario_bounds(scenario),
                clearance=robot.robot_radius + margin,
                resolution=env_float("SOFAI_CBF_REFERENCE_GRID", 0.35),
            )
        reference_index = 0

        states = [robot.X.reshape(-1).copy()]
        inputs = []
        qp_failed = False
        t0 = time.perf_counter()

        for _ in range(int(n_steps)):
            if goal_reached(np.asarray(states, dtype=float), goal, goal_tol):
                break

            reference_goal = goal
            if reference_path is not None:
                reference_goal, reference_index = _reference_target(
                    reference_path,
                    reference_index,
                    robot.X.reshape(-1),
                    waypoint_tol=env_float("SOFAI_CBF_WAYPOINT_TOL", 0.55),
                    lookahead=env_float("SOFAI_CBF_REFERENCE_LOOKAHEAD", 1.4),
                )
            u_ref = _choose_reference_control(robot, reference_goal, rects, goal_tol)
            u = controller.solve_control_problem(
                robot.X,
                {"u_ref": u_ref},
                obstacles if obstacles else None,
            )
            if u is None:
                # Do not silently execute an unconstrained nominal command when
                # the safety QP fails. Returning an unsuccessful rollout is
                # preferable to reporting an unsafe S2 solution as valid.
                qp_failed = True
                break

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
            "dt": float(dt),
            "collision_free": bool(collision_free),
            "solved": bool(solved),
            "runtime": runtime,
            "message": ("safety_qp_failed" if qp_failed else ("success" if solved else "failed_to_reach_goal")) + (";global_reference" if reference_path is not None else ";local_reference"),
            "reference_path": reference_path,
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
