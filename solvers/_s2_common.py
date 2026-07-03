from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else float(default)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


def scenario_rects(scenario) -> list[Rect]:
    return [tuple(map(float, r)) for r in getattr(scenario, "rects", [])]


def scenario_start(scenario) -> np.ndarray:
    return np.asarray(getattr(scenario, "start", (0.0, 0.0)), dtype=float).reshape(-1)


def scenario_goal(scenario) -> np.ndarray:
    return np.asarray(getattr(scenario, "goal", (0.0, 0.0)), dtype=float).reshape(-1)


def scenario_bounds(scenario) -> Rect:
    return tuple(map(float, getattr(scenario, "bounds", (-10.0, -10.0, 10.0, 10.0))))  # type: ignore[return-value]


def scenario_u_max(scenario, default: float = 3.0) -> float:
    return float(getattr(scenario, "u_max", default))


def scenario_goal_tol(scenario, default: float = 0.5) -> float:
    return float(getattr(scenario, "goal_tol", default))


def collision_free_rectangles(states: np.ndarray, rects: Sequence[Rect], margin: float = 0.0) -> bool:
    xy = np.asarray(states, dtype=float)[:, :2]
    xs = xy[:, 0]
    ys = xy[:, 1]
    m = float(margin)
    for xmin, ymin, xmax, ymax in rects:
        inside = (xs >= xmin - m) & (xs <= xmax + m) & (ys >= ymin - m) & (ys <= ymax + m)
        if np.any(inside):
            return False
    return True


def goal_reached(states: np.ndarray, goal: Sequence[float], tol: float = 0.5) -> bool:
    xy = np.asarray(states, dtype=float)[:, :2]
    dx = float(xy[-1, 0]) - float(goal[0])
    dy = float(xy[-1, 1]) - float(goal[1])
    return dx * dx + dy * dy <= float(tol) * float(tol)


def maybe_patch_goal_trajectory(
    states: np.ndarray,
    goal: Sequence[float],
    goal_tol: float,
    *,
    patch_tol: float | None = None,
) -> np.ndarray:
    X = np.asarray(states, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return X

    tol = float(patch_tol if patch_tol is not None else max(float(goal_tol), 0.75))
    goal_xy = np.asarray(goal, dtype=float).reshape(-1)[:2]
    if np.linalg.norm(X[-1, :2] - goal_xy) > tol:
        return X

    goal_state = np.array(X[-1], copy=True)
    goal_state[:2] = goal_xy
    return np.vstack([X, goal_state[None, :]]).astype(np.float32)


def rect_to_superellipse(
    rect: Rect,
    *,
    robot_radius: float,
    margin: float,
    exponent: float,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = rect
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    ax = max(0.5 * (xmax - xmin) - float(robot_radius) - float(margin), 1e-3)
    ay = max(0.5 * (ymax - ymin) - float(robot_radius) - float(margin), 1e-3)
    return np.array([cx, cy, ax, ay, float(exponent), 0.0, 1.0], dtype=float)


def acados_root_candidates() -> list[Path]:
    candidates = []
    env_root = os.environ.get("ACADOS_SOURCE_DIR")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/private/tmp/acados-install"))
    candidates.append(Path(__file__).resolve().parents[1] / "safe_control" / "acados")
    return candidates


def detect_acados_root() -> Path | None:
    for root in acados_root_candidates():
        if (root / "lib" / "libacados.dylib").is_file():
            return root
    return None


def bootstrap_acados_backend() -> Path | None:
    root = detect_acados_root()
    if root is None:
        return None

    source_root = Path(__file__).resolve().parents[1] / "safe_control" / "acados"
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(root))
    if source_root.exists():
        os.environ.setdefault("ACADOS_PYTHON_INTERFACE_PATH", str(source_root / "interfaces" / "acados_template" / "acados_template"))
    tera_path = root / "bin" / "t_renderer"
    if tera_path.is_file():
        os.environ.setdefault("TERA_PATH", str(tera_path))

    lib_dir = root / "lib"
    if lib_dir.is_dir():
        current = os.environ.get("DYLD_LIBRARY_PATH", "")
        parts = [str(lib_dir)]
        if current:
            parts.append(current)
        os.environ["DYLD_LIBRARY_PATH"] = ":".join(parts)

        current_fb = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        fb_parts = [str(lib_dir)]
        if current_fb:
            fb_parts.append(current_fb)
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(fb_parts)

        for lib_name in ("libacados.dylib", "libblasfeo.dylib", "libhpipm.dylib"):
            lib_path = lib_dir / lib_name
            if lib_path.is_file():
                try:
                    ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                except Exception:
                    pass
    return root


def ensure_acados_template_path() -> Path:
    source_root = Path(__file__).resolve().parents[1] / "safe_control" / "acados"
    backend_root = bootstrap_acados_backend() or source_root
    template_path = source_root / "interfaces" / "acados_template"

    if template_path.exists():
        template_path_str = str(template_path)
        if template_path_str not in sys.path:
            sys.path.insert(0, template_path_str)

    return template_path
