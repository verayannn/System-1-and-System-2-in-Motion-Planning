from __future__ import annotations

import os
import sys
import ctypes
import shutil
import tempfile
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


def resolve_mplconfigdir(root: Path | None = None, requested: str | None = None) -> Path:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())

    env_value = os.environ.get("MPLCONFIGDIR")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    if root is not None:
        candidates.append(Path(root).expanduser() / ".cache" / "matplotlib")

    candidates.append(Path(tempfile.gettempdir()) / "matplotlib")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate

    fallback = Path(tempfile.mkdtemp(prefix="matplotlib-"))
    return fallback


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
    candidates.append(Path(tempfile.gettempdir()) / "acados-install")
    candidates.append(Path(__file__).resolve().parents[1] / "safe_control" / "acados")
    return candidates


def _shared_library_patterns(base: str) -> tuple[str, ...]:
    if sys.platform == "darwin":
        return (f"{base}.dylib", f"{base}.dylib.*", f"{base}.so", f"{base}.so.*")
    if sys.platform.startswith("linux"):
        return (f"{base}.so", f"{base}.so.*", f"{base}.dylib", f"{base}.dylib.*")
    return (f"{base}.so", f"{base}.so.*", f"{base}.dylib", f"{base}.dylib.*")


def _shared_library_paths(lib_dir: Path, base: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in _shared_library_patterns(base):
        for path in sorted(lib_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def detect_acados_root() -> Path | None:
    for root in acados_root_candidates():
        lib_dir = root / "lib"
        if any(_shared_library_paths(lib_dir, "libacados")):
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
    lib_dir = root / "lib"
    src_lib_dir = source_root / "lib"
    if src_lib_dir.is_dir():
        lib_dir.mkdir(parents=True, exist_ok=True)
        src_link_libs = src_lib_dir / "link_libs.json"
        dst_link_libs = lib_dir / "link_libs.json"
        if src_link_libs.is_file() and not dst_link_libs.is_file():
            shutil.copy2(src_link_libs, dst_link_libs)

    tera_path = root / "bin" / "t_renderer"
    if not tera_path.is_file():
        src_tera = source_root / "bin" / "t_renderer"
        if src_tera.is_file():
            tera_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_tera, tera_path)
            try:
                tera_path.chmod(0o755)
            except Exception:
                pass
    if tera_path.is_file():
        os.environ.setdefault("TERA_PATH", str(tera_path))

    if lib_dir.is_dir():
        path_key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        current = os.environ.get(path_key, "")
        parts = [str(lib_dir)]
        if current:
            parts.append(current)
        os.environ[path_key] = os.pathsep.join(parts)

        if sys.platform == "darwin":
            current_fb = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            fb_parts = [str(lib_dir)]
            if current_fb:
                fb_parts.append(current_fb)
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(fb_parts)

        for base_name in ("libacados", "libblasfeo", "libhpipm"):
            for lib_path in _shared_library_paths(lib_dir, base_name):
                try:
                    ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                    break
                except Exception:
                    continue
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
