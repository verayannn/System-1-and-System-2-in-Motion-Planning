from __future__ import annotations

import os
import sys
import ctypes
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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


def benchmark_family_from_dictionary(dictionary_name: str) -> str:
    stem = Path(str(dictionary_name)).stem
    for marker in ("_eval_", "_train_"):
        if marker in stem:
            return stem.split(marker, 1)[1] or stem
    return stem


def quality_weights_for_family(family: str) -> Dict[str, float]:
    family = str(family).strip().lower()
    presets: Dict[str, Dict[str, float]] = {
        # Trajectory-shape heavy tasks.
        "dense_clutter": {"path_length": 0.25, "control_effort": 0.15, "smoothness": 0.60},
        "bugtrap": {"path_length": 0.20, "control_effort": 0.10, "smoothness": 0.70},
        "maze_branching": {"path_length": 0.25, "control_effort": 0.10, "smoothness": 0.65},
        "serial_walls": {"path_length": 0.25, "control_effort": 0.15, "smoothness": 0.60},
        "wall_gap": {"path_length": 0.25, "control_effort": 0.15, "smoothness": 0.60},
        # Dynamics-dominated open spaces can afford slightly more effort sensitivity.
        "small_open": {"path_length": 0.35, "control_effort": 0.20, "smoothness": 0.45},
        "large_sparse": {"path_length": 0.30, "control_effort": 0.20, "smoothness": 0.50},
    }
    weights = dict(presets.get(family, {"path_length": 0.25, "control_effort": 0.15, "smoothness": 0.60}))
    total = float(sum(weights.values()))
    if total <= 0.0:
        return {"path_length": 1.0 / 3.0, "control_effort": 1.0 / 3.0, "smoothness": 1.0 / 3.0}
    for key in weights:
        weights[key] = float(weights[key]) / total
    return weights


def _turning_smoothness(states: np.ndarray) -> float:
    xy = np.asarray(states, dtype=float)[:, :2]
    if xy.shape[0] < 3:
        return 0.0
    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    mask = (n1 > 1e-9) & (n2 > 1e-9)
    if not np.any(mask):
        return 0.0
    cos = np.sum(v1[mask] * v2[mask], axis=1) / (n1[mask] * n2[mask])
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.arccos(cos)
    return float(np.sum(np.square(angles)))


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


def selected_success_attempt(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = result.get("attempts", []) or []
    selected_name = str(result.get("selected_attempt") or "").strip()
    if selected_name:
        for attempt in attempts:
            if str(attempt.get("name")) == selected_name and bool(attempt.get("success")):
                return attempt
    for attempt in attempts:
        if bool(attempt.get("success")):
            return attempt
    return None


def _scenario_matrix(scenario: Dict[str, Any] | None, *names: str) -> Optional[np.ndarray]:
    if not isinstance(scenario, dict):
        return None
    for name in names:
        value = scenario.get(name)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.ndim == 2 and arr.size:
            return arr
    return None


def _controls_from_attempt(
    result: Dict[str, Any],
    attempt: Dict[str, Any],
    *,
    dt: float,
) -> np.ndarray:
    inputs = attempt.get("inputs")
    if inputs is not None:
        try:
            arr = np.asarray(inputs, dtype=float)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return arr
        except Exception:
            pass

    states = np.asarray(attempt.get("states", []), dtype=float)
    if states.ndim != 2 or states.shape[0] < 2:
        return np.zeros((0, 2), dtype=float)

    scenario = result.get("scenario")
    A = _scenario_matrix(scenario, "A_query", "A")
    B = _scenario_matrix(scenario, "B_query", "B")
    if A is None or B is None:
        return np.diff(states[:, :2], axis=0) / max(float(dt), 1e-6)

    xy = states[:, :2]
    dx = (xy[1:] - xy[:-1]) / max(float(dt), 1e-6)
    drift = xy[:-1] @ A.T
    residual = dx - drift
    try:
        u = residual @ np.linalg.pinv(B).T
    except Exception:
        u = residual
    return np.asarray(u, dtype=float)


def trajectory_quality_components(result: Dict[str, Any], *, dt: float = 0.05) -> Optional[Dict[str, float]]:
    attempt = selected_success_attempt(result)
    if attempt is None:
        return None

    states = np.asarray(attempt.get("states", []), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        return None

    xy = states[:, :2]
    path_length = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()) if xy.shape[0] > 1 else 0.0
    controls = _controls_from_attempt(result, attempt, dt=dt)
    control_effort = float(np.sum(np.sum(np.square(controls), axis=1))) if controls.size else 0.0
    smoothness = _turning_smoothness(xy)
    if smoothness <= 0.0 and controls.shape[0] > 1:
        smoothness = float(np.sum(np.sum(np.square(np.diff(controls, axis=0)), axis=1)))

    return {
        "path_length": path_length,
        "control_effort": control_effort,
        "smoothness": smoothness,
    }


def quality_reference_values(samples: Sequence[Dict[str, float]]) -> Dict[str, float]:
    refs: Dict[str, float] = {}
    for key in ("path_length", "control_effort", "smoothness"):
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None and float(sample[key]) > 0.0]
        refs[key] = float(np.median(values)) if values else 1.0
        refs[key] = max(refs[key], 1e-9)
    return refs


def quality_score(sample: Dict[str, float], refs: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    if weights is None:
        weights = {"path_length": 1.0 / 3.0, "control_effort": 1.0 / 3.0, "smoothness": 1.0 / 3.0}
    total = float(sum(float(weights.get(key, 0.0)) for key in ("path_length", "control_effort", "smoothness")))
    if total <= 0.0:
        weights = {"path_length": 1.0 / 3.0, "control_effort": 1.0 / 3.0, "smoothness": 1.0 / 3.0}
        total = 1.0
    j = (
        float(weights.get("path_length", 0.0)) * float(sample["path_length"]) / float(refs["path_length"])
        + float(weights.get("control_effort", 0.0)) * float(sample["control_effort"]) / float(refs["control_effort"])
        + float(weights.get("smoothness", 0.0)) * float(sample["smoothness"]) / float(refs["smoothness"])
    ) / total
    return 1.0 / (1.0 + j)


def quality_refs_for_result(result: Dict[str, Any]) -> Dict[str, float]:
    selected = selected_success_attempt(result)
    scenario = result.get("scenario") if isinstance(result, dict) else None
    if selected is None or not isinstance(scenario, dict):
        return {"path_length": 1.0, "control_effort": 1.0, "smoothness": 1.0}

    states = np.asarray(selected.get("states", []), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        return {"path_length": 1.0, "control_effort": 1.0, "smoothness": 1.0}

    start = np.asarray(scenario.get("start", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
    goal = np.asarray(scenario.get("goal", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
    path_ref = max(float(np.linalg.norm(goal - start)), 1e-6)
    u_max = float(scenario.get("u_max", 3.0))
    effort_ref = max(path_ref * max(int(states.shape[0]) - 1, 1) * (u_max**2), 1e-6)
    return {
        "path_length": path_ref,
        "control_effort": effort_ref,
        "smoothness": 1.0,
    }


def acados_root_candidates() -> list[Path]:
    candidates = []
    for env_name in ("ACADOS_SOURCE_DIR", "ACADOS_INSTALL_DIR", "ACADOS_ROOT"):
        env_root = os.environ.get(env_name)
        if env_root:
            candidates.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "safe_control" / "acados",
            repo_root / ".acados",
            Path(tempfile.gettempdir()) / "acados-install",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


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
    env_template = os.environ.get("ACADOS_PYTHON_INTERFACE_PATH")
    template_candidates = []
    if env_template:
        template_env_path = Path(env_template).expanduser()
        template_candidates.extend([template_env_path, template_env_path.parent])
    template_candidates.extend(
        [
            backend_root / "interfaces" / "acados_template",
            source_root / "interfaces" / "acados_template",
        ]
    )

    template_path = template_candidates[-1]
    for candidate in template_candidates:
        if candidate.exists():
            template_path = candidate
            break

    if template_path.exists():
        template_path_str = str(template_path)
        if template_path_str not in sys.path:
            sys.path.insert(0, template_path_str)

    return template_path
