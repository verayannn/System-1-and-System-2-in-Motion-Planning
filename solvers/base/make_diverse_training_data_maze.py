"""
make_diverse_training_data_maze.py

Generate shared System-1 data for maze motion planning.

This script generates one common successful-trajectory set and derives both
System-1 artifacts from it:

    1. S1 retrieval database trajectories:
       db/s1_sfcbf_success_trajs.npz
       db/S1_database_maze.json

    2. Neural S1 training windows:
       db/nn_dataset_maze.npz

The neural dataset is extracted from the same trajectories stored in the S1
database NPZ, so primitive S1 and neural S1 train/evaluate from the same expert
motion primitives.

The neural NPZ is the format used by train_nn_policy.py:
    ctx, sit, dyn, goal, u, next_local, norms, meta

Designed for your current setting:
    x in R^2
    A in R^{2x2}
    B in R^{2x2}
    u in R^2
    dyn_feat = [A_local.flatten(), B_local.flatten(), drift_local]
             = 4 + 4 + 2 = 10 dimensions

Example:


PYTHONPYCACHEPREFIX=/private/tmp/pycache-check \
PYTHONDONTWRITEBYTECODE=1 \
python3 solvers/base/make_diverse_training_data_maze.py \
  --target_trajectories 500 \
  --out_npz db/nn_dataset_maze.npz \
  --traj_out db/s1_sfcbf_success_trajs.npz \
  --db_out db/S1_database_maze.json \
  --scenarios_out db/benchmark_scenarios_maze_s1_db.json \
  --report_out output/diversity_report_s1_db.json \
  --L_c 20 \
  --stride 1 \
  --seed 7

Then train 

"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]


# =============================================================================
# Paths
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
SOLVER_DIR = THIS_DIR.parent
INSTANCE_DIR = SOLVER_DIR.parent


def resolve_output_path(path_like: str) -> Path:
    """Keep generated maze S1 artifacts under this repo unless an absolute path is given."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) >= 3 and path.parts[0] == "maze" and path.parts[1] == "n_s1":
        return SOLVER_DIR / Path(*path.parts[2:])
    return INSTANCE_DIR / path


# =============================================================================
# Import System-2 expert
# =============================================================================

def _setup_imports():
    candidates = [
        THIS_DIR,
        SOLVER_DIR,
        INSTANCE_DIR,
        Path.cwd(),
    ]
    for p in candidates:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

_setup_imports()

try:
    import S2_cbf_maze as s2
except ImportError:
    try:
        from maze.n_s1 import S2_cbf_maze as s2
    except ImportError as e:
        raise ImportError(
            "Could not import S2_cbf_maze.py. Keep it in Solvers/Base/ "
            "or run this script from the mpc-sofai instance root."
        ) from e


# =============================================================================
# Basic shape / dynamics helpers
# =============================================================================

def ensure_A_B(A: Any, B: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Current intended setting:
        A: 2x2
        B: 2x2
        u: 2D

    This function is strict: it will not silently convert B to 2x1.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != (2, 2):
        if A.size == 4:
            A = A.reshape(2, 2)
        else:
            raise ValueError(f"Expected A shape (2,2), got {A.shape}")

    if B.shape != (2, 2):
        if B.size == 4:
            B = B.reshape(2, 2)
        else:
            raise ValueError(
                f"This diverse-data generator is configured for B shape (2,2), got {B.shape}. "
                "If you want scalar input, use a separate 2x1 pipeline."
            )

    return A.astype(float), B.astype(float)


def is_hurwitz(A: np.ndarray) -> bool:
    return bool(np.all(np.real(np.linalg.eigvals(A)) < 0.0))


def stabilize_A(A: np.ndarray, margin: float = 0.05) -> np.ndarray:
    lam = np.linalg.eigvals(A)
    max_real = float(np.max(np.real(lam)))
    if max_real >= -margin:
        A = A - (max_real + margin) * np.eye(2)
    return A


def sample_A(rng: np.random.Generator, regime: str) -> np.ndarray:
    """
    Diverse stable 2x2 dynamics.

    Regimes intentionally cover different qualitative motions:
    - damped: mostly diagonal stable
    - rotate_cw / rotate_ccw: strong swirl
    - shear: non-normal dynamics
    - anisotropic: one slow/one fast direction
    - mixed: random stable with rotation + shear
    """
    if regime == "damped":
        d1 = rng.uniform(0.25, 1.4)
        d2 = rng.uniform(0.25, 1.4)
        A = np.array([[-d1, 0.0], [0.0, -d2]])
        A += rng.normal(scale=0.08, size=(2, 2))

    elif regime == "rotate_cw":
        damping = rng.uniform(0.15, 0.9)
        w = rng.uniform(0.4, 2.2)
        A = np.array([[-damping, w], [-w, -damping]])
        A += rng.normal(scale=0.08, size=(2, 2))

    elif regime == "rotate_ccw":
        damping = rng.uniform(0.15, 0.9)
        w = rng.uniform(0.4, 2.2)
        A = np.array([[-damping, -w], [w, -damping]])
        A += rng.normal(scale=0.08, size=(2, 2))

    elif regime == "shear":
        d1 = rng.uniform(0.25, 1.0)
        d2 = rng.uniform(0.25, 1.0)
        sh = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.5)
        A = np.array([[-d1, sh], [0.0, -d2]])
        if rng.random() < 0.5:
            A = A.T
        A += rng.normal(scale=0.05, size=(2, 2))

    elif regime == "anisotropic":
        slow = rng.uniform(0.05, 0.25)
        fast = rng.uniform(0.8, 2.0)
        theta = rng.uniform(-np.pi, np.pi)
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        A = Q @ np.diag([-slow, -fast]) @ Q.T
        A += rng.normal(scale=0.05, size=(2, 2))

    elif regime == "mixed":
        M = rng.normal(size=(2, 2))
        # Shift eigenvalues to stable side.
        A = M - (1.1 + abs(float(np.max(np.real(np.linalg.eigvals(M)))))) * np.eye(2)
        A += rng.normal(scale=0.15, size=(2, 2))

    else:
        raise ValueError(f"Unknown dynamics regime: {regime}")

    A = stabilize_A(A, margin=0.05)
    return A.astype(float)


def sample_B(rng: np.random.Generator, mode: str = "random_invertible") -> np.ndarray:
    """
    Returns B in R^{2x2}. Different B's increase control-direction diversity.
    """
    if mode == "identity":
        return np.eye(2, dtype=float)

    if mode == "axis_scaled":
        gains = rng.uniform(0.6, 1.8, size=2)
        signs = rng.choice([-1.0, 1.0], size=2)
        return np.diag(gains * signs).astype(float)

    if mode == "rotated_scaled":
        theta = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        gains = rng.uniform(0.6, 1.8, size=2)
        return (R @ np.diag(gains)).astype(float)

    if mode == "random_invertible":
        for _ in range(100):
            B = np.eye(2) + rng.normal(scale=0.35, size=(2, 2))
            if abs(np.linalg.det(B)) > 0.25 and np.linalg.cond(B) < 8.0:
                return B.astype(float)
        return np.eye(2, dtype=float)

    raise ValueError(f"Unknown B mode: {mode}")


# =============================================================================
# Situation vector and local-frame dataset features
# =============================================================================

def _dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    for _ in range(int(iters)):
        m = np.maximum.reduce([
            m,
            np.roll(m, 1, 0),
            np.roll(m, -1, 0),
            np.roll(m, 1, 1),
            np.roll(m, -1, 1),
        ])
    return m


def compute_situation_vector(
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    bounds: List[float],
    start: List[float],
    goal: List[float],
    grid_n: int,
    dt_nom: float,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = map(float, bounds)
    x = np.asarray(start, dtype=float).reshape(2)
    g = np.asarray(goal, dtype=float).reshape(2)
    A, B = ensure_A_B(A, B)

    visited = np.zeros((grid_n, grid_n), dtype=np.uint8)

    for _ in range(int(n_steps_nom)):
        if np.linalg.norm(x - g) <= stop_tol:
            break

        rhs = -(x - g) - A @ x
        try:
            u = np.linalg.lstsq(B, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            u = np.zeros(B.shape[1], dtype=float)

        u = np.clip(u, -u_max_nom, u_max_nom)
        x = x + dt_nom * (A @ x + B @ u)

        if not (xmin <= x[0] <= xmax and ymin <= x[1] <= ymax):
            continue

        i = int((x[1] - ymin) / (ymax - ymin) * grid_n)
        j = int((x[0] - xmin) / (xmax - xmin) * grid_n)
        i = np.clip(i, 0, grid_n - 1)
        j = np.clip(j, 0, grid_n - 1)
        visited[i, j] = 1

    corridor = _dilate4(visited, buffer_cells)

    obs = np.zeros((grid_n, grid_n), dtype=np.uint8)
    dx = (xmax - xmin) / grid_n
    dy = (ymax - ymin) / grid_n

    for rx1, ry1, rx2, ry2 in rects:
        x1, x2 = min(rx1, rx2), max(rx1, rx2)
        y1, y2 = min(ry1, ry2), max(ry1, ry2)
        j1 = int((x1 - xmin) / dx)
        j2 = int((x2 - xmin) / dx)
        i1 = int((y1 - ymin) / dy)
        i2 = int((y2 - ymin) / dy)
        i1 = np.clip(i1, 0, grid_n - 1)
        i2 = np.clip(i2, 0, grid_n - 1)
        j1 = np.clip(j1, 0, grid_n - 1)
        j2 = np.clip(j2, 0, grid_n - 1)
        obs[i1:i2 + 1, j1:j2 + 1] = 1

    return np.maximum(visited, obs & corridor).reshape(-1).astype(np.float32)


def compute_heading_from_context(ctx_pos: np.ndarray, goal: Optional[np.ndarray] = None) -> float:
    if ctx_pos.shape[0] >= 2:
        v = ctx_pos[-1] - ctx_pos[-2]
        if np.linalg.norm(v) > 1e-6:
            return float(np.arctan2(v[1], v[0]))
    if goal is not None:
        vg = goal[:2] - ctx_pos[-1]
        if np.linalg.norm(vg) > 1e-8:
            return float(np.arctan2(vg[1], vg[0]))
    return 0.0


def rotation_from_heading(heading: float) -> np.ndarray:
    c = np.cos(-heading)
    sn = np.sin(-heading)
    return np.array([[c, -sn], [sn, c]], dtype=float)


def to_local_frame(context_pos: np.ndarray, next_pos: np.ndarray, goal: Optional[np.ndarray] = None):
    origin = context_pos[-1].astype(float)
    heading = compute_heading_from_context(context_pos, goal)
    R = rotation_from_heading(heading)
    ctx_local = (context_pos - origin) @ R.T
    next_local = (next_pos - origin) @ R.T
    goal_local = None if goal is None else (goal[:2] - origin) @ R.T
    return ctx_local, next_local, goal_local, origin, heading


def local_dynamics_features(A: np.ndarray, B: np.ndarray, heading: float, origin: np.ndarray) -> np.ndarray:
    A, B = ensure_A_B(A, B)
    R = rotation_from_heading(heading)
    A_l = R @ A @ R.T
    B_l = R @ B
    drift_l = R @ (A @ origin)
    return np.concatenate([
        A_l.reshape(-1),
        B_l.reshape(-1),
        drift_l.reshape(-1),
    ], axis=0).astype(np.float32)


def recover_control_from_transition(x_curr: np.ndarray, x_next: np.ndarray, A: np.ndarray, B: np.ndarray, dt: float, u_clip: float) -> np.ndarray:
    A, B = ensure_A_B(A, B)
    rhs = (x_next - x_curr) / dt - A @ x_curr
    try:
        u = np.linalg.lstsq(B, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        u = np.zeros(B.shape[1], dtype=float)
    return np.clip(u, -u_clip, u_clip).astype(np.float32)


# =============================================================================
# Obstacle-map sampling
# =============================================================================

def rect_contains_point(rect: Rect, p: np.ndarray, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = rect
    return (min(x1, x2) - margin <= p[0] <= max(x1, x2) + margin and
            min(y1, y2) - margin <= p[1] <= max(y1, y2) + margin)


def valid_rect(rect: Rect, start: np.ndarray, goal: np.ndarray, bounds: List[float], min_clearance: float = 0.8) -> bool:
    xmin, ymin, xmax, ymax = bounds
    x1, y1, x2, y2 = rect
    if min(x1, x2) <= xmin or max(x1, x2) >= xmax or min(y1, y2) <= ymin or max(y1, y2) >= ymax:
        return False
    if rect_contains_point(rect, start, margin=min_clearance):
        return False
    if rect_contains_point(rect, goal, margin=min_clearance):
        return False
    return True


def random_rect(rng: np.random.Generator, bounds: List[float], size_range: Tuple[float, float]) -> Rect:
    xmin, ymin, xmax, ymax = bounds
    w = rng.uniform(size_range[0], size_range[1])
    h = rng.uniform(size_range[0], size_range[1])
    cx = rng.uniform(xmin + w / 2 + 0.5, xmax - w / 2 - 0.5)
    cy = rng.uniform(ymin + h / 2 + 0.5, ymax - h / 2 - 0.5)
    return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


def sample_scattered_map(rng: np.random.Generator, bounds: List[float], start: np.ndarray, goal: np.ndarray, difficulty: str) -> List[Rect]:
    if difficulty == "easy":
        n_rects = int(rng.integers(2, 5))
        size_range = (0.5, 1.5)
    elif difficulty == "medium":
        n_rects = int(rng.integers(4, 8))
        size_range = (0.8, 2.2)
    else:
        n_rects = int(rng.integers(7, 12))
        size_range = (1.0, 3.0)

    rects: List[Rect] = []
    tries = 0
    while len(rects) < n_rects and tries < 300:
        tries += 1
        r = random_rect(rng, bounds, size_range)
        if valid_rect(r, start, goal, bounds, min_clearance=0.9):
            rects.append(r)
    return rects


def sample_wall_gap_map(rng: np.random.Generator, bounds: List[float], start: np.ndarray, goal: np.ndarray, difficulty: str) -> List[Rect]:
    xmin, ymin, xmax, ymax = bounds
    vertical = bool(rng.random() < 0.5)
    thickness = rng.uniform(0.35, 0.8)
    gap = {"easy": 4.0, "medium": 2.8, "hard": 1.8}[difficulty]
    gap_center = rng.uniform(-4.0, 4.0)

    rects: List[Rect] = []
    if vertical:
        x = rng.uniform(-3.5, 3.5)
        rects.append((x - thickness / 2, ymin + 0.8, x + thickness / 2, gap_center - gap / 2))
        rects.append((x - thickness / 2, gap_center + gap / 2, x + thickness / 2, ymax - 0.8))
    else:
        y = rng.uniform(-3.5, 3.5)
        rects.append((xmin + 0.8, y - thickness / 2, gap_center - gap / 2, y + thickness / 2))
        rects.append((gap_center + gap / 2, y - thickness / 2, xmax - 0.8, y + thickness / 2))

    return [r for r in rects if valid_rect(r, start, goal, bounds, min_clearance=0.7)]


def sample_bottleneck_map(rng: np.random.Generator, bounds: List[float], start: np.ndarray, goal: np.ndarray, difficulty: str) -> List[Rect]:
    rects = sample_wall_gap_map(rng, bounds, start, goal, difficulty)
    # Add a few scattered distractors.
    extra = sample_scattered_map(rng, bounds, start, goal, "easy" if difficulty == "easy" else "medium")
    rng.shuffle(extra)
    return rects + extra[: int(rng.integers(1, 4))]


def sample_u_shape_map(rng: np.random.Generator, bounds: List[float], start: np.ndarray, goal: np.ndarray, difficulty: str) -> List[Rect]:
    # U shape placed away from start/goal with random orientation. Approximate with 3 rectangles.
    cx = rng.uniform(-4.0, 3.0)
    cy = rng.uniform(-4.0, 3.0)
    width = rng.uniform(2.5, 4.5)
    height = rng.uniform(2.5, 4.5)
    th = rng.uniform(0.35, 0.75)

    # Axis-aligned U opening either up/down/left/right.
    orient = rng.choice(["up", "down", "left", "right"])
    rects: List[Rect] = []
    if orient in ("up", "down"):
        # side bars + bottom/top bar
        rects.append((cx - width / 2, cy - height / 2, cx - width / 2 + th, cy + height / 2))
        rects.append((cx + width / 2 - th, cy - height / 2, cx + width / 2, cy + height / 2))
        if orient == "up":
            rects.append((cx - width / 2, cy - height / 2, cx + width / 2, cy - height / 2 + th))
        else:
            rects.append((cx - width / 2, cy + height / 2 - th, cx + width / 2, cy + height / 2))
    else:
        rects.append((cx - width / 2, cy - height / 2, cx + width / 2, cy - height / 2 + th))
        rects.append((cx - width / 2, cy + height / 2 - th, cx + width / 2, cy + height / 2))
        if orient == "left":
            rects.append((cx + width / 2 - th, cy - height / 2, cx + width / 2, cy + height / 2))
        else:
            rects.append((cx - width / 2, cy - height / 2, cx - width / 2 + th, cy + height / 2))

    rects = [r for r in rects if valid_rect(r, start, goal, bounds, min_clearance=0.8)]
    if difficulty == "hard":
        rects += sample_scattered_map(rng, bounds, start, goal, "easy")[:2]
    return rects


BENCHMARK_MAP_TYPES = {
    "small_open",
    "large_sparse",
    "dense_clutter",
    "wall_gap",
    "serial_walls",
    "maze_branching",
    "bugtrap",
}


def sample_benchmark_family_map(
    rng: np.random.Generator,
    map_type: str,
    bounds: List[float],
    start: np.ndarray,
    goal: np.ndarray,
) -> List[Rect]:
    """Sample one of the benchmark-family maps used by input/generate_benchmark_dictionaries.py."""
    try:
        from input import generate_benchmark_dictionaries as bench_gen
    except Exception as exc:
        raise ImportError(
            "Could not import input/generate_benchmark_dictionaries.py from the SOFAI root."
        ) from exc

    builders = {family.name: family.builder for family in bench_gen.FAMILIES}
    if map_type not in builders:
        raise ValueError(f"Unknown benchmark map type: {map_type}")

    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))
    rects = [tuple(map(float, r)) for r in builders[map_type](py_rng, 0)]
    return [
        r for r in rects
        if valid_rect(r, start, goal, bounds, min_clearance=0.65)
    ]


def sample_map(rng: np.random.Generator, map_type: str, difficulty: str, bounds: List[float], start: np.ndarray, goal: np.ndarray) -> List[Rect]:
    if map_type == "empty":
        return []
    if map_type in BENCHMARK_MAP_TYPES:
        return sample_benchmark_family_map(rng, map_type, bounds, start, goal)
    if map_type == "scattered":
        return sample_scattered_map(rng, bounds, start, goal, difficulty)
    if map_type == "wall_gap":
        return sample_wall_gap_map(rng, bounds, start, goal, difficulty)
    if map_type == "bottleneck":
        return sample_bottleneck_map(rng, bounds, start, goal, difficulty)
    if map_type == "u_shape":
        return sample_u_shape_map(rng, bounds, start, goal, difficulty)
    raise ValueError(f"Unknown map type: {map_type}")


# =============================================================================
# Dataset extraction from successful trajectories
# =============================================================================

def count_windows(T: int, L_c: int, stride: int) -> int:
    if T < L_c + 1:
        return 0
    return 1 + (T - L_c - 1) // stride


def extract_samples_from_trajectory(
    states: np.ndarray,
    inputs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    bounds: List[float],
    start: np.ndarray,
    goal: np.ndarray,
    dyn_id: int,
    map_idx: int,
    traj_id: int,
    L_c: int,
    stride: int,
    grid_n: int,
    dt_nom: float,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
    group_key: str,
) -> List[Dict[str, Any]]:
    A, B = ensure_A_B(A, B)
    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, B.shape[1])

    if len(states) < L_c + 1 or len(inputs) < len(states) - 1:
        return []

    sit_vec = compute_situation_vector(
        A=A,
        B=B,
        rects=rects,
        bounds=bounds,
        start=start.tolist(),
        goal=goal.tolist(),
        grid_n=grid_n,
        dt_nom=dt_nom,
        n_steps_nom=n_steps_nom,
        u_max_nom=u_max_nom,
        buffer_cells=buffer_cells,
        stop_tol=stop_tol,
    )

    samples = []
    T = states.shape[0]
    for s in range(0, T - L_c, stride):
        ctx = states[s:s + L_c, :2]
        x_next = states[s + L_c, :2]
        u_tgt = inputs[s + L_c - 1].reshape(B.shape[1])

        ctx_l, next_l, goal_l, origin, heading = to_local_frame(ctx, x_next, goal)
        dyn_feat = local_dynamics_features(A, B, heading, origin)

        samples.append({
            "ctx": ctx_l.astype(np.float32),
            "sit": sit_vec.astype(np.float32),
            "dyn": dyn_feat.astype(np.float32),
            "goal": goal_l.astype(np.float32),
            "u": u_tgt.astype(np.float32),
            "next_local": next_l.astype(np.float32),
            "traj_id": int(traj_id),
            "dyn_id": int(dyn_id),
            "map_idx": int(map_idx),
            "group": group_key,
        })

    return samples


def balanced_downsample(samples: List[Dict[str, Any]], target: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if len(samples) <= target:
        return samples

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[str(s["group"])].append(i)

    group_names = sorted(groups.keys())
    chosen: List[int] = []
    per_group_base = target // max(1, len(group_names))
    remainder = target % max(1, len(group_names))

    leftovers: List[int] = []
    for gi, g in enumerate(group_names):
        idxs = groups[g]
        rng.shuffle(idxs)
        quota = per_group_base + (1 if gi < remainder else 0)
        take = min(quota, len(idxs))
        chosen.extend(idxs[:take])
        leftovers.extend(idxs[take:])

    if len(chosen) < target:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: target - len(chosen)])

    rng.shuffle(chosen)
    return [samples[i] for i in chosen[:target]]


def system_to_M(A: np.ndarray, B: np.ndarray, omega0: float = 1.0) -> np.ndarray:
    """Match the approximate dynamics descriptor used by S1_usage_maze.py."""
    A, B = ensure_A_B(A, B)
    C = np.eye(2, dtype=float)
    jwI_minus_A = 1j * float(omega0) * np.eye(2) - A
    M_complex = C @ np.linalg.inv(jwI_minus_A) @ B
    return np.real(M_complex).astype(float)


def assign_cluster_id(A: np.ndarray, n_clusters: int) -> int:
    """Small deterministic dynamics grouping for the S1 retrieval DB."""
    eigvals = np.linalg.eigvals(A)
    max_real = float(np.max(np.real(eigvals)))
    imag_mag = float(np.max(np.abs(np.imag(eigvals))))
    shear_mag = float(abs(A[0, 1]) + abs(A[1, 0]))

    if n_clusters <= 1:
        return 0

    # Five coarse groups: slow/fast/rotational/shear/general. Modulo keeps the
    # function valid if the user requests fewer clusters.
    if imag_mag > 0.55:
        cid = 2
    elif shear_mag > 1.2:
        cid = 3
    elif max_real > -0.25:
        cid = 0
    elif max_real < -1.0:
        cid = 1
    else:
        cid = 4
    return int(cid % int(n_clusters))


def build_s1_retrieval_db(
    *,
    dyn_nodes: Dict[str, Dict[str, Any]],
    shared_maps: List[Dict[str, Any]],
    bounds: List[float],
    base_start: np.ndarray,
    base_goal: np.ndarray,
    start_goal_jitter: float,
    n_clusters: int,
    success_count: int,
) -> Dict[str, Any]:
    """Build DB JSON compatible with solvers/base/S1_usage_maze.py."""
    clusters: Dict[int, List[int]] = defaultdict(list)
    for dyn_id_str, node in dyn_nodes.items():
        dyn_id = int(dyn_id_str)
        cid = assign_cluster_id(np.asarray(node["A"], dtype=float), n_clusters=n_clusters)
        node["cluster_id"] = int(cid)
        clusters[int(cid)].append(dyn_id)

    consensus_nodes: Dict[str, Dict[str, Any]] = {}
    dyn_to_cluster: Dict[str, int] = {}
    for cid in sorted(clusters):
        children = sorted(clusters[cid])
        if not children:
            continue
        cd_dyn_id = children[0]
        M_cd = dyn_nodes[str(cd_dyn_id)]["M"]
        consensus_nodes[str(cid)] = {
            "cluster_id": int(cid),
            "cd_dyn_id": int(cd_dyn_id),
            "M_cd": M_cd,
            "dyn_children": [int(x) for x in children],
        }
        for dyn_id in children:
            dyn_to_cluster[str(dyn_id)] = int(cid)

    return {
        "meta": {
            "description": "S1 retrieval DB generated from the same successful trajectories used for neural S1 training",
            "B_shape": "2x2",
            "bounds": list(map(float, bounds)),
            "fixed_start": base_start.tolist(),
            "fixed_goal": base_goal.tolist(),
            "start_goal_jitter": float(start_goal_jitter),
            "successful_trajectories": int(success_count),
            "n_clusters": int(len(consensus_nodes)),
        },
        "shared_maps": shared_maps,
        "db": {
            "consensus_nodes": consensus_nodes,
            "dyn_nodes": dyn_nodes,
            "dyn_to_cluster": dyn_to_cluster,
        },
    }


def save_dataset(samples: List[Dict[str, Any]], out_npz: Path, meta_extra: Dict[str, Any]) -> None:
    if not samples:
        raise RuntimeError("No samples to save.")

    X_ctx = np.stack([s["ctx"] for s in samples], axis=0).astype(np.float32)
    X_sit = np.stack([s["sit"] for s in samples], axis=0).astype(np.float32)
    X_dyn = np.stack([s["dyn"] for s in samples], axis=0).astype(np.float32)
    X_goal = np.stack([s["goal"] for s in samples], axis=0).astype(np.float32)
    Y_u = np.stack([s["u"] for s in samples], axis=0).astype(np.float32)
    Y_next = np.stack([s["next_local"] for s in samples], axis=0).astype(np.float32)

    traj_ids = np.asarray([s["traj_id"] for s in samples], dtype=np.int32)
    dyn_ids = np.asarray([s["dyn_id"] for s in samples], dtype=np.int32)
    map_idxs = np.asarray([s["map_idx"] for s in samples], dtype=np.int32)

    def stats(X: np.ndarray, axis=0):
        mean = X.mean(axis=axis).astype(np.float32)
        std = X.std(axis=axis).astype(np.float32)
        std = np.maximum(std, 1e-6)
        return mean, std

    ctx_mean, ctx_std = stats(X_ctx.reshape(-1, X_ctx.shape[-1]), axis=0)
    sit_mean, sit_std = stats(X_sit, axis=0)
    dyn_mean, dyn_std = stats(X_dyn, axis=0)
    goal_mean, goal_std = stats(X_goal, axis=0)
    u_mean, u_std = stats(Y_u, axis=0)
    next_mean, next_std = stats(Y_next, axis=0)

    X_ctx_norm = ((X_ctx - ctx_mean[None, None, :]) / ctx_std[None, None, :]).astype(np.float32)
    X_sit_norm = ((X_sit - sit_mean[None, :]) / sit_std[None, :]).astype(np.float32)
    X_dyn_norm = ((X_dyn - dyn_mean[None, :]) / dyn_std[None, :]).astype(np.float32)
    X_goal_norm = ((X_goal - goal_mean[None, :]) / goal_std[None, :]).astype(np.float32)
    Y_u_norm = ((Y_u - u_mean[None, :]) / u_std[None, :]).astype(np.float32)
    Y_next_norm = ((Y_next - next_mean[None, :]) / next_std[None, :]).astype(np.float32)

    meta = {
        "n_samples": int(X_ctx.shape[0]),
        "L_c": int(meta_extra["L_c"]),
        "ctx_dim": int(X_ctx.shape[-1]),
        "sit_dim": int(X_sit.shape[-1]),
        "dyn_dim": int(X_dyn.shape[-1]),
        "goal_dim": int(X_goal.shape[-1]),
        "u_dim": int(Y_u.shape[-1]),
        "next_dim": int(Y_next.shape[-1]),
        "dyn_feature_layout": "A_local_flat(4), B_local_flat(2*u_dim), drift_local(2)",
        **meta_extra,
    }

    expected_dyn = 4 + 2 * meta["u_dim"] + 2
    if meta["dyn_dim"] != expected_dyn:
        raise RuntimeError(f"Bad dyn_dim={meta['dyn_dim']}; expected {expected_dyn}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        ctx=X_ctx_norm,
        sit=X_sit_norm,
        dyn=X_dyn_norm,
        goal=X_goal_norm,
        u=Y_u_norm,
        next_local=Y_next_norm,
        traj_id=traj_ids,
        dyn_id=dyn_ids,
        map_idx=map_idxs,
        meta=np.array(meta, dtype=object),
        norm_ctx_mean=ctx_mean,
        norm_ctx_std=ctx_std,
        norm_sit_mean=sit_mean,
        norm_sit_std=sit_std,
        norm_dyn_mean=dyn_mean,
        norm_dyn_std=dyn_std,
        norm_goal_mean=goal_mean,
        norm_goal_std=goal_std,
        norm_u_mean=u_mean,
        norm_u_std=u_std,
        norm_next_mean=next_mean,
        norm_next_std=next_std,
    )

    print(
        f"[ok] wrote dataset: {out_npz}\n"
        f"Summary: {meta['n_samples']} samples | ctx_dim={meta['ctx_dim']} | "
        f"sit_dim={meta['sit_dim']} | dyn_dim={meta['dyn_dim']} | "
        f"goal_dim={meta['goal_dim']} | u_dim={meta['u_dim']}\n"
        f"Expected dyn_dim = 4 + 2*u_dim + 2 = {expected_dyn}"
    )


# =============================================================================
# Main generation loop
# =============================================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--selection_mode", choices=["trajectories", "windows"], default="trajectories",
                   help="`trajectories`: stop after --target_trajectories successful expert runs. "
                        "`windows`: stop after collecting at least --target_motion_primitives neural windows "
                        "(legacy Neural S1 data-generation behavior).")
    p.add_argument("--target_trajectories", type=int, default=500,
                   help="Number of successful SFCBF trajectories to store in the S1 database.")
    p.add_argument("--target_motion_primitives", type=int, default=0,
                   help="Optional exact number of neural training windows to save. <=0 keeps all windows from the selected trajectories.")
    p.add_argument("--out_npz", type=str, default="db/nn_dataset_maze.npz")
    p.add_argument("--traj_out", type=str, default="db/s1_sfcbf_success_trajs.npz")
    p.add_argument("--db_out", type=str, default="db/S1_database_maze.json")
    p.add_argument("--scenarios_out", type=str, default="db/benchmark_scenarios_maze_s1_db.json")
    p.add_argument("--report_out", type=str, default="output/diversity_report.json")

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_attempts", type=int, default=20000)

    p.add_argument("--L_c", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--bounds", type=float, nargs=4, default=[-10.0, -10.0, 10.0, 10.0])
    p.add_argument("--start", type=float, nargs=2, default=[5.0, 5.0])
    p.add_argument("--goal", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--start_goal_jitter", type=float, default=0.0,
                   help="Optional jitter for start and goal. Default 0 keeps benchmark-compatible fixed start/goal.")

    # Dynamics and map diversity.
    p.add_argument("--dynamics_regimes", type=str, nargs="+",
                   default=["damped", "rotate_cw", "rotate_ccw", "shear", "anisotropic", "mixed"])
    p.add_argument("--B_modes", type=str, nargs="+",
                   default=["identity", "axis_scaled", "rotated_scaled", "random_invertible"])
    p.add_argument("--map_types", type=str, nargs="+",
                   default=[
                       "small_open",
                       "large_sparse",
                       "dense_clutter",
                       "wall_gap",
                       "serial_walls",
                       "maze_branching",
                       "bugtrap",
                   ])
    p.add_argument("--difficulties", type=str, nargs="+", default=["easy", "medium", "hard"])

    # S2 expert settings.
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--s2_steps", type=int, default=800)
    p.add_argument("--u_max", type=float, default=3.0)
    p.add_argument("--s2_margin", type=float, default=0.35)
    p.add_argument("--s2_gamma", type=float, default=2.0)
    p.add_argument("--goal_tol", type=float, default=0.5)
    p.add_argument("--collision_margin", type=float, default=0.05)

    # Situation vector settings.
    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument("--n_steps_nom", type=int, default=200)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)

    p.add_argument("--progress_every", type=int, default=25)
    p.add_argument("--db_clusters", type=int, default=5,
                   help="Number of coarse dynamics clusters in the generated S1 retrieval DB.")

    args = p.parse_args()
    args.out_npz = str(resolve_output_path(args.out_npz))
    args.traj_out = str(resolve_output_path(args.traj_out))
    args.db_out = str(resolve_output_path(args.db_out))
    args.scenarios_out = str(resolve_output_path(args.scenarios_out))
    args.report_out = str(resolve_output_path(args.report_out))

    rng = np.random.default_rng(args.seed)
    bounds = list(map(float, args.bounds))
    base_start = np.asarray(args.start, dtype=float)
    base_goal = np.asarray(args.goal, dtype=float)

    selection_mode = str(args.selection_mode).strip().lower()
    target_trajectories = int(args.target_trajectories)
    target_windows = int(args.target_motion_primitives)
    if selection_mode == "trajectories":
        if target_trajectories <= 0:
            raise ValueError("target_trajectories must be positive when --selection_mode trajectories")
    else:
        if target_windows <= 0:
            raise ValueError("target_motion_primitives must be positive when --selection_mode windows")

    samples: List[Dict[str, Any]] = []
    traj_states: List[np.ndarray] = []
    traj_inputs: List[np.ndarray] = []
    traj_dyn_id: List[int] = []
    traj_map_idx: List[int] = []
    traj_success: List[bool] = []
    traj_runtime_sec: List[float] = []
    traj_collision_free: List[bool] = []
    traj_goal_reached: List[bool] = []

    scenarios: List[Dict[str, Any]] = []
    shared_maps: List[Dict[str, Any]] = []
    dyn_nodes: Dict[str, Dict[str, Any]] = {}

    attempt = 0
    success_count = 0
    fail_count = 0
    t0 = time.perf_counter()

    regime_counter = Counter()
    map_counter = Counter()
    difficulty_counter = Counter()
    group_counter = Counter()

    combos = []
    for reg in args.dynamics_regimes:
        for bm in args.B_modes:
            for mt in args.map_types:
                for diff in args.difficulties:
                    combos.append((reg, bm, mt, diff))
    if not combos:
        raise ValueError("No diversity combos available")

    while attempt < args.max_attempts:
        if selection_mode == "trajectories" and success_count >= target_trajectories:
            break
        if selection_mode == "windows" and len(samples) >= target_windows:
            break

        combo = combos[attempt % len(combos)]
        regime, B_mode, map_type, difficulty = combo
        attempt += 1

        # Optional start/goal jitter. Default zero to match your benchmark.
        start = base_start.copy()
        goal = base_goal.copy()
        if args.start_goal_jitter > 0:
            start += rng.normal(scale=args.start_goal_jitter, size=2)
            goal += rng.normal(scale=args.start_goal_jitter, size=2)

        A = sample_A(rng, regime)
        B = sample_B(rng, B_mode)
        A, B = ensure_A_B(A, B)

        rects = sample_map(rng, map_type, difficulty, bounds, start, goal)

        scenario_id = success_count  # successful scenarios only are stored with compact ids
        dyn_id = success_count
        map_idx = success_count if selection_mode == "windows" else 0

        try:
            out = s2.simulate_sfcbf(
                A=A,
                B=B,
                rects=rects,
                start=tuple(start),
                goal=tuple(goal),
                dt=args.dt,
                n_steps=args.s2_steps,
                u_max=args.u_max,
                margin=args.s2_margin,
                gamma=args.s2_gamma,
                goal_tol=args.goal_tol,
                collision_margin=args.collision_margin,
            )
        except Exception as e:
            fail_count += 1
            continue

        if not bool(out.get("success", False)):
            fail_count += 1
            continue

        states = np.asarray(out["states"], dtype=float)
        inputs = np.asarray(out["inputs"], dtype=float)
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, B.shape[1])
        if len(states) < args.L_c + 1 or len(inputs) < len(states) - 1:
            fail_count += 1
            continue

        group_key = f"{regime}|{B_mode}|{map_type}|{difficulty}"
        new_samples = extract_samples_from_trajectory(
            states=states,
            inputs=inputs,
            A=A,
            B=B,
            rects=rects,
            bounds=bounds,
            start=start,
            goal=goal,
            dyn_id=dyn_id,
            map_idx=map_idx,
            traj_id=success_count,
            L_c=args.L_c,
            stride=args.stride,
            grid_n=args.grid_n,
            dt_nom=args.dt,
            n_steps_nom=args.n_steps_nom,
            u_max_nom=args.u_max,
            buffer_cells=args.buffer_cells,
            stop_tol=args.stop_tol,
            group_key=group_key,
        )
        if not new_samples:
            fail_count += 1
            continue

        samples.extend(new_samples)
        traj_states.append(states.astype(np.float32))
        traj_inputs.append(inputs.astype(np.float32))
        traj_dyn_id.append(dyn_id)
        traj_map_idx.append(map_idx)
        traj_success.append(True)
        traj_runtime_sec.append(float(out.get("runtime_sec", 0.0)))
        traj_collision_free.append(bool(out.get("collision_free", True)))
        traj_goal_reached.append(bool(out.get("goal_reached", True)))

        situation_vec = compute_situation_vector(
            A=A,
            B=B,
            rects=rects,
            bounds=bounds,
            start=start.tolist(),
            goal=goal.tolist(),
            grid_n=args.grid_n,
            dt_nom=args.dt,
            n_steps_nom=args.n_steps_nom,
            u_max_nom=args.u_max,
            buffer_cells=args.buffer_cells,
            stop_tol=args.stop_tol,
        )
        M = system_to_M(A, B)

        shared_maps.append({
            "map_idx": map_idx,
            "dyn_id": dyn_id,
            "rectangles": [list(map(float, r)) for r in rects],
            "bounds": list(map(float, bounds)),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "map_type": map_type,
            "difficulty": difficulty,
        })
        dyn_nodes[str(dyn_id)] = {
            "dyn_id": dyn_id,
            "A": A.tolist(),
            "B": B.tolist(),
            "C": np.eye(2, dtype=float).tolist(),
            "M": M.tolist(),
            "regime": regime,
            "B_mode": B_mode,
            "env_types": {
                "maze": {
                    "map_count": 1,
                    "situation_vecs": [situation_vec.astype(np.uint8).tolist()],
                }
            },
        }
        scenarios.append({
            "scenario_id": scenario_id,
            "dyn_id": dyn_id,
            "map_idx": map_idx,
            "A_query": A.tolist(),
            "B_query": B.tolist(),
            "rectangles": [list(map(float, r)) for r in rects],
            "bounds": list(map(float, bounds)),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "u_max": float(args.u_max),
            "goal_tol": float(args.goal_tol),
            "map_type": map_type,
            "difficulty": difficulty,
            "regime": regime,
            "B_mode": B_mode,
        })

        success_count += 1
        regime_counter[regime] += 1
        map_counter[map_type] += 1
        difficulty_counter[difficulty] += 1
        group_counter[group_key] += 1

        progress_goal_reached = (
            success_count >= target_trajectories
            if selection_mode == "trajectories"
            else len(samples) >= target_windows
        )
        if success_count % args.progress_every == 0 or progress_goal_reached:
            elapsed = time.perf_counter() - t0
            target_label = (
                f"target_traj={target_trajectories}"
                if selection_mode == "trajectories"
                else f"target_windows={target_windows}"
            )
            print(
                f"[progress] attempts={attempt} | successful_traj={success_count} | "
                f"failed={fail_count} | samples={len(samples)} | "
                f"{target_label} | "
                f"last_group={group_key} | elapsed={elapsed:.1f}s"
            )

    if selection_mode == "trajectories":
        if success_count < target_trajectories:
            raise RuntimeError(
                f"Only generated {success_count} successful trajectories after {attempt} attempts. "
                f"Increase --max_attempts or reduce --target_trajectories."
            )
    else:
        if len(samples) < target_windows:
            raise RuntimeError(
                f"Only generated {len(samples)} motion-primitives/windows from {success_count} successful trajectories "
                f"after {attempt} attempts. Increase --max_attempts or reduce --target_motion_primitives."
            )

    # The neural dataset is derived only from the trajectories saved above.
    # Optionally downsample windows for faster training, but never introduce
    # windows from trajectories that are not in the S1 database.
    if target_windows > 0:
        selected_samples = balanced_downsample(samples, target_windows, rng)
    else:
        selected_samples = list(samples)

    meta_extra = {
        "generator": "make_diverse_training_data_maze.py",
        "L_c": int(args.L_c),
        "stride": int(args.stride),
        "grid_n": int(args.grid_n),
        "dt_nom": float(args.dt),
        "n_steps_nom": int(args.n_steps_nom),
        "u_max_nom": float(args.u_max),
        "buffer_cells": int(args.buffer_cells),
        "stop_tol": float(args.stop_tol),
        "B_shape": "2x2",
        "selection_mode": selection_mode,
        "target_trajectories": int(target_trajectories),
        "target_motion_primitives": int(target_windows),
        "successful_trajectories_generated": int(success_count),
        "attempts": int(attempt),
    }

    out_npz = Path(args.out_npz)
    save_dataset(selected_samples, out_npz, meta_extra)

    # Save raw successful trajectories and compatible DB/scenarios.
    traj_out = Path(args.traj_out)
    traj_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(traj_out),
        states=np.asarray(traj_states, dtype=object),
        inputs=np.asarray(traj_inputs, dtype=object),
        dyn_id=np.asarray(traj_dyn_id, dtype=np.int32),
        map_idx=np.asarray(traj_map_idx, dtype=np.int32),
        runtime_sec=np.asarray(traj_runtime_sec, dtype=np.float64),
        success=np.asarray(traj_success, dtype=bool),
        collision_free=np.asarray(traj_collision_free, dtype=bool),
        goal_reached=np.asarray(traj_goal_reached, dtype=bool),
    )
    print(f"[ok] wrote trajectories: {traj_out}")

    db = build_s1_retrieval_db(
        dyn_nodes=dyn_nodes,
        shared_maps=shared_maps,
        bounds=bounds,
        base_start=base_start,
        base_goal=base_goal,
        start_goal_jitter=float(args.start_goal_jitter),
        n_clusters=int(args.db_clusters),
        success_count=int(success_count),
    )
    db_out = Path(args.db_out)
    db_out.parent.mkdir(parents=True, exist_ok=True)
    db_out.write_text(json.dumps(db, indent=2))
    print(f"[ok] wrote DB: {db_out}")

    scenarios_out = Path(args.scenarios_out)
    scenarios_out.parent.mkdir(parents=True, exist_ok=True)
    scenarios_out.write_text(json.dumps(scenarios, indent=2))
    print(f"[ok] wrote scenarios: {scenarios_out}")

    selected_group_counter = Counter(str(s["group"]) for s in selected_samples)
    report = {
        "selection_mode": selection_mode,
        "target_trajectories": int(target_trajectories),
        "target_motion_primitives": int(target_windows),
        "saved_motion_primitives": int(len(selected_samples)),
        "raw_motion_primitives_before_downsample": int(len(samples)),
        "successful_trajectories": int(success_count),
        "attempts": int(attempt),
        "failed_attempts": int(fail_count),
        "regime_counter_successful_traj": dict(regime_counter),
        "map_counter_successful_traj": dict(map_counter),
        "difficulty_counter_successful_traj": dict(difficulty_counter),
        "group_counter_successful_traj": dict(group_counter),
        "group_counter_selected_samples": dict(selected_group_counter),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))
    print(f"[ok] wrote report: {report_out}")


if __name__ == "__main__":
    main()

