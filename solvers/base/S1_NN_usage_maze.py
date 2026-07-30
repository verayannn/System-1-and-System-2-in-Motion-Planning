"""
S1_NN_usage_maze.py - neural System-1 rollout utilities for maze scenarios.

This module is dimension-consistent with the maze data generator and trainer:
    A is 2x2
    B is 2xm. In the current maze experiments, B is usually 2x2, so u_dim=2.
    dyn_feat = [A_local.flatten(), B_local.flatten(), drift_local]

For B 2x1, dyn_dim = 8.

The rollout always follows true dynamics:
    x_{k+1} = x_k + dt(Ax_k + Bu_k)

Obstacle handling:
    A one-step safety/progress shield chooses among S1, goal, and blended controls.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

Rect = Tuple[float, float, float, float]


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
SOLVER_DIR = THIS_DIR.parent
INSTANCE_DIR = SOLVER_DIR.parent
LEGACY_CODES_ROOT = Path(
    os.environ.get("SOFAI_LEGACY_CODES_ROOT", INSTANCE_DIR)
).expanduser()


def _strip_legacy_maze_prefix(path: Path) -> Path:
    if len(path.parts) >= 3 and path.parts[0] == "maze" and path.parts[1] == "n_s1":
        return Path(*path.parts[2:])
    if len(path.parts) >= 2 and path.parts[0] == "maze" and path.parts[1] == "bck":
        return Path("input") / Path(*path.parts[2:])
    return path


def resolve_existing_path(path_like: str, *, required: bool = False) -> Path:
    if not path_like:
        return Path()

    path = Path(path_like).expanduser()
    if path.is_absolute():
        if path.exists() or not required:
            return path
        raise FileNotFoundError(f"Required path does not exist: {path}")

    local_suffix = _strip_legacy_maze_prefix(path)
    candidates = [
        Path.cwd() / path,
        INSTANCE_DIR / path,
        INSTANCE_DIR / local_suffix,
        SOLVER_DIR / local_suffix,
        LEGACY_CODES_ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Required path does not exist: {path_like}\nSearched:\n  {searched}")
    return INSTANCE_DIR / path


def resolve_output_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) >= 3 and path.parts[0] == "maze" and path.parts[1] == "n_s1":
        return INSTANCE_DIR / "output" / Path(*path.parts[2:])
    return INSTANCE_DIR / _strip_legacy_maze_prefix(path)


# ============================================================
# Shape helpers
# ============================================================


def ensure_A_B(A: Any, B: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly convert A, B to correct matrix shapes.

    Expected:
        A: 2x2
        B: 2xm

    In current maze experiments:
        B is usually 2x2, so u_dim = 2.

    This function also supports:
        B = [b1, b2]          -> 2x1
        B = [b11,b12,b21,b22] -> 2x2
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # A should be 2x2
    if A.shape != (2, 2):
        if A.size == 4:
            A = A.reshape(2, 2)
        else:
            raise ValueError(f"Expected A shape (2,2), got {A.shape}")

    # B can be flat or matrix
    if B.ndim == 1:
        if B.size == 2:
            # scalar-input case: B is 2x1
            B = B.reshape(2, 1)
        elif B.size == 4:
            # two-input case: B is 2x2
            B = B.reshape(2, 2)
        else:
            raise ValueError(
                f"Flat B must have length 2 or 4 for a 2xm system, got shape {B.shape}"
            )

    # If B was accidentally stored as 1x2, transpose to 2x1
    if B.ndim == 2 and B.shape[0] == 1 and B.shape[1] == 2:
        B = B.reshape(2, 1)

    if B.ndim != 2 or B.shape[0] != 2:
        raise ValueError(f"Expected B shape (2,m), got {B.shape}")

    return A.astype(float), B.astype(float)

# ============================================================
# Situation logic
# ============================================================

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
    A, B = ensure_A_B(A, B)

    xmin, ymin, xmax, ymax = map(float, bounds)
    x = np.asarray(start, dtype=float).reshape(2)
    g = np.asarray(goal, dtype=float).reshape(2)

    visited = np.zeros((grid_n, grid_n), dtype=np.uint8)

    for _ in range(int(n_steps_nom)):
        if float(np.linalg.norm(x - g)) <= float(stop_tol):
            break

        rhs = -(x - g) - A @ x
        try:
            u = np.linalg.lstsq(B, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            u = np.zeros(B.shape[1], dtype=float)

        u = np.clip(u, -u_max_nom, u_max_nom)
        x = x + float(dt_nom) * (A @ x + B @ u)

        if not (xmin <= x[0] <= xmax and ymin <= x[1] <= ymax):
            continue

        i = int((x[1] - ymin) / (ymax - ymin) * grid_n)
        j = int((x[0] - xmin) / (xmax - xmin) * grid_n)
        i = np.clip(i, 0, grid_n - 1)
        j = np.clip(j, 0, grid_n - 1)
        visited[i, j] = 1

    corridor = _dilate4(visited, iters=int(buffer_cells))

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


# ============================================================
# Model
# ============================================================

class NeuralSystem1ControlPolicyCNN(nn.Module):
    def __init__(
        self,
        ctx_shape,
        sit_dim,
        dyn_dim,
        goal_dim,
        u_dim,
        hidden=256,
        cnn_channels=128,
        dropout=0.05,
    ):
        super().__init__()
        self.ctx_shape = tuple(ctx_shape)
        self.sit_dim = int(sit_dim)
        self.dyn_dim = int(dyn_dim)
        self.goal_dim = int(goal_dim)
        self.u_dim = int(u_dim)

        _, ctx_dim = self.ctx_shape

        self.ctx_encoder = nn.Sequential(
            nn.Conv1d(ctx_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.sit_encoder = nn.Sequential(
            nn.Linear(self.sit_dim, hidden // 2),
            nn.ReLU(),
        )

        self.dyn_encoder = nn.Sequential(
            nn.Linear(self.dyn_dim, hidden // 2),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, hidden // 4),
            nn.ReLU(),
        )

        fused_dim = cnn_channels + hidden // 2 + hidden // 2 + hidden // 4
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.u_dim),
        )

    def forward(self, ctx, sit, dyn, goal):
        x_ctx = ctx.transpose(1, 2)
        f_ctx = self.ctx_encoder(x_ctx).squeeze(-1)
        f_sit = self.sit_encoder(sit)
        f_dyn = self.dyn_encoder(dyn)
        f_goal = self.goal_encoder(goal)
        z = torch.cat([f_ctx, f_sit, f_dyn, f_goal], dim=1)
        return self.head(z)


# ============================================================
# Local frame / dynamics features
# ============================================================

def compute_heading_from_context(ctx_pos: np.ndarray, goal_global: np.ndarray) -> float:
    if ctx_pos.shape[0] >= 2:
        v = ctx_pos[-1] - ctx_pos[-2]
        if np.linalg.norm(v) > 1e-6:
            return float(np.arctan2(v[1], v[0]))

    vg = goal_global[:2] - ctx_pos[-1]
    if np.linalg.norm(vg) > 1e-8:
        return float(np.arctan2(vg[1], vg[0]))
    return 0.0


def rotation_from_heading(heading: float) -> np.ndarray:
    c, s = np.cos(-heading), np.sin(-heading)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_to_local(ctx_global: np.ndarray, goal_global: np.ndarray):
    origin = ctx_global[-1].astype(float)
    heading = compute_heading_from_context(ctx_global, goal_global)
    R = rotation_from_heading(heading)
    ctx_local = (ctx_global - origin) @ R.T
    goal_local = (goal_global[:2] - origin) @ R.T
    return ctx_local.astype(np.float32), goal_local.astype(np.float32), origin.astype(np.float32), float(heading)


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


# ============================================================
# Normalization helpers
# ============================================================

def normalize_ctx(ctx: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return (ctx - norm["ctx_mean"][None, None, :]) / norm["ctx_std"][None, None, :]


def normalize_sit(sit: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return (sit - norm["sit_mean"][None, :]) / norm["sit_std"][None, :]


def normalize_dyn(dyn: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    if dyn.shape[-1] != norm["dyn_mean"].shape[-1]:
        raise ValueError(
            f"dyn dimension mismatch in S1 usage: produced {dyn.shape[-1]}, "
            f"model expects {norm['dyn_mean'].shape[-1]}. "
            "Regenerate dataset/retrain, or update local_dynamics_features consistently."
        )
    return (dyn - norm["dyn_mean"][None, :]) / norm["dyn_std"][None, :]


def normalize_goal(goal: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return (goal - norm["goal_mean"][None, :]) / norm["goal_std"][None, :]


def denormalize_u(u: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return u * norm["u_std"][None, :] + norm["u_mean"][None, :]


# ============================================================
# Geometry / dynamics / safety
# ============================================================

def point_in_any_rectangle(p: np.ndarray, rects: List[Rect], margin: float = 0.0) -> bool:
    x, y = float(p[0]), float(p[1])
    for xmin, ymin, xmax, ymax in rects:
        if (xmin - margin) <= x <= (xmax + margin) and (ymin - margin) <= y <= (ymax + margin):
            return True
    return False


def segment_collision_free(p0: np.ndarray, p1: np.ndarray, rects: List[Rect], margin: float = 0.0, n_sub: int = 12) -> bool:
    for a in np.linspace(0.0, 1.0, n_sub):
        p = (1.0 - a) * p0 + a * p1
        if point_in_any_rectangle(p, rects, margin=margin):
            return False
    return True


def collision_free_rectangles(states: np.ndarray, rects: List[Rect], margin: float = 0.0) -> bool:
    states = np.asarray(states, dtype=float)
    if states.size == 0:
        return True
    for p in states:
        if point_in_any_rectangle(p, rects, margin=margin):
            return False
    return True


def estimate_nominal_goal_control(x: np.ndarray, goal: np.ndarray, A: np.ndarray, B: np.ndarray, u_max: float) -> np.ndarray:
    A, B = ensure_A_B(A, B)
    rhs = -(x - goal[:2]) - A @ x
    try:
        u = np.linalg.lstsq(B, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        u = np.zeros(B.shape[1], dtype=float)
    return np.clip(u, -u_max, u_max).astype(np.float32)


def build_initial_history_ending_at_start(
    A: np.ndarray,
    B: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    L_c: int,
    dt_nom: float,
    u_max_nom: float,
) -> np.ndarray:
    A, B = ensure_A_B(A, B)
    hist = np.zeros((L_c, 2), dtype=np.float32)
    x0 = np.asarray(start[:2], dtype=float)
    g = np.asarray(goal[:2], dtype=float)

    u = estimate_nominal_goal_control(x0, g, A, B, u_max_nom)
    d = dt_nom * (A @ x0 + B @ u)

    if np.linalg.norm(d) < 1e-8:
        d = g - x0
    if np.linalg.norm(d) < 1e-8:
        d = np.array([1.0, 0.0], dtype=float)

    step = d / max(np.linalg.norm(d), 1e-8) * min(np.linalg.norm(d), 0.25)
    for k in range(L_c):
        lag = (L_c - 1 - k)
        hist[k, :] = (x0 - lag * step).astype(np.float32)
    return hist


def propagate_dynamics(x: np.ndarray, A: np.ndarray, B: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    A, B = ensure_A_B(A, B)
    u = np.asarray(u, dtype=float).reshape(B.shape[1])
    return (x + float(dt) * (A @ x + B @ u)).astype(np.float32)


def choose_safe_control(
    x_curr: np.ndarray,
    u_pred: np.ndarray,
    u_goal: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    dt: float,
    rects: List[Rect],
    goal: np.ndarray,
    u_max: float,
    collision_margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    A, B = ensure_A_B(A, B)
    u_dim = B.shape[1]
    u_pred = np.asarray(u_pred, dtype=np.float32).reshape(u_dim)
    u_goal = np.asarray(u_goal, dtype=np.float32).reshape(u_dim)

    candidates = [u_pred]
    for a in [0.8, 0.6, 0.4, 0.2]:
        candidates.append(a * u_pred + (1.0 - a) * u_goal)
    candidates.append(u_goal)
    for a in [0.75, 0.5, 0.25, 0.1]:
        candidates.append(a * u_pred)
    candidates.append(np.zeros_like(u_pred))

    best_u = None
    best_x = None
    best_score = float("inf")
    dist_curr = float(np.linalg.norm(x_curr - goal[:2]))

    for u in candidates:
        u = np.clip(u, -u_max, u_max).astype(np.float32).reshape(u_dim)
        x_next = propagate_dynamics(x_curr, A, B, u, dt)

        if not segment_collision_free(x_curr, x_next, rects, margin=collision_margin, n_sub=12):
            continue

        dist_next = float(np.linalg.norm(x_next - goal[:2]))
        progress_penalty = 0.0 if dist_next < dist_curr else 2.0
        deviation_penalty = 0.01 * float(np.linalg.norm(u - u_pred))
        score = dist_next + progress_penalty + deviation_penalty

        if score < best_score:
            best_score = score
            best_u = u
            best_x = x_next

    if best_u is None:
        best_u = np.zeros(u_dim, dtype=np.float32)
        best_x = x_curr.copy().astype(np.float32)

    return best_u, best_x


# ============================================================
# Rollout
# ============================================================

def rollout_neural_s1(
    model: nn.Module,
    ctx_global_init: np.ndarray,
    sit_vec: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    goal: np.ndarray,
    norm: Dict[str, np.ndarray],
    device: torch.device,
    total_steps: int,
    dt_nom: float,
    u_max_nom: float,
    collision_margin: float,
    goal_tol: float,
    rects: List[Rect],
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    A, B = ensure_A_B(A, B)
    ctx = ctx_global_init.copy().astype(np.float32)
    traj_out = []
    controls_out = []

    for k in range(total_steps):
        x_curr = ctx[-1].astype(np.float32)

        if np.linalg.norm(x_curr - goal[:2]) <= goal_tol:
            break

        ctx_local, goal_local, origin, heading = transform_to_local(ctx, goal[:2])
        dyn_feat = local_dynamics_features(A, B, heading, origin)

        ctx_in = normalize_ctx(ctx_local[None, :, :], norm).astype(np.float32)
        sit_in = normalize_sit(sit_vec[None, :], norm).astype(np.float32)
        dyn_in = normalize_dyn(dyn_feat[None, :], norm).astype(np.float32)
        goal_in = normalize_goal(goal_local[None, :], norm).astype(np.float32)

        with torch.no_grad():
            t_ctx = torch.from_numpy(ctx_in).float().to(device)
            t_sit = torch.from_numpy(sit_in).float().to(device)
            t_dyn = torch.from_numpy(dyn_in).float().to(device)
            t_goal = torch.from_numpy(goal_in).float().to(device)
            u_norm = model(t_ctx, t_sit, t_dyn, t_goal).cpu().numpy()

        u_pred = denormalize_u(u_norm, norm).squeeze(0).astype(np.float32)
        u_pred = np.clip(u_pred, -u_max_nom, u_max_nom).reshape(B.shape[1])

        u_goal = estimate_nominal_goal_control(x_curr, goal, A, B, u_max_nom)

        u_safe, x_next = choose_safe_control(
            x_curr=x_curr,
            u_pred=u_pred,
            u_goal=u_goal,
            A=A,
            B=B,
            dt=dt_nom,
            rects=rects,
            goal=goal,
            u_max=u_max_nom,
            collision_margin=collision_margin,
        )

        if debug and k == 0:
            print("DEBUG x_curr =", x_curr)
            print("DEBUG B shape =", B.shape)
            print("DEBUG dyn_feat shape =", dyn_feat.shape)
            print("DEBUG expected dyn shape =", norm["dyn_mean"].shape)
            print("DEBUG heading =", heading)
            print("DEBUG goal_local =", goal_local)
            print("DEBUG u_pred =", u_pred)
            print("DEBUG u_goal =", u_goal)
            print("DEBUG u_safe =", u_safe)
            print("DEBUG x_next =", x_next)

        traj_out.append(x_next.copy())
        controls_out.append(u_safe.copy())
        ctx = np.concatenate([ctx, x_next[None, :]], axis=0)[-ctx.shape[0]:]

    if len(traj_out) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, B.shape[1]), dtype=np.float32)

    return np.stack(traj_out, axis=0).astype(np.float32), np.stack(controls_out, axis=0).astype(np.float32)


# ============================================================
# Benchmark CLI
# ============================================================

def run_nn_s1_benchmark(args):
    args.model = str(resolve_existing_path(args.model, required=True))
    args.scenarios = str(resolve_existing_path(args.scenarios, required=True))
    args.out = str(resolve_output_path(args.out))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    meta = ckpt["meta"]
    model = NeuralSystem1ControlPolicyCNN(
        ctx_shape=meta["ctx_shape"],
        sit_dim=meta["sit_dim"],
        dyn_dim=meta["dyn_dim"],
        goal_dim=meta["goal_dim"],
        u_dim=meta["u_dim"],
        hidden=meta.get("hidden", 256),
        cnn_channels=meta.get("cnn_channels", 128),
        dropout=meta.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    norm_raw = ckpt["norm"]
    norm = {
        "ctx_mean": np.asarray(norm_raw["ctx_mean"], dtype=np.float32),
        "ctx_std": np.asarray(norm_raw["ctx_std"], dtype=np.float32),
        "sit_mean": np.asarray(norm_raw["sit_mean"], dtype=np.float32),
        "sit_std": np.asarray(norm_raw["sit_std"], dtype=np.float32),
        "dyn_mean": np.asarray(norm_raw["dyn_mean"], dtype=np.float32),
        "dyn_std": np.asarray(norm_raw["dyn_std"], dtype=np.float32),
        "goal_mean": np.asarray(norm_raw["goal_mean"], dtype=np.float32),
        "goal_std": np.asarray(norm_raw["goal_std"], dtype=np.float32),
        "u_mean": np.asarray(norm_raw["u_mean"], dtype=np.float32),
        "u_std": np.asarray(norm_raw["u_std"], dtype=np.float32),
    }

    L_c, ctx_dim = meta["ctx_shape"]
    if ctx_dim != 2:
        raise ValueError(f"Expected 2D context, got {meta['ctx_shape']}")

    scenarios = json.loads(Path(args.scenarios).read_text())
    results = []

    print(f"Running Neural S1 on {len(scenarios)} scenarios...")

    for sc in scenarios:
        t0 = time.perf_counter()
        A, B = ensure_A_B(sc["A_query"], sc["B_query"])
        rects = [tuple(map(float, r)) for r in sc["rectangles"]]
        start = np.asarray(sc["start"], dtype=float)
        goal = np.asarray(sc["goal"], dtype=float)
        bounds = sc.get("bounds", [-10, -10, 10, 10])

        sit_vec = compute_situation_vector(
            A=A, B=B, rects=rects, bounds=bounds,
            start=start.tolist(), goal=goal.tolist(),
            grid_n=args.grid_n, dt_nom=args.dt_nom,
            n_steps_nom=args.n_steps_nom, u_max_nom=args.u_max_nom,
            buffer_cells=args.buffer_cells, stop_tol=args.stop_tol,
        )

        ctx_global_init = build_initial_history_ending_at_start(
            A=A, B=B, start=start, goal=goal,
            L_c=L_c, dt_nom=args.dt_nom, u_max_nom=args.u_max_nom,
        )

        pred_g, controls = rollout_neural_s1(
            model=model,
            ctx_global_init=ctx_global_init,
            sit_vec=sit_vec,
            A=A,
            B=B,
            goal=goal,
            norm=norm,
            device=device,
            total_steps=args.total_steps,
            dt_nom=args.dt_nom,
            u_max_nom=args.u_max_nom,
            collision_margin=args.collision_margin,
            goal_tol=args.goal_tol,
            rects=rects,
            debug=args.debug,
        )

        runtime = time.perf_counter() - t0
        cf = collision_free_rectangles(pred_g, rects, margin=args.collision_margin)
        dist_to_goal = float(np.linalg.norm(pred_g[-1] - goal[:2])) if len(pred_g) > 0 else float("inf")
        gr = dist_to_goal <= args.goal_tol
        success = bool(cf and gr)

        results.append({
            "scenario_id": int(sc["scenario_id"]),
            "success": success,
            "runtime_sec": float(runtime),
            "collision_free": bool(cf),
            "goal_reached": bool(gr),
            "final_dist": dist_to_goal,
            "predicted_trajectory": pred_g.tolist(),
            "controls": controls.tolist(),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Benchmark complete. Results saved to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--scenarios", type=str, required=True)
    p.add_argument("--out", type=str, default="output/NN_S1_results.json")

    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument("--dt_nom", type=float, default=0.05)
    p.add_argument("--n_steps_nom", type=int, default=200)
    p.add_argument("--u_max_nom", type=float, default=3.0)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)

    p.add_argument("--collision_margin", type=float, default=0.05)
    p.add_argument("--goal_tol", type=float, default=0.6)
    p.add_argument("--total_steps", type=int, default=120)
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    run_nn_s1_benchmark(args)
