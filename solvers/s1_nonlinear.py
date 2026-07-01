from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from solvers._s2_common import collision_free_rectangles, goal_reached

Rect = Tuple[float, float, float, float]


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


def _get(sc: Any, key: str, default: Any = None) -> Any:
    if isinstance(sc, dict):
        return sc.get(key, default)
    return getattr(sc, key, default)


def _as_array(x: Any, *, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected array with ndim={ndim}, got shape {arr.shape}")
    return arr


def scenario_rects(scenario: Any) -> list[Rect]:
    return [tuple(map(float, r)) for r in _get(scenario, "rectangles", _get(scenario, "rects", []))]


def scenario_start(scenario: Any) -> np.ndarray:
    return np.asarray(_get(scenario, "start", (5.0, 5.0)), dtype=float).reshape(-1)


def scenario_goal(scenario: Any) -> np.ndarray:
    return np.asarray(_get(scenario, "goal", (0.0, 0.0)), dtype=float).reshape(-1)


def scenario_bounds(scenario: Any) -> Tuple[float, float, float, float]:
    return tuple(map(float, _get(scenario, "bounds", (-10.0, -10.0, 10.0, 10.0))))  # type: ignore[return-value]


def scenario_u_max(scenario: Any, default: float = 3.0) -> float:
    return float(_get(scenario, "u_max", default))


def scenario_goal_tol(scenario: Any, default: float = 0.6) -> float:
    return float(_get(scenario, "goal_tol", default))


def rotation_from_heading(heading: float) -> np.ndarray:
    c, s = np.cos(-heading), np.sin(-heading)
    return np.array([[c, -s], [s, c]], dtype=float)


def compute_heading_from_context(ctx_pos: np.ndarray, goal_global: np.ndarray) -> float:
    if ctx_pos.shape[0] >= 2:
        v = ctx_pos[-1] - ctx_pos[-2]
        if np.linalg.norm(v) > 1e-6:
            return float(np.arctan2(v[1], v[0]))
    vg = goal_global[:2] - ctx_pos[-1]
    if np.linalg.norm(vg) > 1e-8:
        return float(np.arctan2(vg[1], vg[0]))
    return 0.0


def transform_to_local(ctx_global: np.ndarray, goal_global: np.ndarray):
    origin = ctx_global[-1].astype(float)
    heading = compute_heading_from_context(ctx_global, goal_global)
    R = rotation_from_heading(heading)
    ctx_local = (ctx_global - origin) @ R.T
    goal_local = (goal_global[:2] - origin) @ R.T
    return ctx_local.astype(np.float32), goal_local.astype(np.float32), origin.astype(np.float32), float(heading)


def _regime_onehot(regime: str) -> np.ndarray:
    order = ["sink", "rotate_cw", "rotate_ccw", "weak_shear"]
    out = np.zeros(len(order), dtype=np.float32)
    if regime in order:
        out[order.index(regime)] = 1.0
    return out


def nonlinear_dynamics_payload(scenario: Any) -> Dict[str, Any]:
    payload = _get(scenario, "nonlinear_dynamics", None)
    if isinstance(payload, dict):
        return payload
    return {}


def nonlinear_drift_global(scenario: Any, x: Sequence[float]) -> np.ndarray:
    payload = nonlinear_dynamics_payload(scenario)
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
                [
                    -a * np.tanh(x1) + shear * np.sin(x2),
                    -b * np.tanh(x2) - 0.5 * shear * np.sin(x1),
                ],
                dtype=np.float32,
            )
        if regime == "rotate_cw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return np.array(
                [
                    -damp * np.tanh(x1) + omega * np.sin(x2),
                    -damp * np.tanh(x2) - omega * np.sin(x1),
                ],
                dtype=np.float32,
            )
        if regime == "rotate_ccw":
            damp = float(params.get("damp", 0.5))
            omega = float(params.get("omega", 1.0))
            return np.array(
                [
                    -damp * np.tanh(x1) - omega * np.sin(x2),
                    -damp * np.tanh(x2) + omega * np.sin(x1),
                ],
                dtype=np.float32,
            )
        a = float(params.get("a", 0.5))
        b = float(params.get("b", 0.5))
        shear = float(params.get("shear", 0.0))
        return np.array(
            [
                -a * np.tanh(x1) + 2.0 * shear * x2,
                -b * np.tanh(x2) + 0.25 * shear * x1,
            ],
            dtype=np.float32,
        )

    A = _get(scenario, "A_query", _get(scenario, "A", None))
    if A is None:
        return np.zeros(2, dtype=np.float32)
    A = np.asarray(A, dtype=float).reshape(2, 2)
    return (A @ np.asarray(x, dtype=float).reshape(2)).astype(np.float32)


def nonlinear_dynamics_features(scenario: Any, x_global: Sequence[float], heading: float) -> np.ndarray:
    drift = nonlinear_drift_global(scenario, x_global)
    R = rotation_from_heading(heading)
    drift_local = R @ drift

    payload = nonlinear_dynamics_payload(scenario)
    params = dict(payload.get("parameters", {}))
    regime = str(payload.get("regime", "")).strip()
    model = str(payload.get("model", "")).strip()

    param_vec = np.array(
        [
            float(params.get("a", 0.0)),
            float(params.get("b", 0.0)),
            float(params.get("shear", 0.0)),
            float(params.get("damp", 0.0)),
            float(params.get("omega", 0.0)),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            drift_local.astype(np.float32),
            _regime_onehot(regime),
            param_vec,
            np.array([1.0 if payload.get("control_map", "identity") == "identity" else 0.0], dtype=np.float32),
            np.array([1.0 if model == "control_affine_tanh_trig_2d" else 0.0], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def local_dynamics_feature_dim() -> int:
    return 13


def nonlinear_local_control_to_global(u_local: np.ndarray, heading: float) -> np.ndarray:
    R = rotation_from_heading(heading)
    return (R.T @ np.asarray(u_local, dtype=float).reshape(2)).astype(np.float32)


def estimate_nominal_goal_control_local(
    scenario: Any,
    x_global: Sequence[float],
    goal_global: Sequence[float],
    heading: float,
    u_max: float,
) -> np.ndarray:
    x = np.asarray(x_global, dtype=float).reshape(2)
    g = np.asarray(goal_global, dtype=float).reshape(2)
    R = rotation_from_heading(heading)
    goal_local = (g - x) @ R.T
    drift_local = R @ nonlinear_drift_global(scenario, x)
    u_local = 1.5 * goal_local - drift_local
    return np.clip(u_local, -float(u_max), float(u_max)).astype(np.float32)


def propagate_global_state(
    scenario: Any,
    x_global: Sequence[float],
    u_local: Sequence[float],
    heading: float,
    dt: float,
) -> np.ndarray:
    x = np.asarray(x_global, dtype=float).reshape(2)
    u_global = nonlinear_local_control_to_global(np.asarray(u_local, dtype=float).reshape(2), heading)
    return (x + float(dt) * (nonlinear_drift_global(scenario, x) + u_global)).astype(np.float32)


def choose_safe_control(
    *,
    scenario: Any,
    x_curr: np.ndarray,
    u_pred_local: np.ndarray,
    u_goal_local: np.ndarray,
    heading: float,
    dt: float,
    rects: List[Rect],
    goal: np.ndarray,
    u_max: float,
    collision_margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    u_pred_local = np.asarray(u_pred_local, dtype=np.float32).reshape(2)
    u_goal_local = np.asarray(u_goal_local, dtype=np.float32).reshape(2)

    candidates = [u_pred_local]
    for a in [0.8, 0.6, 0.4, 0.2]:
        candidates.append(a * u_pred_local + (1.0 - a) * u_goal_local)
    candidates.append(u_goal_local)
    for a in [0.75, 0.5, 0.25, 0.1]:
        candidates.append(a * u_pred_local)
    candidates.append(np.zeros_like(u_pred_local))

    best_u = None
    best_x = None
    best_score = float("inf")
    dist_curr = float(np.linalg.norm(x_curr - goal[:2]))

    for u in candidates:
        u = np.clip(u, -float(u_max), float(u_max)).astype(np.float32)
        x_next = propagate_global_state(scenario, x_curr, u, heading, dt)
        if not np.isfinite(x_next).all():
            continue
        if not collision_free_rectangles(np.vstack([x_curr[:2], x_next[:2]]), rects, margin=float(collision_margin)):
            continue
        dist_next = float(np.linalg.norm(x_next - goal[:2]))
        score = dist_next + (0.0 if dist_next < dist_curr else 2.0) + 0.01 * float(np.linalg.norm(u - u_pred_local))
        if score < best_score:
            best_score = score
            best_u = u
            best_x = x_next

    if best_u is None:
        best_u = np.zeros(2, dtype=np.float32)
        best_x = x_curr.copy().astype(np.float32)
    return best_u, best_x


def _dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    for _ in range(int(iters)):
        m = np.maximum.reduce([m, np.roll(m, 1, 0), np.roll(m, -1, 0), np.roll(m, 1, 1), np.roll(m, -1, 1)])
    return m


def compute_situation_vector(
    scenario: Any,
    grid_n: int,
    dt_nom: float,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
) -> np.ndarray:
    rects = scenario_rects(scenario)
    xmin, ymin, xmax, ymax = scenario_bounds(scenario)
    start = scenario_start(scenario)
    goal = scenario_goal(scenario)

    x = np.asarray(start, dtype=float).reshape(2)
    visited = np.zeros((grid_n, grid_n), dtype=np.uint8)
    ctx = np.repeat(x[None, :], 2, axis=0)

    for _ in range(int(n_steps_nom)):
        if float(np.linalg.norm(x - goal)) <= float(stop_tol):
            break
        heading = compute_heading_from_context(ctx, goal)
        u_local = estimate_nominal_goal_control_local(scenario, x, goal, heading, u_max_nom)
        x = propagate_global_state(scenario, x, u_local, heading, dt_nom)
        ctx = np.vstack([ctx[-1:], x[None, :]])
        if not (xmin <= x[0] <= xmax and ymin <= x[1] <= ymax):
            continue
        i = int((x[1] - ymin) / max(ymax - ymin, 1e-8) * grid_n)
        j = int((x[0] - xmin) / max(xmax - xmin, 1e-8) * grid_n)
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
        obs[i1 : i2 + 1, j1 : j2 + 1] = 1

    return np.maximum(visited, obs & corridor).reshape(-1).astype(np.float32)


def build_initial_history_ending_at_start(
    scenario: Any,
    L_c: int,
    dt_nom: float,
    u_max_nom: float,
) -> np.ndarray:
    start = scenario_start(scenario)
    goal = scenario_goal(scenario)
    hist = np.zeros((L_c, 2), dtype=np.float32)
    x0 = np.asarray(start[:2], dtype=float)
    ctx = np.repeat(x0[None, :], 2, axis=0)
    heading = compute_heading_from_context(ctx, goal)
    u_local = estimate_nominal_goal_control_local(scenario, x0, goal, heading, u_max_nom)
    step = propagate_global_state(scenario, x0, u_local, heading, dt_nom) - x0
    if np.linalg.norm(step) < 1e-8:
        step = np.array([1.0, 0.0], dtype=float)
    step = step / max(np.linalg.norm(step), 1e-8) * min(np.linalg.norm(step), 0.25)
    for k in range(L_c):
        lag = L_c - 1 - k
        hist[k, :] = (x0 - lag * step).astype(np.float32)
    return hist


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def load_s1_checkpoint(model_path: Path, device: torch.device):
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
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
    norm = {k: np.asarray(v, dtype=np.float32) for k, v in norm_raw.items()}
    return model, norm, meta


def save_s1_checkpoint(model: nn.Module, model_out: Path, meta: Dict[str, Any], norm: Dict[str, np.ndarray]) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "nonlinear_point_policy",
            "meta": meta,
            "norm": {k: np.asarray(v, dtype=np.float32) for k, v in norm.items()},
        },
        str(model_out),
    )


def rollout_policy(
    model: nn.Module,
    scenario: Any,
    norm: Dict[str, np.ndarray],
    device: torch.device,
    *,
    total_steps: int,
    dt_nom: float,
    u_max_nom: float,
    collision_margin: float,
    goal_tol: float,
    grid_n: int,
    n_steps_nom: int,
    buffer_cells: int,
    stop_tol: float,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rects = scenario_rects(scenario)
    goal = scenario_goal(scenario)
    ctx = build_initial_history_ending_at_start(scenario, int(norm["ctx_mean"].shape[0]), dt_nom, u_max_nom)
    sit_vec = compute_situation_vector(scenario, grid_n, dt_nom, n_steps_nom, u_max_nom, buffer_cells, stop_tol)
    x_curr = ctx[-1].astype(np.float32)

    traj_out: List[np.ndarray] = [x_curr.copy()]
    controls_out: List[np.ndarray] = []

    for k in range(int(total_steps)):
        if np.linalg.norm(x_curr - goal[:2]) <= float(goal_tol):
            break

        ctx_local, goal_local, origin, heading = transform_to_local(ctx, goal[:2])
        dyn_feat = nonlinear_dynamics_features(scenario, x_curr, heading)

        ctx_in = _normalize(ctx_local[None, :, :], norm["ctx_mean"][None, :, :], norm["ctx_std"][None, :, :]).astype(np.float32)
        sit_in = _normalize(sit_vec[None, :], norm["sit_mean"][None, :], norm["sit_std"][None, :]).astype(np.float32)
        dyn_in = _normalize(dyn_feat[None, :], norm["dyn_mean"][None, :], norm["dyn_std"][None, :]).astype(np.float32)
        goal_in = _normalize(goal_local[None, :], norm["goal_mean"][None, :], norm["goal_std"][None, :]).astype(np.float32)

        with torch.no_grad():
            t_ctx = torch.from_numpy(ctx_in).float().to(device)
            t_sit = torch.from_numpy(sit_in).float().to(device)
            t_dyn = torch.from_numpy(dyn_in).float().to(device)
            t_goal = torch.from_numpy(goal_in).float().to(device)
            u_local_norm = model(t_ctx, t_sit, t_dyn, t_goal).cpu().numpy()

        u_local = (u_local_norm * norm["u_std"][None, :] + norm["u_mean"][None, :]).squeeze(0).astype(np.float32)
        u_local = np.clip(u_local, -float(u_max_nom), float(u_max_nom))
        u_goal_local = estimate_nominal_goal_control_local(scenario, x_curr, goal, heading, u_max_nom)

        u_safe_local, x_next = choose_safe_control(
            scenario=scenario,
            x_curr=x_curr,
            u_pred_local=u_local,
            u_goal_local=u_goal_local,
            heading=heading,
            dt=dt_nom,
            rects=rects,
            goal=goal,
            u_max=u_max_nom,
            collision_margin=collision_margin,
        )

        if debug and k == 0:
            print("DEBUG x_curr =", x_curr)
            print("DEBUG dyn_feat =", dyn_feat)
            print("DEBUG u_local =", u_local)
            print("DEBUG u_goal_local =", u_goal_local)
            print("DEBUG u_safe_local =", u_safe_local)
            print("DEBUG x_next =", x_next)

        controls_out.append(u_safe_local.copy())
        traj_out.append(x_next.copy())
        ctx = np.concatenate([ctx, x_next[None, :]], axis=0)[-ctx.shape[0]:]
        x_curr = x_next

    traj = np.stack(traj_out, axis=0).astype(np.float32)
    controls = np.stack(controls_out, axis=0).astype(np.float32) if controls_out else np.zeros((0, 2), dtype=np.float32)
    info = {
        "collision_free": bool(collision_free_rectangles(traj[:, :2], rects, margin=float(collision_margin))),
        "solved": bool(goal_reached(traj[:, :2], goal, float(goal_tol))),
        "final_dist": float(np.linalg.norm(traj[-1, :2] - goal[:2])) if len(traj) else float("inf"),
    }
    return traj, controls, info


def load_result_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows
