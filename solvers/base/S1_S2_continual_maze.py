"""
S1_S2_continual_maze.py - memory + neural System-1 continual learning.

This is the maintained Base implementation used by S1_memory_neural.py. It keeps
the full-retrain behavior from the original continual_full_retrain.py while
using local mpc-sofai paths by default.

Convention:
    A: 2x2
    B: 2xm, usually 2x2 in the current maze experiments
    dyn = [A_local.flatten(), B_local.flatten(), drift_local]

Run from sofai_tool/sofai_instances/mpc-sofai:
    python Solvers/Base/S1_S2_continual_maze.py \
      --scenarios input/benchmark_scenarios_maze_1199_block200.json \
      --initial_model Solvers/s1_policy_control_cnn_diverse_5k.pth \
      --base_dataset Solvers/nn_dataset_maze_diverse_5k.npz \
      --base_memory_traj_npz Solvers/s1_sfcbf_success_trajs_diverse_5k.npz \
      --base_memory_scenarios Solvers/benchmark_scenarios_maze_diverse_5k.json \
      --workdir output/paper_1199_full_retrain_s2only
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

Rect = Tuple[float, float, float, float]


# ============================================================
# Robust imports
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
ROOT_DIR = PARENT_DIR.parent
for p in [THIS_DIR, PARENT_DIR, ROOT_DIR]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

try:
    import S1_NN_usage_maze as s1
except ImportError as e:
    raise ImportError("Could not import S1_NN_usage_maze.py. Put it in the same folder as this script.") from e

_DEFAULT_S2_SOLVER = None


def load_default_s2_solver():
    """Load the historical SFCBF System-2 solver only when a S2 path needs it."""
    global _DEFAULT_S2_SOLVER
    if _DEFAULT_S2_SOLVER is not None:
        return _DEFAULT_S2_SOLVER

    try:
        import S2_cbf_maze as s2_solver
    except ImportError:
        try:
            s2_solver = importlib.import_module("system_2.S2_cbf_maze")
        except ImportError:
            try:
                s2_solver = importlib.import_module("maze.system_2.S2_cbf_maze")
            except ImportError as e:
                raise ImportError(
                    "Could not import the default S2_cbf_maze.py. This is only required "
                    "for the legacy hybrid/full-retrain CLI paths; S1_memory_neural.py "
                    "does not need this solver."
                ) from e

    _DEFAULT_S2_SOLVER = s2_solver
    return _DEFAULT_S2_SOLVER


# ============================================================
# Shape helpers
# ============================================================

def ensure_A_B(A: Any, B: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Keep B as 2xm. Do NOT force it to 2x1."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != (2, 2):
        A = A.reshape(2, 2)

    if B.ndim == 1:
        if B.size == 2:
            B = B.reshape(2, 1)
        elif B.size == 4:
            B = B.reshape(2, 2)
        else:
            raise ValueError(f"Flat B must have length 2 or 4 for 2xm system, got shape {B.shape}")

    if B.ndim != 2 or B.shape[0] != 2:
        raise ValueError(f"Expected B shape (2,m), got {B.shape}")

    return A.astype(float), B.astype(float)


def expected_dyn_dim_from_u_dim(u_dim: int) -> int:
    return 4 + 2 * int(u_dim) + 2


# ============================================================
# Local frame / feature utilities
# ============================================================

def compute_heading_from_context(ctx_pos: np.ndarray, goal: np.ndarray) -> float:
    if ctx_pos.shape[0] >= 2:
        v = ctx_pos[-1] - ctx_pos[-2]
        if np.linalg.norm(v) > 1e-6:
            return float(np.arctan2(v[1], v[0]))

    vg = goal[:2] - ctx_pos[-1]
    if np.linalg.norm(vg) > 1e-8:
        return float(np.arctan2(vg[1], vg[0]))

    return 0.0


def rotation_from_heading(heading: float) -> np.ndarray:
    c, sn = np.cos(-heading), np.sin(-heading)
    return np.array([[c, -sn], [sn, c]], dtype=float)


def to_local_frame(context_pos: np.ndarray, next_pos: np.ndarray, goal: np.ndarray):
    origin = context_pos[-1].astype(float)
    heading = compute_heading_from_context(context_pos, goal)
    R = rotation_from_heading(heading)

    ctx_local = (context_pos - origin) @ R.T
    next_local = (next_pos - origin) @ R.T
    goal_local = (goal[:2] - origin) @ R.T

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


def collision_free_rectangles(states: np.ndarray, rects: List[Rect], margin: float = 0.0) -> bool:
    return s1.collision_free_rectangles(states, rects, margin=margin)


def goal_reached(states: np.ndarray, goal: np.ndarray, goal_tol: float) -> bool:
    if len(states) == 0:
        return False
    return float(np.linalg.norm(states[-1, :2] - goal[:2])) <= float(goal_tol)


# ============================================================
# Load S1 model
# ============================================================

def load_s1_model(model_path: Path, device: torch.device):
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    meta = ckpt["meta"]

    model = s1.NeuralSystem1ControlPolicyCNN(
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

    if "next_mean" in norm_raw and "next_std" in norm_raw:
        norm["next_mean"] = np.asarray(norm_raw["next_mean"], dtype=np.float32)
        norm["next_std"] = np.asarray(norm_raw["next_std"], dtype=np.float32)

    L_c, ctx_dim = meta["ctx_shape"]
    if ctx_dim != 2:
        raise ValueError(f"Expected 2D context, got {meta['ctx_shape']}")

    u_dim = int(meta["u_dim"])
    expected_dyn_dim = expected_dyn_dim_from_u_dim(u_dim)
    if int(meta["dyn_dim"]) != expected_dyn_dim:
        raise ValueError(
            f"Loaded model has dyn_dim={meta['dyn_dim']}, expected {expected_dyn_dim} for u_dim={u_dim}."
        )

    print(
        f"[load S1] {model_path} | ctx_shape={meta['ctx_shape']} | "
        f"u_dim={u_dim} | dyn_dim={meta['dyn_dim']}"
    )

    return model, norm, int(L_c), meta


# ============================================================
# Confidence-aware S1 rollout
# ============================================================

def normalize_ctx_np(ctx: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return ((ctx - norm["ctx_mean"][None, None, :]) / norm["ctx_std"][None, None, :]).astype(np.float32)


def normalize_vec_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def denormalize_u_np(u_norm: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return (u_norm * norm["u_std"][None, :] + norm["u_mean"][None, :]).astype(np.float32)


def current_local_inputs(ctx_global: np.ndarray, goal_global: np.ndarray):
    origin = ctx_global[-1].astype(float)
    heading = compute_heading_from_context(ctx_global, goal_global)
    R = rotation_from_heading(heading)

    ctx_local = (ctx_global - origin) @ R.T
    goal_local = (goal_global[:2] - origin) @ R.T

    return ctx_local.astype(np.float32), goal_local.astype(np.float32), origin.astype(np.float32), float(heading)


def predict_u_with_mc_dropout(
    model,
    ctx_in: np.ndarray,
    sit_in: np.ndarray,
    dyn_in: np.ndarray,
    goal_in: np.ndarray,
    norm: Dict[str, np.ndarray],
    device,
    args,
):
    """Run NN K times with dropout on; return mean control, uncertainty, and local confidence."""
    confidence_method = str(getattr(args, "confidence_method", "heuristic")).strip().lower()
    K = int(args.mc_dropout_samples)
    if confidence_method != "mc_dropout":
        K = 1

    t_ctx = torch.from_numpy(ctx_in).float().to(device)
    t_sit = torch.from_numpy(sit_in).float().to(device)
    t_dyn = torch.from_numpy(dyn_in).float().to(device)
    t_goal = torch.from_numpy(goal_in).float().to(device)

    if K <= 1:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            u_norm = model(t_ctx, t_sit, t_dyn, t_goal).detach().cpu().numpy()
        if was_training:
            model.train()
        u_raw = denormalize_u_np(u_norm, norm).squeeze(0)
        return u_raw.astype(np.float32), 0.0, 1.0

    was_training = model.training
    model.train()  # keep dropout ON

    preds = []
    with torch.no_grad():
        for _ in range(max(1, K)):
            u_norm = model(t_ctx, t_sit, t_dyn, t_goal).detach().cpu().numpy()
            u_raw = denormalize_u_np(u_norm, norm).squeeze(0)
            preds.append(u_raw)

    if not was_training:
        model.eval()

    preds = np.stack(preds, axis=0).astype(np.float32)
    u_mean = preds.mean(axis=0)
    u_var = 0.0 if K <= 1 else float(np.mean(np.var(preds, axis=0)))

    scale = max(float(args.local_uncertainty_scale), 1e-8)
    local_conf = float(np.exp(-u_var / scale))

    return u_mean.astype(np.float32), u_var, local_conf


def heuristic_local_confidence(
    *,
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
    goal_tol: float,
    args,
) -> Tuple[float, float]:
    """Cheap local confidence from one-step behavior.

    This avoids MC-dropout's K neural forward passes per rollout step. It uses
    signals already available in the controller loop:
      - predicted one-step progress toward the goal,
      - agreement with the nominal goal-seeking control,
      - immediate segment safety.
    """
    x_curr = np.asarray(x_curr, dtype=float).reshape(2)
    u_pred = np.asarray(u_pred, dtype=float).reshape(-1)
    u_goal = np.asarray(u_goal, dtype=float).reshape(-1)
    goal_xy = np.asarray(goal, dtype=float).reshape(-1)[:2]

    x_next = propagate_dynamics_local(x_curr, A, B, u_pred, dt).astype(float)

    dist_curr = float(np.linalg.norm(x_curr - goal_xy))
    dist_next = float(np.linalg.norm(x_next - goal_xy))
    progress = dist_curr - dist_next

    progress_scale = max(float(goal_tol), 0.25)
    progress_conf = 1.0 / (1.0 + np.exp(-4.0 * progress / progress_scale))

    denom = max(float(u_max), 1e-6) * max(float(np.sqrt(max(1, u_pred.size))), 1.0)
    control_gap = float(np.linalg.norm(u_pred - u_goal)) / denom
    align_scale = max(float(getattr(args, "confidence_control_scale", 1.0)), 1e-6)
    align_conf = float(np.exp(-control_gap / align_scale))

    safe = s1.segment_collision_free(
        x_curr,
        x_next,
        rects,
        margin=float(collision_margin),
        n_sub=int(getattr(args, "confidence_safety_substeps", 6)),
    )
    safety_conf = 1.0 if safe else 0.05

    local_conf = (
        0.45 * float(progress_conf)
        + 0.35 * float(align_conf)
        + 0.20 * float(safety_conf)
    )
    local_conf = float(np.clip(local_conf, 0.0, 1.0))
    local_uncertainty = float(1.0 - local_conf)
    return local_uncertainty, local_conf


def estimate_nominal_goal_control_local(
    x: np.ndarray,
    goal: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    u_max: float,
) -> np.ndarray:
    rhs = -(x - goal[:2]) - A @ x
    try:
        u = np.linalg.lstsq(B, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        u = np.zeros(B.shape[1], dtype=float)
    return np.clip(u, -u_max, u_max).astype(np.float32)


def propagate_dynamics_local(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    return (x + dt * (A @ x + B @ u)).astype(np.float32)


def choose_safe_control_confidence(
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
):
    """One-step safety/progress shield."""
    candidates = []

    candidates.append(u_pred)
    for a in [0.75, 0.5, 0.25]:
        candidates.append(a * u_pred + (1.0 - a) * u_goal)
    candidates.append(u_goal)

    for a in [0.5, 0.25, 0.1]:
        candidates.append(a * u_pred)

    candidates.append(np.zeros_like(u_pred))

    best_u = None
    best_x = None
    best_score = float("inf")

    dist_curr = float(np.linalg.norm(x_curr - goal[:2]))

    for u in candidates:
        u = np.clip(u, -u_max, u_max).astype(np.float32)
        x_next = propagate_dynamics_local(x_curr, A, B, u, dt)

        if not s1.segment_collision_free(
            x_curr,
            x_next,
            rects,
            margin=collision_margin,
            n_sub=12,
        ):
            continue

        dist_next = float(np.linalg.norm(x_next - goal[:2]))

        # Less aggressive, so S1 is not over-suppressed.
        progress_penalty = 0.0 if dist_next < dist_curr else 2.0
        deviation_penalty = 0.01 * float(np.linalg.norm(u - u_pred))
        score = dist_next + progress_penalty + deviation_penalty

        if score < best_score:
            best_score = score
            best_u = u
            best_x = x_next

    if best_u is None:
        best_u = np.zeros_like(u_pred, dtype=np.float32)
        best_x = x_curr.copy().astype(np.float32)

    return best_u, best_x


def rollout_neural_s1_with_confidence(
    model,
    ctx_global_init: np.ndarray,
    sit_vec: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    goal: np.ndarray,
    norm: Dict[str, np.ndarray],
    device,
    total_steps: int,
    dt_nom: float,
    u_max_nom: float,
    collision_margin: float,
    goal_tol: float,
    rects: List[Rect],
    global_s1_accept_rate: float,
    args,
):
    """S1 rollout with combined local/global confidence."""
    ctx = ctx_global_init.copy().astype(np.float32)

    traj_out = []
    controls_out = []

    local_conf_list = []
    local_uncertainty_list = []
    combined_conf_list = []

    low_conf_count = 0
    confidence_triggered = False

    w_global = min(max(float(args.confidence_global_weight), 0.0), 1.0)
    global_conf = min(max(float(global_s1_accept_rate), 0.0), 1.0)

    for k in range(int(total_steps)):
        x_curr = ctx[-1].astype(np.float32)

        if np.linalg.norm(x_curr - goal[:2]) <= goal_tol:
            break

        ctx_local, goal_local, origin, heading = current_local_inputs(ctx, goal[:2])
        dyn_feat = local_dynamics_features(A, B, heading, origin)

        ctx_in = normalize_ctx_np(ctx_local[None, :, :], norm)
        sit_in = normalize_vec_np(sit_vec[None, :], norm["sit_mean"], norm["sit_std"])
        dyn_in = normalize_vec_np(dyn_feat[None, :], norm["dyn_mean"], norm["dyn_std"])
        goal_in = normalize_vec_np(goal_local[None, :], norm["goal_mean"], norm["goal_std"])

        u_pred, local_uncertainty, local_conf = predict_u_with_mc_dropout(
            model=model,
            ctx_in=ctx_in,
            sit_in=sit_in,
            dyn_in=dyn_in,
            goal_in=goal_in,
            norm=norm,
            device=device,
            args=args,
        )

        u_pred = np.clip(u_pred, -u_max_nom, u_max_nom)

        u_goal = estimate_nominal_goal_control_local(
            x=x_curr,
            goal=goal,
            A=A,
            B=B,
            u_max=u_max_nom,
        )

        if str(getattr(args, "confidence_method", "heuristic")).strip().lower() != "mc_dropout":
            local_uncertainty, local_conf = heuristic_local_confidence(
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
                goal_tol=goal_tol,
                args=args,
            )

        combined_conf = w_global * global_conf + (1.0 - w_global) * local_conf

        local_conf_list.append(float(local_conf))
        local_uncertainty_list.append(float(local_uncertainty))
        combined_conf_list.append(float(combined_conf))

        if (
            args.enable_confidence_switch
            and k >= int(args.confidence_min_steps)
            and combined_conf < float(args.confidence_threshold)
        ):
            low_conf_count += 1
        else:
            low_conf_count = 0

        if low_conf_count >= int(args.confidence_patience):
            confidence_triggered = True
            break

        u_safe, x_next = choose_safe_control_confidence(
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

        traj_out.append(x_next.copy())
        controls_out.append(u_safe.copy())

        ctx = np.concatenate([ctx, x_next[None, :]], axis=0)[-ctx.shape[0]:]

    if len(traj_out) == 0:
        states = np.zeros((0, 2), dtype=np.float32)
        controls = np.zeros((0, B.shape[1]), dtype=np.float32)
    else:
        states = np.stack(traj_out, axis=0).astype(np.float32)
        controls = np.stack(controls_out, axis=0).astype(np.float32)

    conf_info = {
        "confidence_triggered": bool(confidence_triggered),
        "global_s1_accept_rate_used": float(global_conf),
        "local_confidence_mean": float(np.mean(local_conf_list)) if local_conf_list else 0.0,
        "local_confidence_min": float(np.min(local_conf_list)) if local_conf_list else 0.0,
        "local_uncertainty_mean": float(np.mean(local_uncertainty_list)) if local_uncertainty_list else float("inf"),
        "local_uncertainty_max": float(np.max(local_uncertainty_list)) if local_uncertainty_list else float("inf"),
        "combined_confidence_mean": float(np.mean(combined_conf_list)) if combined_conf_list else 0.0,
        "combined_confidence_min": float(np.min(combined_conf_list)) if combined_conf_list else 0.0,
        "low_conf_count": int(low_conf_count),
    }

    return states, controls, conf_info


# ============================================================
# Run S1 / S2
# ============================================================

def run_s1_on_scenario(
    sc: Dict[str, Any],
    model,
    norm: Dict[str, np.ndarray],
    L_c: int,
    device,
    args,
    global_s1_accept_rate: float,
) -> Dict[str, Any]:
    sid = int(sc.get("scenario_id", -1))
    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    rects = [tuple(map(float, r)) for r in sc["rectangles"]]
    start = np.asarray(sc.get("start", (5.0, 5.0)), dtype=float)
    goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)
    bounds = sc.get("bounds", [-10, -10, 10, 10])

    t0 = time.perf_counter()

    sit_vec = s1.compute_situation_vector(
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

    ctx_global_init = s1.build_initial_history_ending_at_start(
        A=A,
        B=B,
        start=start,
        goal=goal,
        L_c=L_c,
        dt_nom=args.dt,
        u_max_nom=args.u_max,
    )

    states_no_start, controls, conf_info = rollout_neural_s1_with_confidence(
        model=model,
        ctx_global_init=ctx_global_init,
        sit_vec=sit_vec,
        A=A,
        B=B,
        goal=goal,
        norm=norm,
        device=device,
        total_steps=args.s1_steps,
        dt_nom=args.dt,
        u_max_nom=args.u_max,
        collision_margin=args.collision_margin,
        goal_tol=args.goal_tol,
        rects=rects,
        global_s1_accept_rate=global_s1_accept_rate,
        args=args,
    )

    runtime = float(time.perf_counter() - t0)

    if len(states_no_start) > 0:
        states = np.vstack([start.reshape(1, 2), states_no_start])
    else:
        states = start.reshape(1, 2)

    cf = collision_free_rectangles(states, rects, margin=args.collision_margin)
    gr = goal_reached(states, goal, args.goal_tol)
    success = bool(cf and gr)

    return {
        "scenario_id": sid,
        "source": "S1",
        "used_system": "S1",
        "fallback_to_s2": False,
        "success": success,
        "collision_free": bool(cf),
        "goal_reached": bool(gr),
        "runtime_sec": runtime,
        "states": states.tolist(),
        "inputs": controls.tolist(),
        "final_dist": float(np.linalg.norm(states[-1, :2] - goal[:2])),
        **conf_info,
    }


def run_s2_on_scenario(sc: Dict[str, Any], args) -> Dict[str, Any]:
    s2_solver = load_default_s2_solver()
    sid = int(sc.get("scenario_id", -1))
    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    rects = [tuple(map(float, r)) for r in sc["rectangles"]]
    start = tuple(map(float, sc.get("start", (5.0, 5.0))))
    goal = tuple(map(float, sc.get("goal", (0.0, 0.0))))

    out = s2_solver.simulate_sfcbf(
        A=A,
        B=B,
        rects=rects,
        start=start,
        goal=goal,
        dt=args.dt,
        n_steps=args.s2_steps,
        u_max=args.u_max,
        margin=args.s2_margin,
        gamma=args.s2_gamma,
        goal_tol=args.goal_tol,
        collision_margin=args.collision_margin,
    )

    states = np.asarray(out["states"], dtype=float)
    goal_np = np.asarray(goal, dtype=float)

    return {
        "scenario_id": sid,
        "source": "S2",
        "used_system": "S2",
        "fallback_to_s2": True,
        "success": bool(out["success"]),
        "collision_free": bool(out["collision_free"]),
        "goal_reached": bool(out["goal_reached"]),
        "runtime_sec": float(out["runtime_sec"]),
        "states": out["states"],
        "inputs": out["inputs"],
        "final_dist": float(np.linalg.norm(states[-1, :2] - goal_np[:2])),
        "ok_qp_all_steps": bool(out.get("ok_qp_all_steps", False)),
    }


def _s1_attempt_summary(s1_out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": s1_out["success"],
        "collision_free": s1_out["collision_free"],
        "goal_reached": s1_out["goal_reached"],
        "final_dist": s1_out["final_dist"],
        "runtime_sec": s1_out["runtime_sec"],
        "states": s1_out["states"],
        "inputs": s1_out["inputs"],
        "confidence_triggered": s1_out.get("confidence_triggered", False),
        "global_s1_accept_rate_used": s1_out.get("global_s1_accept_rate_used", 0.0),
        "local_confidence_mean": s1_out.get("local_confidence_mean", 0.0),
        "local_confidence_min": s1_out.get("local_confidence_min", 0.0),
        "local_uncertainty_mean": s1_out.get("local_uncertainty_mean", float("inf")),
        "local_uncertainty_max": s1_out.get("local_uncertainty_max", float("inf")),
        "combined_confidence_mean": s1_out.get("combined_confidence_mean", 0.0),
        "combined_confidence_min": s1_out.get("combined_confidence_min", 0.0),
    }


def _s1_memory_attempt_summary(mem_out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "attempted": bool(mem_out.get("attempted", False)),
        "success": bool(mem_out.get("success", False)),
        "collision_free": bool(mem_out.get("collision_free", False)),
        "goal_reached": bool(mem_out.get("goal_reached", False)),
        "final_dist": float(mem_out.get("final_dist", float("inf"))),
        "runtime_sec": float(mem_out.get("runtime_sec", 0.0)),
        "best_score": float(mem_out.get("episodic_memory_score", 0.0)),
        "best_dyn_distance": float(mem_out.get("episodic_memory_dyn_distance", float("inf"))),
        "best_map_similarity": float(mem_out.get("episodic_memory_map_similarity", 0.0)),
        "candidates_above_threshold": int(mem_out.get("episodic_memory_candidates", 0)),
        "candidates_tried": int(mem_out.get("episodic_memory_tried", 0)),
        "matched_scenario_id": mem_out.get("episodic_memory_matched_scenario_id", None),
    }


def empty_episodic_memory() -> Dict[str, Any]:
    return {
        "version": 1,
        "description": "Successful S2 trajectories promoted into System-1 episodic retrieval memory.",
        "items": [],
    }


def load_episodic_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return empty_episodic_memory()
    payload = json.loads(path.read_text())
    if "items" not in payload or not isinstance(payload["items"], list):
        raise ValueError(f"Invalid episodic memory file: {path}")
    payload.setdefault("version", 1)
    payload.setdefault("description", "Successful S2 trajectories promoted into System-1 episodic retrieval memory.")
    return payload


def save_episodic_memory(memory: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, indent=2))


def episodic_memory_size(memory: Optional[Dict[str, Any]]) -> int:
    if memory is None:
        return 0
    return int(len(memory.get("items", [])))


def cosine_binary_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = int(np.sum(a))
    bb = int(np.sum(b))
    if aa == 0 or bb == 0:
        return 0.0
    inter = int(np.sum((a.astype(bool) & b.astype(bool)).astype(np.uint8)))
    return float(inter) / float(np.sqrt(aa * bb))


def scenario_bucket_key(sc: Dict[str, Any], switch_reason: str = "") -> str:
    difficulty = str(sc.get("difficulty", "unknown"))
    map_type = str(sc.get("map_type", "unknown"))
    regime = str(sc.get("regime", sc.get("dynamics_regime", "unknown")))
    b_mode = str(sc.get("B_mode", sc.get("b_mode", "unknown")))
    reason = str(switch_reason or "unknown")
    return "|".join([difficulty, map_type, regime, b_mode, reason])


def compute_scenario_situation_vector(sc: Dict[str, Any], args) -> np.ndarray:
    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    rects = [tuple(map(float, r)) for r in sc["rectangles"]]
    start = np.asarray(sc.get("start", (5.0, 5.0)), dtype=float)
    goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)
    bounds = sc.get("bounds", [-10, -10, 10, 10])
    return s1.compute_situation_vector(
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
    ).astype(np.uint8)


def episodic_dynamics_distance(A_query: np.ndarray, B_query: np.ndarray, item: Dict[str, Any]) -> float:
    try:
        A_item, B_item = ensure_A_B(item["A_query"], item["B_query"])
    except Exception:
        return float("inf")
    if B_query.shape != B_item.shape:
        return float("inf")
    dA = float(np.linalg.norm(A_query - A_item, ord="fro") / (np.linalg.norm(A_item, ord="fro") + 1e-8))
    dB = float(np.linalg.norm(B_query - B_item, ord="fro") / (np.linalg.norm(B_item, ord="fro") + 1e-8))
    return float(np.sqrt(0.6 * dA * dA + 0.4 * dB * dB))


def make_episodic_memory_item(
    sc: Dict[str, Any],
    out: Dict[str, Any],
    args,
    block_id: int,
) -> Optional[Dict[str, Any]]:
    states = np.asarray(out.get("states", []), dtype=float)
    inputs = np.asarray(out.get("inputs", []), dtype=float)
    if states.ndim != 2 or len(states) < 2 or inputs.size == 0:
        return None

    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, B.shape[1])
    if inputs.shape[1] != B.shape[1]:
        return None

    max_steps = int(args.episodic_memory_max_traj_steps)
    if max_steps > 0:
        inputs = inputs[:max_steps]
        states = states[:max_steps + 1]

    sit_vec = compute_scenario_situation_vector(sc, args)
    switch_reason = str(out.get("switch_reason", "s2_success"))
    sc_id = int(sc.get("scenario_id", out.get("scenario_id", -1)))

    return {
        "scenario_id": sc_id,
        "created_block": int(block_id),
        "source": "S2",
        "memory_origin": str(out.get("memory_origin", "online_s2")),
        "dyn_id": int(sc.get("dyn_id", sc.get("base_dyn_id", -1))),
        "map_idx": int(sc.get("map_idx", sc_id)),
        "switch_reason": switch_reason,
        "bucket": scenario_bucket_key(sc, switch_reason=switch_reason),
        "difficulty": sc.get("difficulty", ""),
        "map_type": sc.get("map_type", ""),
        "regime": sc.get("regime", sc.get("dynamics_regime", "")),
        "B_mode": sc.get("B_mode", sc.get("b_mode", "")),
        "novelty_score": float(scenario_novelty_score(sc)),
        "A_query": A.tolist(),
        "B_query": B.tolist(),
        "situation_vec": sit_vec.astype(int).tolist(),
        "states": states[:, :2].tolist(),
        "inputs": inputs.tolist(),
        "success": bool(out.get("success", False)),
        "collision_free": bool(out.get("collision_free", False)),
        "goal_reached": bool(out.get("goal_reached", False)),
        "final_dist": float(out.get("final_dist", np.nan)),
        "runtime_sec": float(out.get("runtime_sec", np.nan)),
    }


def episodic_memory_item_key(item: Dict[str, Any]) -> Tuple[str, int, int, int]:
    return (
        str(item.get("memory_origin", "unknown")),
        int(item.get("scenario_id", -1)),
        int(item.get("dyn_id", -1)),
        int(item.get("map_idx", -1)),
    )


def cap_episodic_memory(memory: Dict[str, Any], args):
    items = list(memory.get("items", []))
    if not items:
        memory["items"] = []
        return

    max_per_bucket = int(args.episodic_memory_max_per_bucket)
    if max_per_bucket > 0:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for i, item in enumerate(items):
            item = dict(item)
            item["_insert_order"] = i
            buckets.setdefault(str(item.get("bucket", "unknown")), []).append(item)

        kept = []
        for bucket_items in buckets.values():
            bucket_items.sort(
                key=lambda z: (
                    float(z.get("novelty_score", 0.0)),
                    int(z.get("created_block", 0)),
                    int(z.get("_insert_order", 0)),
                ),
                reverse=True,
            )
            kept.extend(bucket_items[:max_per_bucket])
        items = kept

    max_total = int(args.episodic_memory_max_total)
    if max_total > 0 and len(items) > max_total:
        items.sort(
            key=lambda z: (
                float(z.get("novelty_score", 0.0)),
                int(z.get("created_block", 0)),
                int(z.get("_insert_order", 0)),
            ),
            reverse=True,
        )
        items = items[:max_total]

    for item in items:
        item.pop("_insert_order", None)
    memory["items"] = items


def infer_base_memory_paths(args) -> Tuple[Optional[Path], Optional[Path]]:
    traj_path = Path(args.base_memory_traj_npz) if args.base_memory_traj_npz else None
    scenarios_path = Path(args.base_memory_scenarios) if args.base_memory_scenarios else None

    base_dataset = Path(args.base_dataset)
    if traj_path is None or scenarios_path is None:
        stem = base_dataset.stem
        suffix = None
        if stem.startswith("nn_dataset_maze_"):
            suffix = stem[len("nn_dataset_maze_"):]
        elif stem == "nn_dataset_maze":
            suffix = "maze"

        if suffix:
            if traj_path is None:
                candidate = base_dataset.with_name(f"s1_sfcbf_success_trajs_{suffix}.npz")
                if candidate.exists():
                    traj_path = candidate
            if scenarios_path is None:
                candidate = base_dataset.with_name(f"benchmark_scenarios_maze_{suffix}.json")
                if candidate.exists():
                    scenarios_path = candidate

    return traj_path, scenarios_path


def add_base_trajectories_to_episodic_memory(
    memory: Optional[Dict[str, Any]],
    args,
) -> int:
    if memory is None or not args.enable_episodic_memory or not args.use_base_episodic_memory:
        return 0

    traj_path, scenarios_path = infer_base_memory_paths(args)
    if traj_path is None or scenarios_path is None or not traj_path.exists() or not scenarios_path.exists():
        print(
            "[memory] base memory skipped: could not find base trajectory NPZ and scenario JSON. "
            "Pass --base_memory_traj_npz and --base_memory_scenarios if you want base motion-primitive memory."
        )
        return 0

    traj_npz = np.load(str(traj_path), allow_pickle=True)
    scenarios = json.loads(scenarios_path.read_text())
    by_pair = {
        (int(sc.get("dyn_id", -1)), int(sc.get("map_idx", -1))): sc
        for sc in scenarios
    }
    by_id = {int(sc.get("scenario_id", i)): sc for i, sc in enumerate(scenarios)}

    existing = {episodic_memory_item_key(item) for item in memory.get("items", [])}
    added = 0
    max_items = int(args.base_memory_max_items)
    n = int(len(traj_npz["states"]))

    for i in range(n):
        if max_items > 0 and added >= max_items:
            break
        if "success" in traj_npz and not bool(traj_npz["success"][i]):
            continue

        dyn_id = int(traj_npz["dyn_id"][i]) if "dyn_id" in traj_npz else i
        map_idx = int(traj_npz["map_idx"][i]) if "map_idx" in traj_npz else i
        sc = by_pair.get((dyn_id, map_idx), by_id.get(i))
        if sc is None:
            continue

        states = np.asarray(traj_npz["states"][i], dtype=float)
        inputs = np.asarray(traj_npz["inputs"][i], dtype=float)
        if states.ndim != 2 or len(states) < 2 or inputs.size == 0:
            continue

        goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)
        out = {
            "states": states[:, :2].tolist(),
            "inputs": inputs.tolist(),
            "success": True,
            "collision_free": True,
            "goal_reached": True,
            "final_dist": float(np.linalg.norm(states[-1, :2] - goal[:2])),
            "runtime_sec": 0.0,
            "switch_reason": "base_motion_primitive",
            "memory_origin": "base_motion_primitives",
        }
        item = make_episodic_memory_item(sc, out, args, block_id=0)
        item["source"] = "S2_base"
        item["memory_origin"] = "base_motion_primitives"
        item["switch_reason"] = "base_motion_primitive"
        item["bucket"] = scenario_bucket_key(sc, switch_reason="base_motion_primitive")
        item["dyn_id"] = dyn_id
        item["map_idx"] = map_idx

        key = episodic_memory_item_key(item)
        if key in existing:
            continue

        memory.setdefault("items", []).append(item)
        existing.add(key)
        added += 1

    cap_episodic_memory(memory, args)
    print(
        f"[memory] added base motion-primitive memories: {added} | "
        f"traj_npz={traj_path} | scenarios={scenarios_path} | total={episodic_memory_size(memory)}"
    )
    return added


def add_s2_success_to_episodic_memory(
    memory: Optional[Dict[str, Any]],
    sc: Dict[str, Any],
    out: Dict[str, Any],
    args,
    block_id: int,
) -> bool:
    if memory is None or not args.enable_episodic_memory or not args.episodic_memory_store_s2_success:
        return False
    if out.get("used_system") != "S2":
        return False
    if not (out.get("success", False) and out.get("collision_free", False) and out.get("goal_reached", False)):
        return False

    item = make_episodic_memory_item(sc, out, args, block_id=block_id)
    if item is None:
        return False
    key = episodic_memory_item_key(item)
    if key in {episodic_memory_item_key(x) for x in memory.get("items", [])}:
        return False

    memory.setdefault("items", []).append(item)
    cap_episodic_memory(memory, args)
    return True


def replay_memory_controls_on_query(
    sc: Dict[str, Any],
    item: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    rects = [tuple(map(float, r)) for r in sc["rectangles"]]
    start = np.asarray(sc.get("start", (5.0, 5.0)), dtype=float)
    goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)

    inputs = np.asarray(item.get("inputs", []), dtype=float)
    if inputs.size == 0:
        return {"success": False, "collision_free": False, "goal_reached": False, "states": [start.tolist()], "inputs": []}
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, B.shape[1])
    if inputs.shape[1] != B.shape[1]:
        return {"success": False, "collision_free": False, "goal_reached": False, "states": [start.tolist()], "inputs": []}

    max_steps = int(args.episodic_memory_replay_steps)
    if max_steps <= 0:
        max_steps = len(inputs)
    max_steps = min(max_steps, len(inputs))

    states = [start.astype(float)]
    used_inputs = []
    x = start.astype(float)
    collision_free = True

    for u in inputs[:max_steps]:
        u = np.clip(np.asarray(u, dtype=float).reshape(B.shape[1]), -float(args.u_max), float(args.u_max))
        x_next = x + float(args.dt) * (A @ x + B @ u)
        if not s1.segment_collision_free(
            x,
            x_next,
            rects,
            margin=float(args.collision_margin),
            n_sub=12,
        ):
            collision_free = False
            states.append(x_next.astype(float))
            used_inputs.append(u.astype(float))
            break
        states.append(x_next.astype(float))
        used_inputs.append(u.astype(float))
        x = x_next.astype(float)
        if float(np.linalg.norm(x - goal[:2])) <= float(args.goal_tol):
            break

    states_np = np.asarray(states, dtype=float)
    if collision_free:
        collision_free = collision_free_rectangles(states_np, rects, margin=args.collision_margin)
    reached = goal_reached(states_np, goal, args.goal_tol)

    return {
        "success": bool(collision_free and reached),
        "collision_free": bool(collision_free),
        "goal_reached": bool(reached),
        "states": states_np[:, :2].tolist(),
        "inputs": np.asarray(used_inputs, dtype=float).reshape(-1, B.shape[1]).tolist() if used_inputs else [],
        "final_dist": float(np.linalg.norm(states_np[-1, :2] - goal[:2])),
    }


def run_episodic_memory_on_scenario(
    sc: Dict[str, Any],
    memory: Optional[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    sid = int(sc.get("scenario_id", -1))
    t0 = time.perf_counter()
    base = {
        "scenario_id": sid,
        "source": "S1_memory",
        "attempted": False,
        "success": False,
        "collision_free": False,
        "goal_reached": False,
        "runtime_sec": 0.0,
        "final_dist": float("inf"),
        "episodic_memory_score": 0.0,
        "episodic_memory_dyn_distance": float("inf"),
        "episodic_memory_map_similarity": 0.0,
        "episodic_memory_candidates": 0,
        "episodic_memory_tried": 0,
        "episodic_memory_matched_scenario_id": None,
    }
    if memory is None or not args.enable_episodic_memory:
        base["runtime_sec"] = float(time.perf_counter() - t0)
        return base

    items = list(memory.get("items", []))
    if not items:
        base["runtime_sec"] = float(time.perf_counter() - t0)
        return base

    A_query, B_query = ensure_A_B(sc["A_query"], sc["B_query"])
    v_query = compute_scenario_situation_vector(sc, args)
    candidates = []

    dyn_sigma = max(float(args.episodic_memory_dyn_sigma), 1e-8)
    for item in items:
        try:
            v_item = np.asarray(item.get("situation_vec", []), dtype=np.uint8)
            if v_item.shape != v_query.shape:
                continue
            map_sim = cosine_binary_np(v_query, v_item)
            if map_sim < float(args.episodic_memory_map_threshold):
                continue
            dyn_dist = episodic_dynamics_distance(A_query, B_query, item)
            if not np.isfinite(dyn_dist):
                continue
            score = float(np.exp(-(dyn_dist * dyn_dist) / (dyn_sigma * dyn_sigma)) * map_sim)
            if score >= float(args.episodic_memory_score_threshold):
                candidates.append((score, dyn_dist, map_sim, item))
        except Exception:
            continue

    candidates.sort(key=lambda z: z[0], reverse=True)
    base["episodic_memory_candidates"] = int(len(candidates))
    if candidates:
        base["attempted"] = True
        base["episodic_memory_score"] = float(candidates[0][0])
        base["episodic_memory_dyn_distance"] = float(candidates[0][1])
        base["episodic_memory_map_similarity"] = float(candidates[0][2])
        base["episodic_memory_matched_scenario_id"] = candidates[0][3].get("scenario_id", None)
    else:
        base["runtime_sec"] = float(time.perf_counter() - t0)
        return base

    best_failed = None
    for score, dyn_dist, map_sim, item in candidates[:max(1, int(args.episodic_memory_top_k))]:
        replay = replay_memory_controls_on_query(sc, item, args)
        base["episodic_memory_tried"] += 1
        replay.update({
            "scenario_id": sid,
            "source": "S1_memory",
            "used_system": "S1_memory",
            "fallback_to_s2": False,
            "runtime_sec": float(time.perf_counter() - t0),
            "attempted": True,
            "episodic_memory_score": float(score),
            "episodic_memory_dyn_distance": float(dyn_dist),
            "episodic_memory_map_similarity": float(map_sim),
            "episodic_memory_candidates": int(len(candidates)),
            "episodic_memory_tried": int(base["episodic_memory_tried"]),
            "episodic_memory_matched_scenario_id": item.get("scenario_id", None),
            "episodic_memory_created_block": item.get("created_block", None),
        })
        if replay["success"]:
            replay["switch_reason"] = "s1_memory_success"
            return replay
        if best_failed is None or float(replay.get("final_dist", float("inf"))) < float(best_failed.get("final_dist", float("inf"))):
            best_failed = replay

    if best_failed is not None:
        base.update({
            "collision_free": bool(best_failed.get("collision_free", False)),
            "goal_reached": bool(best_failed.get("goal_reached", False)),
            "final_dist": float(best_failed.get("final_dist", float("inf"))),
            "states": best_failed.get("states", []),
            "inputs": best_failed.get("inputs", []),
        })
    base["runtime_sec"] = float(time.perf_counter() - t0)
    return base


def run_hybrid_on_scenario(
    sc: Dict[str, Any],
    model,
    norm: Optional[Dict[str, np.ndarray]],
    L_c: Optional[int],
    device,
    args,
    global_s1_accept_rate: float,
    episodic_memory: Optional[Dict[str, Any]] = None,
):
    memory_attempt = None

    if args.enable_episodic_memory and args.episodic_memory_try_before_neural:
        mem_out = run_episodic_memory_on_scenario(sc, episodic_memory, args)
        if mem_out.get("attempted", False):
            if mem_out.get("success", False):
                return mem_out
            memory_attempt = _s1_memory_attempt_summary(mem_out)

    if model is not None and norm is not None and L_c is not None:
        s1_out = run_s1_on_scenario(
            sc=sc,
            model=model,
            norm=norm,
            L_c=L_c,
            device=device,
            args=args,
            global_s1_accept_rate=global_s1_accept_rate,
        )

        if s1_out.get("confidence_triggered", False):
            if args.enable_episodic_memory and not args.episodic_memory_try_before_neural:
                mem_out = run_episodic_memory_on_scenario(sc, episodic_memory, args)
                if mem_out.get("attempted", False):
                    if mem_out.get("success", False):
                        mem_out["switch_reason"] = "s1_memory_after_low_confidence_success"
                        mem_out["s1_attempt"] = _s1_attempt_summary(s1_out)
                        return mem_out
                    memory_attempt = _s1_memory_attempt_summary(mem_out)
            s2_out = run_s2_on_scenario(sc, args)
            s2_out["used_system"] = "S2"
            s2_out["fallback_to_s2"] = True
            s2_out["switch_reason"] = "low_combined_confidence"
            s2_out["s1_attempt"] = _s1_attempt_summary(s1_out)
            if memory_attempt is not None:
                s2_out["s1_memory_attempt"] = memory_attempt
            return s2_out

        if s1_out["success"]:
            s1_out["switch_reason"] = "s1_success"
            if memory_attempt is not None:
                s1_out["s1_memory_attempt"] = memory_attempt
            return s1_out

        if args.enable_episodic_memory and not args.episodic_memory_try_before_neural:
            mem_out = run_episodic_memory_on_scenario(sc, episodic_memory, args)
            if mem_out.get("attempted", False):
                if mem_out.get("success", False):
                    mem_out["switch_reason"] = "s1_memory_after_neural_failure_success"
                    mem_out["s1_attempt"] = _s1_attempt_summary(s1_out)
                    return mem_out
                memory_attempt = _s1_memory_attempt_summary(mem_out)

        s2_out = run_s2_on_scenario(sc, args)
        s2_out["used_system"] = "S2"
        s2_out["fallback_to_s2"] = True
        s2_out["switch_reason"] = "s1_failed"
        s2_out["s1_attempt"] = _s1_attempt_summary(s1_out)
        if memory_attempt is not None:
            s2_out["s1_memory_attempt"] = memory_attempt
        return s2_out

    if args.enable_episodic_memory and not args.episodic_memory_try_before_neural:
        mem_out = run_episodic_memory_on_scenario(sc, episodic_memory, args)
        if mem_out.get("attempted", False):
            if mem_out.get("success", False):
                return mem_out
            memory_attempt = _s1_memory_attempt_summary(mem_out)

    s2_out = run_s2_on_scenario(sc, args)
    if memory_attempt is not None:
        s2_out["s1_memory_attempt"] = memory_attempt
        s2_out["switch_reason"] = "s1_memory_failed"
    return s2_out


# ============================================================
# DAgger collection
# ============================================================

def scenario_novelty_score(sc: Dict[str, Any]) -> float:
    """Heuristic novelty score from benchmark metadata.

    The balanced benchmark generator writes difficulty/map/dynamics labels.
    When older scenario files do not contain these fields, this returns a
    conservative middle value instead of treating every record as novel.
    """
    difficulty = str(sc.get("difficulty", "")).lower()
    map_type = str(sc.get("map_type", "")).lower()
    regime = str(sc.get("regime", sc.get("dynamics_regime", ""))).lower()
    b_mode = str(sc.get("B_mode", sc.get("b_mode", ""))).lower()

    diff_score = {"easy": 0.15, "medium": 0.55, "hard": 1.0}.get(difficulty, 0.45)
    map_score = {
        "scattered": 0.35,
        "wall_gap": 0.65,
        "bottleneck": 0.85,
        "u_shape": 0.90,
    }.get(map_type, 0.50)
    regime_score = {
        "damped": 0.25,
        "axis_scaled": 0.45,
        "anisotropic": 0.60,
        "shear": 0.70,
        "rotate_cw": 0.80,
        "rotate_ccw": 0.80,
        "mixed": 0.85,
    }.get(regime, 0.50)
    b_score = {
        "identity": 0.15,
        "axis_scaled": 0.45,
        "rotated_scaled": 0.70,
        "random_invertible": 0.85,
    }.get(b_mode, 0.45)

    return float(np.clip(0.35 * diff_score + 0.30 * map_score + 0.20 * regime_score + 0.15 * b_score, 0.0, 1.0))


def dagger_record_weight(sc: Dict[str, Any], out: Dict[str, Any], args) -> float:
    novelty = scenario_novelty_score(sc)
    reason = str(out.get("switch_reason", ""))
    low_conf_bonus = float(args.dagger_low_conf_bonus) if reason == "low_combined_confidence" else 0.0
    failed_bonus = float(args.dagger_failed_s1_bonus) if reason == "s1_failed" else 0.0
    raw = float(args.dagger_sample_weight) * (
        1.0
        + float(args.dagger_novelty_weight) * novelty
        + low_conf_bonus
        + failed_bonus
    )
    return float(np.clip(raw, float(args.dagger_min_sample_weight), float(args.max_sample_weight)))


def s2_full_record_weight(sc: Dict[str, Any], out: Dict[str, Any], args) -> float:
    """Weight for full successful S2 trajectory distillation.

    Full S2 records are broader than local DAgger corrections, so they are
    useful for transfer. Keep them less aggressive than local failure
    corrections to avoid overwriting the base policy.
    """
    novelty = scenario_novelty_score(sc)
    reason = str(out.get("switch_reason", ""))
    fallback_bonus = 0.25 if reason in {"low_combined_confidence", "s1_failed"} else 0.0
    raw = float(args.s2_full_sample_weight) * (
        1.0
        + float(args.s2_full_novelty_weight) * novelty
        + fallback_bonus
    )
    return float(np.clip(raw, float(args.dagger_min_sample_weight), float(args.max_sample_weight)))


def _record_from_expert_step(
    *,
    sc_id: int,
    ctx: np.ndarray,
    u_expert: np.ndarray,
    x_next_expert: np.ndarray,
    sc: Dict[str, Any],
    record_type: str,
    switch_reason: str,
    sample_weight: float,
    novelty: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    states = np.vstack([ctx[:, :2], x_next_expert.reshape(1, 2)])
    rec = {
        "scenario_id": int(sc_id),
        "states": states.tolist(),
        "inputs": u_expert.reshape(1, -1).tolist(),
        "success": True,
        "used_system": record_type,
        "record_type": record_type,
        "switch_reason": switch_reason,
        "novelty_score": float(novelty),
        "sample_weight": float(sample_weight),
        "difficulty": sc.get("difficulty", ""),
        "map_type": sc.get("map_type", ""),
        "regime": sc.get("regime", sc.get("dynamics_regime", "")),
        "B_mode": sc.get("B_mode", sc.get("b_mode", "")),
    }
    if extra:
        rec.update(extra)
    return rec


def collect_dagger_records_from_failed_s1(out: Dict[str, Any], scenarios_by_id: Dict[int, Dict[str, Any]], args):
    records = []
    if out.get("used_system") != "S2" or out.get("s1_attempt") is None:
        return records

    sc_id = int(out["scenario_id"])
    sc = scenarios_by_id[sc_id]
    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    rects = [tuple(map(float, r)) for r in sc["rectangles"]]
    goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)

    switch_reason = str(out.get("switch_reason", ""))
    if switch_reason == "low_combined_confidence" and not args.dagger_collect_low_conf:
        return records
    if switch_reason == "s1_failed" and not args.dagger_collect_failed_s1:
        return records

    s2_successful = bool(out.get("success", False) and out.get("collision_free", False) and out.get("goal_reached", False))
    if args.dagger_require_s2_success and not s2_successful:
        return records

    s1_states = np.asarray(out["s1_attempt"]["states"], dtype=float)
    if s1_states.ndim != 2 or len(s1_states) < args.L_c + 1:
        return records

    s1_final_dist = float(out["s1_attempt"].get("final_dist", np.inf))
    s2_final_dist = float(out.get("final_dist", np.inf))
    final_improvement = s1_final_dist - s2_final_dist
    if (not s2_successful) and final_improvement < float(args.dagger_min_final_dist_improvement):
        return records

    attempt_unc = float(out["s1_attempt"].get("local_uncertainty_mean", np.nan))
    if np.isfinite(attempt_unc) and attempt_unc > float(args.dagger_max_attempt_uncertainty):
        return records

    base_weight = dagger_record_weight(sc, out, args)
    novelty = scenario_novelty_score(sc)

    for k in range(0, len(s1_states) - args.L_c, args.dagger_stride):
        ctx = s1_states[k:k + args.L_c, :2]
        expert_start = tuple(ctx[-1])

        s2_solver = load_default_s2_solver()
        s2_step = s2_solver.simulate_sfcbf(
            A=A,
            B=B,
            rects=rects,
            start=expert_start,
            goal=tuple(goal),
            dt=args.dt,
            n_steps=args.dagger_s2_steps,
            u_max=args.u_max,
            margin=args.s2_margin,
            gamma=args.s2_gamma,
            goal_tol=args.goal_tol,
            collision_margin=args.collision_margin,
        )

        inputs = np.asarray(s2_step.get("inputs", []), dtype=float)
        states = np.asarray(s2_step.get("states", []), dtype=float)
        if inputs.size == 0 or len(states) < 2:
            continue

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, B.shape[1])

        u_expert = inputs[0].reshape(B.shape[1])
        x_next_expert = states[1, :2]

        if not s1.segment_collision_free(ctx[-1, :2], x_next_expert[:2], rects, margin=args.collision_margin, n_sub=12):
            continue

        dist_curr = float(np.linalg.norm(ctx[-1, :2] - goal[:2]))
        dist_next = float(np.linalg.norm(x_next_expert[:2] - goal[:2]))
        if dist_next - dist_curr > float(args.dagger_allow_nonprogress):
            continue

        records.append(_record_from_expert_step(
            sc_id=sc_id,
            ctx=ctx,
            u_expert=u_expert,
            x_next_expert=x_next_expert,
            sc=sc,
            record_type="S2_dagger",
            switch_reason=switch_reason,
            sample_weight=base_weight,
            novelty=novelty,
            extra={
                "s2_successful": s2_successful,
                "s1_final_dist": s1_final_dist,
                "s2_final_dist": s2_final_dist,
                "final_dist_improvement": final_improvement,
                "local_dist_improvement": dist_curr - dist_next,
            },
        ))

    return records


def collect_full_s2_records(out: Dict[str, Any], scenarios_by_id: Dict[int, Dict[str, Any]], args):
    """Distill full successful S2 fallback trajectories into S1 training records.

    Local DAgger teaches recovery from states visited by the failing S1 policy.
    Full S2 distillation teaches the complete successful behavior, which is the
    part that can transfer better to future maps/dynamics.
    """
    records = []
    if not args.collect_full_s2_trajs:
        return records
    if out.get("used_system") != "S2":
        return records
    if args.s2_full_require_fallback and out.get("s1_attempt") is None:
        return records
    if not (out.get("success", False) and out.get("collision_free", False) and out.get("goal_reached", False)):
        return records

    sc_id = int(out["scenario_id"])
    sc = scenarios_by_id[sc_id]
    A, B = ensure_A_B(sc["A_query"], sc["B_query"])
    rects = [tuple(map(float, r)) for r in sc["rectangles"]]
    goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)

    states = np.asarray(out.get("states", []), dtype=float)
    inputs = np.asarray(out.get("inputs", []), dtype=float)
    if states.ndim != 2 or inputs.size == 0:
        return records
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, B.shape[1])

    L_c = int(args.L_c)
    max_t = min(len(inputs) - 1, len(states) - 2)
    if max_t < L_c - 1:
        return records

    weight = s2_full_record_weight(sc, out, args)
    novelty = scenario_novelty_score(sc)
    switch_reason = str(out.get("switch_reason", ""))
    kept_for_traj = 0

    for t in range(L_c - 1, max_t + 1, max(1, int(args.s2_full_stride))):
        if args.s2_full_max_records_per_traj > 0 and kept_for_traj >= int(args.s2_full_max_records_per_traj):
            break

        ctx = states[t - L_c + 1:t + 1, :2]
        u_expert = inputs[t].reshape(B.shape[1])
        x_next = states[t + 1, :2]

        if not s1.segment_collision_free(ctx[-1, :2], x_next[:2], rects, margin=args.collision_margin, n_sub=12):
            continue

        dist_curr = float(np.linalg.norm(ctx[-1, :2] - goal[:2]))
        dist_next = float(np.linalg.norm(x_next[:2] - goal[:2]))
        if dist_next - dist_curr > float(args.s2_full_allow_nonprogress):
            continue

        records.append(_record_from_expert_step(
            sc_id=sc_id,
            ctx=ctx,
            u_expert=u_expert,
            x_next_expert=x_next,
            sc=sc,
            record_type="S2_full",
            switch_reason=switch_reason,
            sample_weight=weight,
            novelty=novelty,
            extra={
                "local_dist_improvement": dist_curr - dist_next,
                "s2_traj_len": int(len(states)),
            },
        ))
        kept_for_traj += 1

    return records


# ============================================================
# Continual dataset: replay + small DAgger
# ============================================================

def _get_norm_from_npz(data, key: str) -> Tuple[np.ndarray, np.ndarray]:
    return data[f"norm_{key}_mean"].astype(np.float32), data[f"norm_{key}_std"].astype(np.float32)


def _denorm_np(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray, is_ctx: bool = False) -> np.ndarray:
    if is_ctx:
        return x_norm * std[None, None, :] + mean[None, None, :]
    return x_norm * std[None, :] + mean[None, :]


def _norm_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray, is_ctx: bool = False) -> np.ndarray:
    if is_ctx:
        return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def _reference_stats(reference_norm: Dict[str, np.ndarray], base_data=None) -> Dict[str, np.ndarray]:
    ref = {
        "ctx_mean": reference_norm["ctx_mean"].astype(np.float32),
        "ctx_std": reference_norm["ctx_std"].astype(np.float32),
        "sit_mean": reference_norm["sit_mean"].astype(np.float32),
        "sit_std": reference_norm["sit_std"].astype(np.float32),
        "dyn_mean": reference_norm["dyn_mean"].astype(np.float32),
        "dyn_std": reference_norm["dyn_std"].astype(np.float32),
        "goal_mean": reference_norm["goal_mean"].astype(np.float32),
        "goal_std": reference_norm["goal_std"].astype(np.float32),
        "u_mean": reference_norm["u_mean"].astype(np.float32),
        "u_std": reference_norm["u_std"].astype(np.float32),
    }
    if "next_mean" in reference_norm and "next_std" in reference_norm:
        ref["next_mean"] = reference_norm["next_mean"].astype(np.float32)
        ref["next_std"] = reference_norm["next_std"].astype(np.float32)
    elif base_data is not None:
        ref["next_mean"] = base_data["norm_next_mean"].astype(np.float32)
        ref["next_std"] = base_data["norm_next_std"].astype(np.float32)
    else:
        raise ValueError("reference_norm is missing next_mean/next_std and no base_data fallback was provided.")
    return ref


def _sample_base_replay(
    base_dataset: Path,
    replay_size: int,
    ref: Dict[str, np.ndarray],
    rng: np.random.Generator,
    base_sample_weight: float,
):
    data = np.load(str(base_dataset), allow_pickle=True)
    n = data["ctx"].shape[0]
    if n == 0 or replay_size <= 0:
        return None

    m = min(int(replay_size), n)
    idx = rng.choice(n, size=m, replace=False)

    b_ctx_mean, b_ctx_std = _get_norm_from_npz(data, "ctx")
    b_sit_mean, b_sit_std = _get_norm_from_npz(data, "sit")
    b_dyn_mean, b_dyn_std = _get_norm_from_npz(data, "dyn")
    b_goal_mean, b_goal_std = _get_norm_from_npz(data, "goal")
    b_u_mean, b_u_std = _get_norm_from_npz(data, "u")
    b_next_mean, b_next_std = _get_norm_from_npz(data, "next")

    ctx_raw = _denorm_np(data["ctx"][idx], b_ctx_mean, b_ctx_std, is_ctx=True)
    sit_raw = _denorm_np(data["sit"][idx], b_sit_mean, b_sit_std)
    dyn_raw = _denorm_np(data["dyn"][idx], b_dyn_mean, b_dyn_std)
    goal_raw = _denorm_np(data["goal"][idx], b_goal_mean, b_goal_std)
    u_raw = _denorm_np(data["u"][idx], b_u_mean, b_u_std)
    next_raw = _denorm_np(data["next_local"][idx], b_next_mean, b_next_std)

    if dyn_raw.shape[1] != ref["dyn_mean"].shape[0]:
        raise ValueError(
            f"Base dataset dyn_dim={dyn_raw.shape[1]} but model dyn_dim={ref['dyn_mean'].shape[0]}. "
            "Regenerate base dataset/retrain initial model with the same B convention."
        )
    if u_raw.shape[1] != ref["u_mean"].shape[0]:
        raise ValueError(
            f"Base dataset u_dim={u_raw.shape[1]} but model u_dim={ref['u_mean'].shape[0]}."
        )

    out = {
        "ctx": _norm_np(ctx_raw, ref["ctx_mean"], ref["ctx_std"], is_ctx=True),
        "sit": _norm_np(sit_raw, ref["sit_mean"], ref["sit_std"]),
        "dyn": _norm_np(dyn_raw, ref["dyn_mean"], ref["dyn_std"]),
        "goal": _norm_np(goal_raw, ref["goal_mean"], ref["goal_std"]),
        "u": _norm_np(u_raw, ref["u_mean"], ref["u_std"]),
        "next_local": _norm_np(next_raw, ref["next_mean"], ref["next_std"]),
        "traj_id": data["traj_id"][idx].astype(np.int32),
        "dyn_id": data["dyn_id"][idx].astype(np.int32),
        "map_idx": data["map_idx"][idx].astype(np.int32),
        "sample_weight": np.full(m, float(base_sample_weight), dtype=np.float32),
    }
    return out


def _build_dagger_arrays(
    scenarios_by_id: Dict[int, Dict[str, Any]],
    dagger_records: List[Dict[str, Any]],
    max_samples: int,
    ref: Dict[str, np.ndarray],
    args,
    rng: np.random.Generator,
):
    if len(dagger_records) == 0 or max_samples <= 0:
        return None

    if len(dagger_records) > max_samples:
        if args.dagger_priority_sampling:
            weights = np.asarray(
                [max(float(r.get("sample_weight", args.dagger_sample_weight)), 1e-6) for r in dagger_records],
                dtype=np.float64,
            )
            weights = weights / max(float(weights.sum()), 1e-12)
            selected_idx = rng.choice(len(dagger_records), size=int(max_samples), replace=False, p=weights)
        else:
            selected_idx = rng.choice(len(dagger_records), size=int(max_samples), replace=False)
        selected = [dagger_records[i] for i in selected_idx]
    else:
        selected = list(dagger_records)

    samples_ctx, samples_sit, samples_dyn, samples_goal, samples_u, samples_next = [], [], [], [], [], []
    samples_weight = []
    samples_traj_id, samples_dyn_id, samples_map_idx = [], [], []

    L_c = int(args.L_c)
    kept = 0
    for traj_id, res in enumerate(selected):
        states = np.asarray(res["states"], dtype=float)
        inputs = np.asarray(res["inputs"], dtype=float)
        if states.ndim != 2 or len(states) < L_c + 1:
            continue

        sc = scenarios_by_id[int(res["scenario_id"])]
        A, B = ensure_A_B(sc["A_query"], sc["B_query"])
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, B.shape[1])

        rects = [tuple(map(float, r)) for r in sc["rectangles"]]
        goal = np.asarray(sc.get("goal", (0.0, 0.0)), dtype=float)
        start = np.asarray(sc.get("start", (5.0, 5.0)), dtype=float)
        bounds = sc.get("bounds", [-10, -10, 10, 10])

        sit_vec = s1.compute_situation_vector(
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

        ctx = states[:L_c, :2]
        x_next = states[L_c, :2]
        u_tgt = inputs[0].reshape(B.shape[1])

        ctx_l, next_l, goal_l, origin, heading = to_local_frame(ctx, x_next, goal)
        dyn_feat = local_dynamics_features(A, B, heading, origin)

        if dyn_feat.shape[0] != ref["dyn_mean"].shape[0]:
            raise ValueError(f"Dagger dyn_dim={dyn_feat.shape[0]} but model dyn_dim={ref['dyn_mean'].shape[0]}")
        if u_tgt.shape[0] != ref["u_mean"].shape[0]:
            raise ValueError(f"Dagger u_dim={u_tgt.shape[0]} but model u_dim={ref['u_mean'].shape[0]}")

        samples_ctx.append(ctx_l.astype(np.float32))
        samples_sit.append(sit_vec.astype(np.float32))
        samples_dyn.append(dyn_feat.astype(np.float32))
        samples_goal.append(goal_l.astype(np.float32))
        samples_u.append(u_tgt.astype(np.float32))
        samples_next.append(next_l.astype(np.float32))
        samples_weight.append(float(np.clip(res.get("sample_weight", args.dagger_sample_weight), 1e-6, args.max_sample_weight)))
        samples_traj_id.append(1_000_000 + traj_id)
        samples_dyn_id.append(int(sc.get("base_dyn_id", -1)))
        samples_map_idx.append(int(sc.get("map_idx", res["scenario_id"])))
        kept += 1

    if kept == 0:
        return None

    X_ctx = np.stack(samples_ctx).astype(np.float32)
    X_sit = np.stack(samples_sit).astype(np.float32)
    X_dyn = np.stack(samples_dyn).astype(np.float32)
    X_goal = np.stack(samples_goal).astype(np.float32)
    Y_u = np.stack(samples_u).astype(np.float32)
    Y_next = np.stack(samples_next).astype(np.float32)

    out = {
        "ctx": _norm_np(X_ctx, ref["ctx_mean"], ref["ctx_std"], is_ctx=True),
        "sit": _norm_np(X_sit, ref["sit_mean"], ref["sit_std"]),
        "dyn": _norm_np(X_dyn, ref["dyn_mean"], ref["dyn_std"]),
        "goal": _norm_np(X_goal, ref["goal_mean"], ref["goal_std"]),
        "u": _norm_np(Y_u, ref["u_mean"], ref["u_std"]),
        "next_local": _norm_np(Y_next, ref["next_mean"], ref["next_std"]),
        "traj_id": np.asarray(samples_traj_id, dtype=np.int32),
        "dyn_id": np.asarray(samples_dyn_id, dtype=np.int32),
        "map_idx": np.asarray(samples_map_idx, dtype=np.int32),
        "sample_weight": np.asarray(samples_weight, dtype=np.float32),
    }
    return out


def _concat_dataset_parts(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    parts = [p for p in parts if p is not None]
    if not parts:
        raise RuntimeError("No dataset parts to concatenate.")
    return {
        "ctx": np.concatenate([p["ctx"] for p in parts], axis=0).astype(np.float32),
        "sit": np.concatenate([p["sit"] for p in parts], axis=0).astype(np.float32),
        "dyn": np.concatenate([p["dyn"] for p in parts], axis=0).astype(np.float32),
        "goal": np.concatenate([p["goal"] for p in parts], axis=0).astype(np.float32),
        "u": np.concatenate([p["u"] for p in parts], axis=0).astype(np.float32),
        "next_local": np.concatenate([p["next_local"] for p in parts], axis=0).astype(np.float32),
        "traj_id": np.concatenate([p["traj_id"] for p in parts], axis=0).astype(np.int32),
        "dyn_id": np.concatenate([p["dyn_id"] for p in parts], axis=0).astype(np.int32),
        "map_idx": np.concatenate([p["map_idx"] for p in parts], axis=0).astype(np.int32),
        "sample_weight": np.concatenate([p["sample_weight"] for p in parts], axis=0).astype(np.float32),
    }


def count_records_by_type(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        key = str(r.get("record_type", r.get("used_system", "unknown")))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_continual_dataset_with_replay(
    scenarios_by_id: Dict[int, Dict[str, Any]],
    dagger_records: List[Dict[str, Any]],
    out_npz: Path,
    args,
    reference_norm: Dict[str, np.ndarray],
):
    seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(out_npz.name)) % 100000
    rng = np.random.default_rng(int(args.seed) + seed_offset)
    base_path = Path(args.base_dataset)
    if not base_path.exists():
        raise FileNotFoundError(f"Base replay dataset not found: {base_path}")

    base_data = np.load(str(base_path), allow_pickle=True)
    ref = _reference_stats(reference_norm, base_data=base_data)

    base_part = _sample_base_replay(
        base_dataset=base_path,
        replay_size=args.base_replay_size,
        ref=ref,
        rng=rng,
        base_sample_weight=args.base_sample_weight,
    )
    dagger_part = _build_dagger_arrays(
        scenarios_by_id=scenarios_by_id,
        dagger_records=dagger_records,
        max_samples=args.dagger_replay_size,
        ref=ref,
        args=args,
        rng=rng,
    )

    dagger_weight_scale = 1.0
    if base_part is not None and dagger_part is not None and args.max_effective_dagger_fraction < 1.0:
        base_weight_sum = float(np.sum(base_part["sample_weight"]))
        dagger_weight_sum = float(np.sum(dagger_part["sample_weight"]))
        max_frac = max(float(args.max_effective_dagger_fraction), 1e-6)
        target_dagger_sum = (max_frac / max(1.0 - max_frac, 1e-6)) * max(base_weight_sum, 1e-6)
        if dagger_weight_sum > target_dagger_sum:
            dagger_weight_scale = target_dagger_sum / max(dagger_weight_sum, 1e-6)
            dagger_part["sample_weight"] = (dagger_part["sample_weight"] * dagger_weight_scale).astype(np.float32)

    combined = _concat_dataset_parts([base_part, dagger_part])

    n = combined["ctx"].shape[0]
    perm = rng.permutation(n)
    for k in combined:
        combined[k] = combined[k][perm]

    u_dim = combined["u"].shape[1]
    dyn_dim = combined["dyn"].shape[1]
    expected_dyn_dim = expected_dyn_dim_from_u_dim(u_dim)
    if dyn_dim != expected_dyn_dim:
        raise ValueError(f"Combined dyn_dim={dyn_dim}, expected {expected_dyn_dim} for u_dim={u_dim}")

    meta = {
        "n_samples": int(n),
        "n_base_replay": int(base_part["ctx"].shape[0]) if base_part is not None else 0,
        "n_dagger_replay": int(dagger_part["ctx"].shape[0]) if dagger_part is not None else 0,
        "n_correction_records_available": int(len(dagger_records)),
        "n_local_dagger_records_available": int(count_records_by_type(dagger_records).get("S2_dagger", 0)),
        "n_full_s2_records_available": int(count_records_by_type(dagger_records).get("S2_full", 0)),
        "L_c": int(args.L_c),
        "ctx_dim": int(combined["ctx"].shape[-1]),
        "sit_dim": int(combined["sit"].shape[-1]),
        "dyn_dim": int(dyn_dim),
        "goal_dim": int(combined["goal"].shape[-1]),
        "u_dim": int(u_dim),
        "next_dim": int(combined["next_local"].shape[-1]),
        "dyn_feature_layout": "A_local_flat(4), B_local_flat(2*u_dim), drift_local(2)",
        "grid_n": int(args.grid_n),
        "dt_nom": float(args.dt),
        "n_steps_nom": int(args.n_steps_nom),
        "u_max_nom": float(args.u_max),
        "buffer_cells": int(args.buffer_cells),
        "stop_tol": float(args.stop_tol),
        "base_sample_weight": float(args.base_sample_weight),
        "dagger_sample_weight": float(args.dagger_sample_weight),
        "dagger_priority_sampling": bool(args.dagger_priority_sampling),
        "dagger_require_s2_success": bool(args.dagger_require_s2_success),
        "dagger_novelty_weight": float(args.dagger_novelty_weight),
        "collect_full_s2_trajs": bool(args.collect_full_s2_trajs),
        "s2_full_sample_weight": float(args.s2_full_sample_weight),
        "max_effective_dagger_fraction": float(args.max_effective_dagger_fraction),
        "dagger_weight_scale": float(dagger_weight_scale),
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        ctx=combined["ctx"],
        sit=combined["sit"],
        dyn=combined["dyn"],
        goal=combined["goal"],
        u=combined["u"],
        next_local=combined["next_local"],
        traj_id=combined["traj_id"],
        dyn_id=combined["dyn_id"],
        map_idx=combined["map_idx"],
        sample_weight=combined["sample_weight"],
        meta=np.array(meta, dtype=object),
        norm_ctx_mean=ref["ctx_mean"],
        norm_ctx_std=ref["ctx_std"],
        norm_sit_mean=ref["sit_mean"],
        norm_sit_std=ref["sit_std"],
        norm_dyn_mean=ref["dyn_mean"],
        norm_dyn_std=ref["dyn_std"],
        norm_goal_mean=ref["goal_mean"],
        norm_goal_std=ref["goal_std"],
        norm_u_mean=ref["u_mean"],
        norm_u_std=ref["u_std"],
        norm_next_mean=ref["next_mean"],
        norm_next_std=ref["next_std"],
    )

    print(
        f"[dataset] wrote {out_npz} | total={n} | "
        f"base={meta['n_base_replay']} | dagger={meta['n_dagger_replay']} | "
        f"dagger_weight_scale={dagger_weight_scale:.3f} | dyn_dim={dyn_dim} | u_dim={u_dim}"
    )
    return meta


# ============================================================
# Train + behavior gate
# ============================================================

def train_s1_model(dataset_npz: Path, model_out: Path, args, current_model_path: Optional[Path]):
    train_script = Path(args.train_script)
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset", str(dataset_npz),
        "--model_out", str(model_out),
        "--epochs", str(args.train_epochs),
        "--batch", str(args.train_batch),
        "--lr", str(args.train_lr),
        "--val_frac", str(args.val_frac),
        "--lambda_u", str(args.lambda_u),
        "--lambda_next", str(args.lambda_next),
        "--lambda_dir", str(args.lambda_dir),
        "--lambda_speed", str(args.lambda_speed),
        "--lambda_progress", str(args.lambda_progress),
        "--lambda_u_range", str(args.lambda_u_range),
        "--loss_type", str(args.train_loss_type),
        "--huber_delta", str(args.huber_delta),
        "--behavior_u_clip", str(args.behavior_u_clip),
        "--speed_loss_min_step", str(args.speed_loss_min_step),
        "--speed_loss_clip", str(args.speed_loss_clip),
        "--progress_fraction", str(args.progress_fraction),
        "--near_goal_boost", str(args.near_goal_boost),
        "--progress_boost", str(args.progress_boost),
        "--max_sample_weight", str(args.max_sample_weight),
    ]
    if current_model_path is not None:
        cmd += ["--init_model", str(current_model_path)]
    if args.train_head_only:
        cmd += ["--train_head_only"]

    print("[train] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate_s1_only_on_scenarios(
    scenarios: List[Dict[str, Any]],
    model,
    norm,
    L_c: int,
    device,
    args,
    global_s1_accept_rate: float = 0.5,
    disable_confidence_switch: bool = True,
) -> Dict[str, float]:
    """Evaluate raw S1 behavior on a probe set.

    This fixes the old bug where run_s1_on_scenario was called without
    global_s1_accept_rate, causing every probe scenario to be counted as failure.
    """
    results = []
    probe_args = copy.copy(args)
    if disable_confidence_switch:
        probe_args.enable_confidence_switch = False

    for sc in scenarios:
        try:
            out = run_s1_on_scenario(
                sc=sc,
                model=model,
                norm=norm,
                L_c=L_c,
                device=device,
                args=probe_args,
                global_s1_accept_rate=global_s1_accept_rate,
            )
            results.append(out)
        except Exception as e:
            results.append({
                "success": False,
                "collision_free": False,
                "goal_reached": False,
                "runtime_sec": 0.0,
            })
            print(f"[probe warn] S1 evaluation failed for scenario {sc.get('scenario_id', '?')}: {e}")

    if len(results) == 0:
        return {
            "s1_accept_rate": 0.0,
            "success_rate": 0.0,
            "collision_free_rate": 0.0,
            "goal_reach_rate": 0.0,
        }

    succ = np.asarray([r["success"] for r in results], dtype=bool)
    cf = np.asarray([r["collision_free"] for r in results], dtype=bool)
    gr = np.asarray([r["goal_reached"] for r in results], dtype=bool)
    return {
        "s1_accept_rate": float(succ.mean()),
        "success_rate": float(succ.mean()),
        "collision_free_rate": float(cf.mean()),
        "goal_reach_rate": float(gr.mean()),
    }


def make_probe_set(
    scenarios: List[Dict[str, Any]],
    current_block: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    probe = []
    if args.probe_include_current_block:
        probe.extend(current_block)

    if args.probe_fixed_size > 0:
        probe.extend(scenarios[: min(args.probe_fixed_size, len(scenarios))])

    if args.probe_recent_size > 0:
        probe.extend(scenarios[max(0, len(scenarios) - int(args.probe_recent_size)):])

    if args.probe_random_seen_size > 0 and len(scenarios) > 0:
        rng = np.random.default_rng(int(args.seed) + 1701)
        m = min(int(args.probe_random_seen_size), len(scenarios))
        idx = rng.choice(len(scenarios), size=m, replace=False)
        probe.extend([scenarios[i] for i in idx])

    seen = set()
    unique = []
    for sc in probe:
        sid = int(sc.get("scenario_id", len(unique)))
        if sid not in seen:
            unique.append(sc)
            seen.add(sid)
    if args.max_probe_eval > 0 and len(unique) > args.max_probe_eval:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(unique), size=args.max_probe_eval, replace=False)
        unique = [unique[i] for i in idx]
    return unique


def maybe_accept_candidate_model(
    current_model_path: Optional[Path],
    candidate_model_path: Path,
    latest_model_path: Path,
    scenarios: List[Dict[str, Any]],
    current_block: List[Dict[str, Any]],
    old_block_s1_accept: float,
    old_block_success: float,
    device,
    args,
):
    """Memory-preserving behavior gate.

    Accept the candidate if it does not catastrophically degrade old memory/safety.
    Do not require monotonic immediate S1_accept improvement every block.
    """
    if not args.behavior_gate:
        shutil.copyfile(candidate_model_path, latest_model_path)
        model, norm, L_c, _ = load_s1_model(latest_model_path, device)
        return True, model, norm, L_c, {"accepted": True, "reason": "gate disabled"}

    probe = make_probe_set(scenarios, current_block, args)
    if len(probe) == 0:
        shutil.copyfile(candidate_model_path, latest_model_path)
        model, norm, L_c, _ = load_s1_model(latest_model_path, device)
        return True, model, norm, L_c, {"accepted": True, "reason": "empty probe"}

    candidate_model, candidate_norm, candidate_L_c, _ = load_s1_model(candidate_model_path, device)
    cand_metrics = evaluate_s1_only_on_scenarios(
        scenarios=probe,
        model=candidate_model,
        norm=candidate_norm,
        L_c=candidate_L_c,
        device=device,
        args=args,
        global_s1_accept_rate=max(float(old_block_s1_accept), 0.5),
        disable_confidence_switch=True,
    )

    if current_model_path is not None:
        old_model, old_norm, old_L_c, _ = load_s1_model(current_model_path, device)
        old_metrics = evaluate_s1_only_on_scenarios(
            scenarios=probe,
            model=old_model,
            norm=old_norm,
            L_c=old_L_c,
            device=device,
            args=args,
            global_s1_accept_rate=max(float(old_block_s1_accept), 0.5),
            disable_confidence_switch=True,
        )
        old_accept = old_metrics["s1_accept_rate"]
        old_success = old_metrics["success_rate"]
        old_cf = old_metrics["collision_free_rate"]
        old_goal = old_metrics["goal_reach_rate"]
    else:
        old_accept = float(old_block_s1_accept)
        old_success = float(old_block_s1_accept)
        old_cf = np.nan
        old_goal = np.nan

    new_accept = cand_metrics["s1_accept_rate"]
    new_success = cand_metrics["success_rate"]
    new_cf = cand_metrics["collision_free_rate"]
    new_goal = cand_metrics["goal_reach_rate"]

    # Memory-preserving gate. By default it does not accept a candidate that
    # lowers raw S1 probe acceptance; this avoids gradual drift from noisy DAgger.
    if args.require_probe_accept_non_decrease:
        accept_not_collapsed = new_accept >= old_accept + args.min_probe_accept_gain
    else:
        accept_not_collapsed = new_accept >= old_accept - args.max_accept_drop
    success_not_collapsed = new_success >= old_success - args.max_success_drop
    cf_not_collapsed = (
        new_cf >= args.min_probe_collision_free
        and (not np.isfinite(old_cf) or new_cf >= old_cf - args.max_collision_free_drop)
    )
    goal_not_collapsed = (
        not np.isfinite(old_goal) or new_goal >= old_goal - args.max_goal_drop
    )

    accepted = bool(accept_not_collapsed and success_not_collapsed and cf_not_collapsed and goal_not_collapsed)

    failed_checks = []
    if not accept_not_collapsed:
        failed_checks.append("s1_accept")
    if not success_not_collapsed:
        failed_checks.append("success")
    if not cf_not_collapsed:
        failed_checks.append("collision_free")
    if not goal_not_collapsed:
        failed_checks.append("goal")
    reason = "accepted_memory_preserved" if accepted else "rejected_" + "_".join(failed_checks)

    gate_info = {
        "accepted": accepted,
        "old_probe_s1_accept": old_accept,
        "new_probe_s1_accept": new_accept,
        "old_probe_success": old_success,
        "new_probe_success": new_success,
        "old_probe_collision_free": old_cf,
        "new_probe_collision_free": new_cf,
        "old_probe_goal": old_goal,
        "new_probe_goal": new_goal,
        "probe_size": len(probe),
        "reason": reason,
    }

    print(
        "[gate] "
        f"old_accept={100*old_accept:.1f}% | new_accept={100*new_accept:.1f}% | "
        f"old_success={100*old_success:.1f}% | new_success={100*new_success:.1f}% | "
        f"old_cf={100*old_cf:.1f}% | new_cf={100*new_cf:.1f}% | "
        f"old_goal={100*old_goal:.1f}% | new_goal={100*new_goal:.1f}% | "
        f"accepted={accepted}"
    )

    if accepted:
        shutil.copyfile(candidate_model_path, latest_model_path)
        return True, candidate_model, candidate_norm, candidate_L_c, gate_info

    if current_model_path is not None:
        model, norm, L_c, _ = load_s1_model(current_model_path, device)
        return False, model, norm, L_c, gate_info

    return False, None, None, None, gate_info


# ============================================================
# Summary helpers
# ============================================================

def summarize_block(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    succ = np.array([r["success"] for r in results], dtype=bool)
    cf = np.array([r["collision_free"] for r in results], dtype=bool)
    gr = np.array([r["goal_reached"] for r in results], dtype=bool)
    rt = np.array([r["runtime_sec"] for r in results], dtype=float)
    used_s1 = np.array([r.get("used_system") in {"S1", "S1_memory"} for r in results], dtype=bool)
    used_s1_neural = np.array([r.get("used_system") == "S1" for r in results], dtype=bool)
    used_s1_memory = np.array([r.get("used_system") == "S1_memory" for r in results], dtype=bool)
    fallback = np.array([bool(r.get("fallback_to_s2", False)) for r in results], dtype=bool)
    attempted_s1 = np.array(
        [
            (r.get("used_system") in {"S1", "S1_memory"})
            or (r.get("s1_attempt") is not None)
            or (r.get("s1_memory_attempt") is not None)
        for r in results],
        dtype=bool,
    )
    attempted_memory = np.array(
        [
            (r.get("used_system") == "S1_memory")
            or (
                isinstance(r.get("s1_memory_attempt"), dict)
                and bool(r["s1_memory_attempt"].get("attempted", False))
            )
        for r in results],
        dtype=bool,
    )

    s1_attempt_success = []
    s1_failed_switch = []
    memory_scores = []
    memory_map_sims = []
    memory_dyn_dists = []

    combined_conf = []
    local_conf = []
    local_unc = []
    low_conf_switch = []

    for r in results:
        if r.get("used_system") == "S1_memory":
            s1_attempt_success.append(bool(r.get("success", False)))
            s1_failed_switch.append(False)
            memory_scores.append(r.get("episodic_memory_score", np.nan))
            memory_map_sims.append(r.get("episodic_memory_map_similarity", np.nan))
            memory_dyn_dists.append(r.get("episodic_memory_dyn_distance", np.nan))
            low_conf_switch.append(False)
        elif r.get("used_system") == "S1":
            s1_attempt_success.append(bool(r.get("success", False)))
            s1_failed_switch.append(False)
            combined_conf.append(r.get("combined_confidence_mean", np.nan))
            local_conf.append(r.get("local_confidence_mean", np.nan))
            local_unc.append(r.get("local_uncertainty_mean", np.nan))
            low_conf_switch.append(False)
        elif r.get("s1_attempt") is not None:
            a = r["s1_attempt"]
            s1_attempt_success.append(bool(a.get("success", False)))
            s1_failed_switch.append(r.get("switch_reason") == "s1_failed")
            combined_conf.append(a.get("combined_confidence_mean", np.nan))
            local_conf.append(a.get("local_confidence_mean", np.nan))
            local_unc.append(a.get("local_uncertainty_mean", np.nan))
            low_conf_switch.append(bool(a.get("confidence_triggered", False)))
            if isinstance(r.get("s1_memory_attempt"), dict):
                ma = r["s1_memory_attempt"]
                memory_scores.append(ma.get("best_score", np.nan))
                memory_map_sims.append(ma.get("best_map_similarity", np.nan))
                memory_dyn_dists.append(ma.get("best_dyn_distance", np.nan))
        elif isinstance(r.get("s1_memory_attempt"), dict):
            ma = r["s1_memory_attempt"]
            s1_attempt_success.append(bool(ma.get("success", False)))
            s1_failed_switch.append(False)
            memory_scores.append(ma.get("best_score", np.nan))
            memory_map_sims.append(ma.get("best_map_similarity", np.nan))
            memory_dyn_dists.append(ma.get("best_dyn_distance", np.nan))

    def nanmean_safe(x, default=np.nan):
        arr = np.asarray(x, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return default
        return float(np.nanmean(arr))

    return {
        "n": int(n),
        "success_rate": float(succ.mean()),
        "collision_free_rate": float(cf.mean()),
        "goal_reach_rate": float(gr.mean()),
        "avg_runtime_sec": float(rt.mean()),
        "median_runtime_sec": float(np.median(rt)),
        "s1_accept_rate": float(used_s1.mean()),
        "s1_neural_accept_rate": float(used_s1_neural.mean()),
        "s1_memory_accept_rate": float(used_s1_memory.mean()),
        "s1_attempt_rate": float(attempted_s1.mean()),
        "s1_memory_attempt_rate": float(attempted_memory.mean()),
        "s1_raw_success_rate": float(np.sum(s1_attempt_success) / max(1, n)) if s1_attempt_success else 0.0,
        "s1_success_given_attempt_rate": float(np.mean(s1_attempt_success)) if s1_attempt_success else 0.0,
        "fallback_to_s2_rate": float(fallback.mean()),
        "avg_combined_confidence": nanmean_safe(combined_conf),
        "avg_local_confidence": nanmean_safe(local_conf),
        "avg_local_uncertainty": nanmean_safe(local_unc),
        "avg_memory_score": nanmean_safe(memory_scores),
        "avg_memory_map_similarity": nanmean_safe(memory_map_sims),
        "avg_memory_dyn_distance": nanmean_safe(memory_dyn_dists),
        "low_confidence_switch_rate": float(np.mean(low_conf_switch)) if low_conf_switch else 0.0,
        "s1_failed_switch_rate": float(np.mean(s1_failed_switch)) if s1_failed_switch else 0.0,
    }


def append_learning_curve(csv_path: Path, row: Dict[str, Any]):
    exists = csv_path.exists()
    fieldnames = list(row.keys())
    if exists:
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)
        if old_header != fieldnames:
            backup = csv_path.with_name(f"{csv_path.stem}.schema_mismatch_{int(time.time())}{csv_path.suffix}")
            shutil.move(str(csv_path), str(backup))
            print(f"[warn] moved old learning curve with incompatible schema to: {backup}")
            exists = False

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _float_or_nan(x: Any) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def plot_learning_curves(csv_path: Path, out_dir: Path):
    """Write paper/debug plots from learning_curve.csv.

    The script should still run on minimal environments, so plotting is best
    effort. If matplotlib is unavailable, the CSV remains the source of truth.
    """
    if not csv_path.exists():
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: matplotlib unavailable ({e})")
        return

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def col(name: str) -> np.ndarray:
        return np.asarray([_float_or_nan(r.get(name, "")) for r in rows], dtype=float)

    blocks = col("block_id")
    if np.all(np.isnan(blocks)):
        blocks = np.arange(1, len(rows) + 1, dtype=float)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = ax.reshape(-1)

    for name, label in [
        ("block_s1_accept_rate", "S1 accept"),
        ("block_s1_memory_accept_rate", "S1 memory"),
        ("block_s1_neural_accept_rate", "S1 neural"),
        ("block_s1_raw_success_rate", "S1 raw success"),
        ("block_success_rate", "Hybrid success"),
        ("block_collision_free_rate", "Collision-free"),
        ("block_goal_reach_rate", "Goal reach"),
    ]:
        vals = col(name)
        if not np.all(np.isnan(vals)):
            ax[0].plot(blocks, 100.0 * vals, marker="o", label=label)
    ax[0].set_title("Block outcome rates")
    ax[0].set_xlabel("Block")
    ax[0].set_ylabel("Rate (%)")
    ax[0].set_ylim(0, 105)
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(fontsize=8)

    for name, label in [
        ("block_fallback_to_s2_rate", "Fallback to S2"),
        ("block_s1_memory_attempt_rate", "Memory attempted"),
        ("block_low_confidence_switch_rate", "Low confidence"),
        ("block_s1_failed_switch_rate", "S1 failed"),
    ]:
        vals = col(name)
        if not np.all(np.isnan(vals)):
            ax[1].plot(blocks, 100.0 * vals, marker="o", label=label)
    ax[1].set_title("Switching behavior")
    ax[1].set_xlabel("Block")
    ax[1].set_ylabel("Rate (%)")
    ax[1].set_ylim(0, 105)
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(fontsize=8)

    ax[2].plot(blocks, col("block_avg_runtime_sec"), marker="o", color="tab:purple", label="Runtime")
    ax[2].set_title("Average runtime")
    ax[2].set_xlabel("Block")
    ax[2].set_ylabel("Seconds")
    ax[2].grid(True, alpha=0.25)

    dagger_total = col("n_dagger_records_total")
    dagger_used = col("n_dagger_replay_used")
    ax2 = ax[2].twinx()
    ax2.plot(blocks, dagger_total, marker="s", color="tab:orange", alpha=0.7, label="DAgger total")
    ax2.plot(blocks, dagger_used, marker="^", color="tab:brown", alpha=0.7, label="DAgger replay")
    ax2.set_ylabel("Records")
    lines, labels = ax[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[2].legend(lines + lines2, labels + labels2, fontsize=8)

    old_accept = col("old_probe_s1_accept")
    new_accept = col("new_probe_s1_accept")
    update_blocks = np.isfinite(old_accept) & np.isfinite(new_accept)
    if np.any(update_blocks):
        ax[3].plot(blocks[update_blocks], 100.0 * old_accept[update_blocks], marker="o", label="Old probe S1")
        ax[3].plot(blocks[update_blocks], 100.0 * new_accept[update_blocks], marker="o", label="New probe S1")
        ax[3].bar(
            blocks[update_blocks],
            100.0 * (new_accept[update_blocks] - old_accept[update_blocks]),
            alpha=0.25,
            label="New - old",
        )
    ax[3].axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax[3].set_title("Probe update check")
    ax[3].set_xlabel("Block")
    ax[3].set_ylabel("Probe S1 accept (%)")
    ax[3].grid(True, alpha=0.25)
    ax[3].legend(fontsize=8)

    fig.savefig(out_dir / "learning_curve_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    conf = col("block_avg_combined_confidence")
    unc = col("block_avg_local_uncertainty")
    ax.plot(blocks, conf, marker="o", label="Combined confidence")
    ax.set_xlabel("Block")
    ax.set_ylabel("Confidence")
    ax.grid(True, alpha=0.25)
    axb = ax.twinx()
    axb.plot(blocks, unc, marker="s", color="tab:red", alpha=0.75, label="Local uncertainty")
    axb.set_ylabel("Uncertainty")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)
    ax.set_title("Confidence and uncertainty")
    fig.savefig(out_dir / "confidence_uncertainty.png", dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main_dagger_continual():
    p = argparse.ArgumentParser()

    p.add_argument("--scenarios", type=str, required=True)
    p.add_argument("--initial_model", type=str, default="")
    p.add_argument("--allow_s2_only_start", action="store_true",
                   help="Allow running without an initial S1 model. Default is to fail fast because S1_accept is meaningless without S1.")
    p.add_argument("--base_dataset", type=str, default="Solvers/nn_dataset_maze.npz")
    p.add_argument("--workdir", type=str, default="output/continual_s1s2")

    p.add_argument("--batch_size_scenarios", type=int, default=100)
    p.add_argument("--max_scenarios", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--s1_steps", type=int, default=120)
    p.add_argument("--s2_steps", type=int, default=800)
    p.add_argument("--s2_margin", type=float, default=0.35)
    p.add_argument("--s2_gamma", type=float, default=2.0)

    # Confidence-based S1 -> S2 switching.
    p.add_argument("--enable_confidence_switch", action="store_true", default=True)
    p.add_argument("--no_confidence_switch", dest="enable_confidence_switch", action="store_false")
    p.add_argument("--confidence_method", choices=["heuristic", "mc_dropout"], default="heuristic",
                   help="heuristic is one NN forward per step; mc_dropout is slower but estimates variance.")
    p.add_argument("--mc_dropout_samples", type=int, default=8)
    p.add_argument("--local_uncertainty_scale", type=float, default=0.25)
    p.add_argument("--confidence_control_scale", type=float, default=1.0)
    p.add_argument("--confidence_safety_substeps", type=int, default=6)
    p.add_argument("--confidence_global_weight", type=float, default=0.35)
    p.add_argument("--confidence_threshold", type=float, default=0.35) #### need to learn the confidence threshold?
    p.add_argument("--confidence_patience", type=int, default=3)
    p.add_argument("--confidence_min_steps", type=int, default=5)
    p.add_argument("--initial_global_s1_accept_rate", type=float, default=0.5)

    # Episodic System-1 retrieval memory. This is the primary online continual
    # learning mechanism: successful S2 fallback trajectories become retrievable
    # System-1 memories before slower neural consolidation.
    p.add_argument("--enable_episodic_memory", action="store_true", default=True)
    p.add_argument("--no_episodic_memory", dest="enable_episodic_memory", action="store_false")
    p.add_argument("--episodic_memory_path", type=str, default="",
                   help="Optional path for S2 trajectory memory JSON. Default: workdir/episodic_s2_memory.json.")
    p.add_argument("--resume_episodic_memory", action="store_true",
                   help="Load an existing memory file instead of starting from an empty memory.")
    p.add_argument("--episodic_memory_try_before_neural", action="store_true", default=True,
                   help="Try high-confidence retrieval memory before neural S1.")
    p.add_argument("--episodic_memory_try_after_neural", dest="episodic_memory_try_before_neural", action="store_false",
                   help="Try retrieval memory only after neural S1 fails or has low confidence.")
    p.add_argument("--episodic_memory_store_s2_success", action="store_true", default=True,
                   help="Store successful S2 fallback trajectories into episodic S1 memory.")
    p.add_argument("--no_episodic_memory_store_s2_success", dest="episodic_memory_store_s2_success", action="store_false")
    p.add_argument("--use_base_episodic_memory", action="store_true", default=True,
                   help="Initialize episodic memory from the successful motion-primitive trajectories used to build the base neural S1 dataset.")
    p.add_argument("--no_base_episodic_memory", dest="use_base_episodic_memory", action="store_false")
    p.add_argument("--base_memory_traj_npz", type=str, default="",
                   help="Successful trajectory NPZ from make_diverse_training_data_maze.py. Default is inferred from --base_dataset.")
    p.add_argument("--base_memory_scenarios", type=str, default="",
                   help="Scenario JSON matching --base_memory_traj_npz. Default is inferred from --base_dataset.")
    p.add_argument("--base_memory_max_items", type=int, default=0,
                   help="Maximum base motion-primitive trajectories to add to memory; <=0 means all.")
    p.add_argument("--episodic_memory_top_k", type=int, default=5)
    p.add_argument("--episodic_memory_score_threshold", type=float, default=0.65,
                   help="Minimum combined dynamics-map similarity before a memory candidate can be replayed.")
    p.add_argument("--episodic_memory_map_threshold", type=float, default=0.45,
                   help="Minimum map cosine similarity before a memory candidate can be considered.")
    p.add_argument("--episodic_memory_dyn_sigma", type=float, default=0.45,
                   help="Scale for exp(-d_dyn^2/sigma^2) in episodic retrieval.")
    p.add_argument("--episodic_memory_replay_steps", type=int, default=0,
                   help="Max stored controls to replay; <=0 means replay the stored trajectory length.")
    p.add_argument("--episodic_memory_max_traj_steps", type=int, default=0,
                   help="Max S2 controls stored per memory; <=0 means store the full S2 trajectory.")
    p.add_argument("--episodic_memory_max_per_bucket", type=int, default=25)
    p.add_argument("--episodic_memory_max_total", type=int, default=2000)

    p.add_argument("--dagger_stride", type=int, default=4)
    p.add_argument("--dagger_s2_steps", type=int, default=2)
    p.add_argument("--max_dagger_records", type=int, default=30000)
    p.add_argument("--dagger_replay_size", type=int, default=3000)
    p.add_argument("--base_replay_size", type=int, default=50000)
    p.add_argument("--max_effective_dagger_fraction", type=float, default=0.35,
                   help="Cap DAgger's total loss weight fraction after sample weighting.")
    p.add_argument("--base_sample_weight", type=float, default=1.0,
                   help="Per-sample weight for replayed base data.")
    p.add_argument("--dagger_sample_weight", type=float, default=4.0,
                   help="Per-sample weight for DAgger correction data in continual fine-tuning.")
    p.add_argument("--dagger_min_sample_weight", type=float, default=1.0)
    p.add_argument("--dagger_priority_sampling", action="store_true", default=True,
                   help="Sample DAgger replay with probability proportional to each record's utility weight.")
    p.add_argument("--no_dagger_priority_sampling", dest="dagger_priority_sampling", action="store_false")
    p.add_argument("--dagger_require_s2_success", action="store_true", default=True,
                   help="Only learn from S2 fallback trajectories that solved the scenario.")
    p.add_argument("--no_dagger_require_s2_success", dest="dagger_require_s2_success", action="store_false")
    p.add_argument("--dagger_collect_low_conf", action="store_true", default=True)
    p.add_argument("--no_dagger_collect_low_conf", dest="dagger_collect_low_conf", action="store_false")
    p.add_argument("--dagger_collect_failed_s1", action="store_true", default=True)
    p.add_argument("--no_dagger_collect_failed_s1", dest="dagger_collect_failed_s1", action="store_false")
    p.add_argument("--dagger_novelty_weight", type=float, default=0.75,
                   help="Increase correction weights for hard/novel maps and dynamics.")
    p.add_argument("--dagger_low_conf_bonus", type=float, default=0.35)
    p.add_argument("--dagger_failed_s1_bonus", type=float, default=0.50)
    p.add_argument("--dagger_min_final_dist_improvement", type=float, default=0.2,
                   help="If S2 did not fully solve the scenario, require this final-distance improvement before learning from it.")
    p.add_argument("--dagger_allow_nonprogress", type=float, default=0.35,
                   help="Allow a local expert step to temporarily move this much farther from the goal.")
    p.add_argument("--dagger_max_attempt_uncertainty", type=float, default=float("inf"),
                   help="Reject extremely unstable S1 attempts from the DAgger buffer.")
    p.add_argument("--collect_full_s2_trajs", action="store_true", default=True,
                   help="Distill complete successful S2 fallback trajectories into S1 training records.")
    p.add_argument("--no_collect_full_s2_trajs", dest="collect_full_s2_trajs", action="store_false")
    p.add_argument("--s2_full_require_fallback", action="store_true", default=True,
                   help="Only collect full S2 trajectories when they came from an S1 fallback.")
    p.add_argument("--no_s2_full_require_fallback", dest="s2_full_require_fallback", action="store_false")
    p.add_argument("--s2_full_stride", type=int, default=3,
                   help="Stride along successful S2 trajectories when building full-trajectory distillation records.")
    p.add_argument("--s2_full_max_records_per_traj", type=int, default=12)
    p.add_argument("--s2_full_sample_weight", type=float, default=2.0)
    p.add_argument("--s2_full_novelty_weight", type=float, default=0.5)
    p.add_argument("--s2_full_allow_nonprogress", type=float, default=0.35)
    p.add_argument("--min_correction_records_for_update", type=int, default=150,
                   help="Minimum useful correction records before attempting a continual update.")

    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--u_max", type=float, default=3.0)
    p.add_argument("--goal_tol", type=float, default=0.6)
    p.add_argument("--collision_margin", type=float, default=0.05)

    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument("--n_steps_nom", type=int, default=200)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)

    p.add_argument("--L_c", type=int, default=20)

    p.add_argument("--train_script", type=str, default="Solvers/Base/train_nn_policy.py")
    p.add_argument("--train_epochs", type=int, default=2,
                   help="Small fine-tuning epochs for continual learning.")
    p.add_argument("--train_batch", type=int, default=128)
    p.add_argument("--train_lr", type=float, default=5e-5,
                   help="Small LR preserves previous S1 memory.")
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--lambda_next", type=float, default=1.0)
    p.add_argument("--lambda_dir", type=float, default=0.5)
    p.add_argument("--lambda_speed", type=float, default=0.5)
    p.add_argument("--lambda_progress", type=float, default=0.5)
    p.add_argument("--lambda_u_range", type=float, default=0.1,
                   help="Penalty for denormalized controls outside --behavior_u_clip during neural consolidation.")
    p.add_argument("--train_loss_type", type=str, choices=["mse", "huber"], default="huber")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--behavior_u_clip", type=float, default=3.0,
                   help="Physical control clip used inside behavior losses during neural consolidation.")
    p.add_argument("--speed_loss_min_step", type=float, default=0.2)
    p.add_argument("--speed_loss_clip", type=float, default=50.0)
    p.add_argument("--train_head_only", action="store_true", default=True,
                   help="Freeze encoders and fine-tune only the policy head for conservative consolidation.")
    p.add_argument("--no_train_head_only", dest="train_head_only", action="store_false")
    p.add_argument("--progress_fraction", type=float, default=0.85)
    p.add_argument("--near_goal_boost", type=float, default=2.0)
    p.add_argument("--progress_boost", type=float, default=2.0)
    p.add_argument("--max_sample_weight", type=float, default=25.0)

    # Behavior-level gate for update acceptance.
    p.add_argument("--behavior_gate", action="store_true", default=True)
    p.add_argument("--no_behavior_gate", dest="behavior_gate", action="store_false")
    p.add_argument("--require_probe_accept_non_decrease", action="store_true", default=True,
                   help="Reject candidate updates that lower raw S1 probe acceptance.")
    p.add_argument("--allow_probe_accept_drop", dest="require_probe_accept_non_decrease", action="store_false")
    p.add_argument("--min_probe_accept_gain", type=float, default=0.01,
                   help="Optional required improvement in raw S1 probe acceptance.")
    p.add_argument("--max_accept_drop", type=float, default=0.08,
                   help="Legacy relaxed gate tolerance, only used with --allow_probe_accept_drop.")
    p.add_argument("--max_success_drop", type=float, default=0.05,
                   help="Allow at most this success drop on probe.")
    p.add_argument("--max_collision_free_drop", type=float, default=0.0)
    p.add_argument("--max_goal_drop", type=float, default=0.05)
    p.add_argument("--min_probe_collision_free", type=float, default=0.0)
    p.add_argument("--probe_fixed_size", type=int, default=50)
    p.add_argument("--probe_recent_size", type=int, default=150)
    p.add_argument("--probe_random_seen_size", type=int, default=100)
    p.add_argument("--probe_include_current_block", action="store_true", default=True)
    p.add_argument("--no_probe_include_current_block", dest="probe_include_current_block", action="store_false")
    p.add_argument("--probe_evaluate_old", action="store_true", default=True,
                   help="Kept for CLI compatibility. The gate now always evaluates old and candidate S1 on the same probe when an old model exists.")
    p.add_argument("--max_probe_eval", type=int, default=200)

    # Do not fine-tune after every block unless you really want to.
    p.add_argument("--update_every_blocks", type=int, default=2,
                   help="Only fine-tune S1 every N blocks.")
    p.add_argument("--skip_final_update", action="store_true",
                   help="Do not fine-tune after the final scenario block, since it cannot affect reported performance.")
    p.add_argument("--plot_curves", action="store_true", default=True)
    p.add_argument("--no_plot_curves", dest="plot_curves", action="store_false")

    args = p.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    memory_path = Path(args.episodic_memory_path) if args.episodic_memory_path else workdir / "episodic_s2_memory.json"
    if args.enable_episodic_memory and args.resume_episodic_memory:
        episodic_memory = load_episodic_memory(memory_path)
        print(f"[memory] loaded episodic S1 memory: {memory_path} | items={episodic_memory_size(episodic_memory)}")
    elif args.enable_episodic_memory:
        episodic_memory = empty_episodic_memory()
        print(f"[memory] starting empty episodic S1 memory: {memory_path}")
    else:
        episodic_memory = None
        print("[memory] episodic S1 memory disabled.")
    base_memory_added_initial = add_base_trajectories_to_episodic_memory(episodic_memory, args)
    if args.enable_episodic_memory and episodic_memory is not None:
        save_episodic_memory(episodic_memory, memory_path)

    scenarios = json.loads(Path(args.scenarios).read_text())
    if args.max_scenarios > 0:
        scenarios = scenarios[:args.max_scenarios]

    scenarios_by_id = {int(sc.get("scenario_id", i)): sc for i, sc in enumerate(scenarios)}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    current_model_path: Optional[Path] = None
    if args.initial_model:
        p0 = Path(args.initial_model)
        if p0.exists():
            current_model_path = p0
        else:
            if not args.allow_s2_only_start:
                raise FileNotFoundError(
                    f"Initial S1 model not found: {p0}. "
                    "Train it first or pass an existing --initial_model. "
                    "Use --allow_s2_only_start only if you intentionally want S2-only execution."
                )
            print(f"[warn] initial model not found: {p0}. Starting with S2 only.")
    elif not args.allow_s2_only_start:
        raise ValueError(
            "Missing --initial_model. S1_accept is meaningless in S2-only mode. "
            "Pass an initial S1 model, or use --allow_s2_only_start intentionally."
        )

    model, norm, L_c = None, None, None
    if current_model_path is not None:
        model, norm, L_c, _ = load_s1_model(current_model_path, device)
        print(f"[init] loaded S1 model: {current_model_path}")
    else:
        print("[init] no S1 model loaded. First block will use S2 only.")

    all_results: List[Dict[str, Any]] = []
    dagger_records: List[Dict[str, Any]] = []
    learning_curve_csv = workdir / "learning_curve.csv"

    n = len(scenarios)
    Bsz = int(args.batch_size_scenarios)

    for block_id, start_idx in enumerate(range(0, n, Bsz), start=1):
        end_idx = min(start_idx + Bsz, n)
        block_scenarios = scenarios[start_idx:end_idx]
        print(f"\n========== BLOCK {block_id}: scenarios {start_idx}:{end_idx} ==========")

        block_results: List[Dict[str, Any]] = []
        block_memory_added = 0

        for j, sc in enumerate(block_scenarios, start=start_idx):
            if len(all_results) > 0:
                global_s1_accept_rate = summarize_block(all_results)["s1_accept_rate"]
            else:
                global_s1_accept_rate = float(args.initial_global_s1_accept_rate)

            out = run_hybrid_on_scenario(
                sc=sc,
                model=model,
                norm=norm,
                L_c=L_c,
                device=device,
                args=args,
                global_s1_accept_rate=global_s1_accept_rate,
                episodic_memory=episodic_memory,
            )

            block_results.append(out)
            all_results.append(out)

            new_records = collect_dagger_records_from_failed_s1(out, scenarios_by_id, args)
            new_records.extend(collect_full_s2_records(out, scenarios_by_id, args))
            dagger_records.extend(new_records)
            if len(dagger_records) > args.max_dagger_records:
                dagger_records = dagger_records[-args.max_dagger_records:]
            if add_s2_success_to_episodic_memory(episodic_memory, sc, out, args, block_id=block_id):
                block_memory_added += 1

            if (j - start_idx + 1) % 10 == 0 or j + 1 == end_idx:
                tmp = summarize_block(block_results)
                record_counts = count_records_by_type(dagger_records)
                print(
                    f"[block {block_id}] {j-start_idx+1}/{len(block_scenarios)} | "
                    f"succ={100*tmp['success_rate']:.1f}% | "
                    f"cf={100*tmp['collision_free_rate']:.1f}% | "
                    f"goal={100*tmp['goal_reach_rate']:.1f}% | "
                    f"S1_accept={100*tmp['s1_accept_rate']:.1f}% | "
                    f"S1_mem={100*tmp['s1_memory_accept_rate']:.1f}% | "
                    f"S1_raw={100*tmp['s1_raw_success_rate']:.1f}% | "
                    f"low_conf={100*tmp['low_confidence_switch_rate']:.1f}% | "
                    f"conf={tmp['avg_combined_confidence']:.3f} | "
                    f"corrections={len(dagger_records)} "
                    f"(local={record_counts.get('S2_dagger', 0)}, full={record_counts.get('S2_full', 0)}) | "
                    f"memory={episodic_memory_size(episodic_memory)}"
                )

        block_summary = summarize_block(block_results)
        total_summary = summarize_block(all_results)

        (workdir / f"results_block_{block_id:03d}.json").write_text(json.dumps(block_results, indent=2))
        (workdir / "results_all.json").write_text(json.dumps(all_results, indent=2))
        if args.enable_episodic_memory and episodic_memory is not None:
            save_episodic_memory(episodic_memory, memory_path)

        block_record_counts = count_records_by_type(dagger_records)
        print(
            f"[summary block {block_id}] "
            f"succ={100*block_summary['success_rate']:.1f}% | "
            f"cf={100*block_summary['collision_free_rate']:.1f}% | "
            f"goal={100*block_summary['goal_reach_rate']:.1f}% | "
            f"S1_accept={100*block_summary['s1_accept_rate']:.1f}% | "
            f"S1_mem={100*block_summary['s1_memory_accept_rate']:.1f}% | "
            f"S1_raw={100*block_summary['s1_raw_success_rate']:.1f}% | "
            f"corrections={len(dagger_records)} "
            f"(local={block_record_counts.get('S2_dagger', 0)}, full={block_record_counts.get('S2_full', 0)}) | "
            f"memory={episodic_memory_size(episodic_memory)} (+{block_memory_added})"
        )

        gate_info = {
            "accepted": False,
            "old_probe_s1_accept": np.nan,
            "new_probe_s1_accept": np.nan,
            "old_probe_success": np.nan,
            "new_probe_success": np.nan,
            "old_probe_collision_free": np.nan,
            "new_probe_collision_free": np.nan,
            "old_probe_goal": np.nan,
            "new_probe_goal": np.nan,
            "probe_size": 0,
            "reason": "not_updated",
        }
        dataset_meta = {"n_base_replay": 0, "n_dagger_replay": 0, "n_samples": 0, "dagger_weight_scale": np.nan}

        should_update = (
            current_model_path is not None
            and norm is not None
            and len(dagger_records) >= int(args.min_correction_records_for_update)
            and int(args.update_every_blocks) > 0
            and block_id % int(args.update_every_blocks) == 0
            and (not args.skip_final_update or end_idx < n)
        )

        if should_update:
            dataset_path = workdir / f"continual_dataset_block_{block_id:03d}.npz"
            candidate_model = workdir / f"s1_policy_control_cnn_candidate_block_{block_id:03d}.pth"
            latest_model = workdir / "s1_policy_control_cnn_latest.pth"

            dataset_meta = build_continual_dataset_with_replay(
                scenarios_by_id=scenarios_by_id,
                dagger_records=dagger_records,
                out_npz=dataset_path,
                args=args,
                reference_norm=norm,
            )

            train_s1_model(dataset_path, candidate_model, args, current_model_path=current_model_path)

            accepted, model, norm, L_c, gate_info = maybe_accept_candidate_model(
                current_model_path=current_model_path,
                candidate_model_path=candidate_model,
                latest_model_path=latest_model,
                scenarios=scenarios[:end_idx],
                current_block=block_scenarios,
                old_block_s1_accept=block_summary["s1_accept_rate"],
                old_block_success=block_summary["success_rate"],
                device=device,
                args=args,
            )
            if accepted:
                current_model_path = latest_model
                print(f"[update] accepted candidate and loaded: {current_model_path}")
            else:
                print(f"[update] rejected candidate; keeping previous S1 model. reason={gate_info.get('reason')}")
        else:
            if current_model_path is None or norm is None:
                print("[update] skipped: no current model.")
            elif len(dagger_records) < int(args.min_correction_records_for_update):
                print(
                    f"[update] skipped: corrections={len(dagger_records)} < "
                    f"min_correction_records_for_update={args.min_correction_records_for_update}"
                )
            else:
                print(f"[update] skipped: update_every_blocks={args.update_every_blocks}; will update on a later block.")

        row = {
            "block_id": block_id,
            "scenario_start": start_idx,
            "scenario_end": end_idx,
            "block_avg_combined_confidence": block_summary["avg_combined_confidence"],
            "block_avg_local_confidence": block_summary["avg_local_confidence"],
            "block_avg_local_uncertainty": block_summary["avg_local_uncertainty"],
            "block_low_confidence_switch_rate": block_summary["low_confidence_switch_rate"],
            "block_success_rate": block_summary["success_rate"],
            "block_collision_free_rate": block_summary["collision_free_rate"],
            "block_goal_reach_rate": block_summary["goal_reach_rate"],
            "block_s1_accept_rate": block_summary["s1_accept_rate"],
            "block_s1_neural_accept_rate": block_summary["s1_neural_accept_rate"],
            "block_s1_memory_accept_rate": block_summary["s1_memory_accept_rate"],
            "block_s1_attempt_rate": block_summary["s1_attempt_rate"],
            "block_s1_memory_attempt_rate": block_summary["s1_memory_attempt_rate"],
            "block_s1_raw_success_rate": block_summary["s1_raw_success_rate"],
            "block_s1_success_given_attempt_rate": block_summary["s1_success_given_attempt_rate"],
            "block_s1_failed_switch_rate": block_summary["s1_failed_switch_rate"],
            "block_fallback_to_s2_rate": block_summary["fallback_to_s2_rate"],
            "block_avg_runtime_sec": block_summary["avg_runtime_sec"],
            "block_avg_memory_score": block_summary["avg_memory_score"],
            "block_avg_memory_map_similarity": block_summary["avg_memory_map_similarity"],
            "block_avg_memory_dyn_distance": block_summary["avg_memory_dyn_distance"],
            "total_success_rate": total_summary["success_rate"],
            "total_collision_free_rate": total_summary["collision_free_rate"],
            "total_goal_reach_rate": total_summary["goal_reach_rate"],
            "total_s1_accept_rate": total_summary["s1_accept_rate"],
            "total_s1_neural_accept_rate": total_summary["s1_neural_accept_rate"],
            "total_s1_memory_accept_rate": total_summary["s1_memory_accept_rate"],
            "total_s1_attempt_rate": total_summary["s1_attempt_rate"],
            "total_s1_memory_attempt_rate": total_summary["s1_memory_attempt_rate"],
            "total_s1_raw_success_rate": total_summary["s1_raw_success_rate"],
            "total_s1_success_given_attempt_rate": total_summary["s1_success_given_attempt_rate"],
            "episodic_memory_size": episodic_memory_size(episodic_memory),
            "episodic_memory_base_items_initial": base_memory_added_initial,
            "episodic_memory_added_block": block_memory_added,
            "n_dagger_records_total": len(dagger_records),
            "n_local_dagger_records_total": block_record_counts.get("S2_dagger", 0),
            "n_full_s2_records_total": block_record_counts.get("S2_full", 0),
            "n_base_replay_used": dataset_meta.get("n_base_replay", 0),
            "n_dagger_replay_used": dataset_meta.get("n_dagger_replay", 0),
            "dagger_weight_scale": dataset_meta.get("dagger_weight_scale", np.nan),
            "update_accepted": gate_info.get("accepted", False),
            "update_reason": gate_info.get("reason", ""),
            "old_probe_s1_accept": gate_info.get("old_probe_s1_accept", np.nan),
            "new_probe_s1_accept": gate_info.get("new_probe_s1_accept", np.nan),
            "old_probe_success": gate_info.get("old_probe_success", np.nan),
            "new_probe_success": gate_info.get("new_probe_success", np.nan),
            "old_probe_collision_free": gate_info.get("old_probe_collision_free", np.nan),
            "new_probe_collision_free": gate_info.get("new_probe_collision_free", np.nan),
            "old_probe_goal": gate_info.get("old_probe_goal", np.nan),
            "new_probe_goal": gate_info.get("new_probe_goal", np.nan),
            "probe_size": gate_info.get("probe_size", 0),
        }
        append_learning_curve(learning_curve_csv, row)
        if args.plot_curves:
            plot_learning_curves(learning_curve_csv, workdir)

    print("\n========== DONE ==========")
    print(f"[ok] all results: {workdir / 'results_all.json'}")
    print(f"[ok] learning curve: {learning_curve_csv}")
    if args.enable_episodic_memory:
        print(f"[ok] episodic S1 memory: {memory_path} | items={episodic_memory_size(episodic_memory)}")
    if args.plot_curves:
        print(f"[ok] plots: {workdir / 'learning_curve_overview.png'}")
        print(f"[ok] plots: {workdir / 'confidence_uncertainty.png'}")


# ============================================================
# Full-retrain CLI path helpers
# ============================================================

LEGACY_CODES_ROOT = Path(os.environ.get("SOFAI_LEGACY_CODES_ROOT", "/Users/apple/Desktop/S1:2 codes")).expanduser()


def resolve_full_retrain_path(path_like: str, *, required: bool = False) -> Path:
    """Resolve old maze/n_s1 paths when this script is run from mpc-sofai."""
    p = Path(path_like).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([
            ROOT_DIR / p,
            PARENT_DIR / p,
            THIS_DIR / p,
            LEGACY_CODES_ROOT / p,
        ])
        raw = str(p)
        if raw.startswith("maze/n_s1/"):
            suffix = raw[len("maze/n_s1/"):]
            candidates.extend([PARENT_DIR / suffix, THIS_DIR / suffix])
        elif raw.startswith("maze/bck/"):
            suffix = raw[len("maze/bck/"):]
            candidates.extend([ROOT_DIR / "input" / suffix])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Could not resolve path: {path_like}\nTried:\n  {tried}")
    return candidates[0]


def resolve_full_retrain_output_path(path_like: str) -> Path:
    """Resolve relative output paths under the mpc-sofai instance root."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def resolve_full_retrain_inputs(args):
    args.scenarios = str(resolve_full_retrain_path(args.scenarios, required=True))
    args.initial_model = str(resolve_full_retrain_path(args.initial_model, required=True))
    args.base_dataset = str(resolve_full_retrain_path(args.base_dataset, required=True))
    args.train_script = str(resolve_full_retrain_path(args.train_script, required=True))
    args.workdir = str(resolve_full_retrain_output_path(args.workdir))
    if getattr(args, "episodic_memory_path", ""):
        args.episodic_memory_path = str(resolve_full_retrain_output_path(args.episodic_memory_path))
    if args.base_memory_traj_npz:
        args.base_memory_traj_npz = str(resolve_full_retrain_path(args.base_memory_traj_npz, required=True))
    if args.base_memory_scenarios:
        args.base_memory_scenarios = str(resolve_full_retrain_path(args.base_memory_scenarios, required=True))
    return args


# ============================================================
# Full neural retrain from successful S2 trajectories only
# Merged from continual_full_retrain.py
# ============================================================

def append_csv_row(csv_path: Path, row: Dict[str, Any]):
    exists = csv_path.exists()
    fieldnames = list(row.keys())
    if exists:
        with csv_path.open("r", newline="") as f:
            old_header = next(csv.reader(f), None)
        if old_header != fieldnames:
            backup = csv_path.with_name(f"{csv_path.stem}.schema_mismatch_{int(time.time())}{csv_path.suffix}")
            shutil.move(str(csv_path), str(backup))
            print(f"[warn] moved old CSV with incompatible schema to: {backup}")
            exists = False

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_full_retrain_curves(csv_path: Path, out_dir: Path):
    if not csv_path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: matplotlib unavailable ({e})")
        return

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def col(name: str) -> np.ndarray:
        return np.asarray([_float_or_nan(r.get(name, "")) for r in rows], dtype=float)

    blocks = col("block_id")
    if np.all(np.isnan(blocks)):
        blocks = np.arange(1, len(rows) + 1, dtype=float)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = ax.reshape(-1)

    for name, label in [
        ("block_success_rate", "Hybrid success"),
        ("block_s1_accept_rate", "S1 accept"),
        ("block_s1_memory_accept_rate", "S1 memory"),
        ("block_s1_neural_accept_rate", "S1 neural"),
    ]:
        vals = col(name)
        if not np.all(np.isnan(vals)):
            ax[0].plot(blocks, 100.0 * vals, marker="o", label=label)
    ax[0].set_title("Outcome Rates")
    ax[0].set_xlabel("Block")
    ax[0].set_ylabel("Rate (%)")
    ax[0].set_ylim(0, 105)
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(fontsize=8)

    for name, label in [
        ("block_fallback_to_s2_rate", "Fallback to S2"),
        ("block_s1_memory_attempt_rate", "Memory attempted"),
        ("block_low_confidence_switch_rate", "Low confidence"),
        ("block_s1_failed_switch_rate", "Neural S1 failed"),
    ]:
        vals = col(name)
        if not np.all(np.isnan(vals)):
            ax[1].plot(blocks, 100.0 * vals, marker="o", label=label)
    ax[1].set_title("Switching")
    ax[1].set_xlabel("Block")
    ax[1].set_ylabel("Rate (%)")
    ax[1].set_ylim(0, 105)
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(fontsize=8)

    ax[2].plot(blocks, col("block_avg_runtime_sec"), marker="o", color="tab:purple", label="Runtime")
    ax[2].set_title("Average Runtime")
    ax[2].set_xlabel("Block")
    ax[2].set_ylabel("Seconds")
    ax[2].grid(True, alpha=0.25)
    ax2 = ax[2].twinx()
    ax2.plot(blocks, col("n_s2_full_records_total"), marker="s", color="tab:orange", label="S2 full records")
    ax2.plot(blocks, col("n_s2_replay_used"), marker="^", color="tab:brown", label="S2 replay used")
    ax2.set_ylabel("Records")
    lines, labels = ax[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[2].legend(lines + lines2, labels + labels2, fontsize=8)

    old_accept = col("old_probe_s1_accept")
    new_accept = col("new_probe_s1_accept")
    mask = np.isfinite(old_accept) & np.isfinite(new_accept)
    if np.any(mask):
        ax[3].plot(blocks[mask], 100.0 * old_accept[mask], marker="o", label="Old probe S1")
        ax[3].plot(blocks[mask], 100.0 * new_accept[mask], marker="o", label="New probe S1")
        ax[3].bar(blocks[mask], 100.0 * (new_accept[mask] - old_accept[mask]), alpha=0.25, label="New - old")
    ax[3].axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax[3].set_title("Neural Update Gate")
    ax[3].set_xlabel("Block")
    ax[3].set_ylabel("Probe S1 accept (%)")
    ax[3].grid(True, alpha=0.25)
    ax[3].legend(fontsize=8)

    fig.savefig(out_dir / "full_retrain_learning_curve.png", dpi=200)
    plt.close(fig)


def collect_successful_full_s2_records(out: Dict[str, Any], scenarios_by_id: Dict[int, Dict[str, Any]], args):
    """Collect only full successful S2 trajectories.

    No local DAgger states are used here. If S1 fails and S2 solves the task,
    the successful S2 trajectory is sampled into supervised S1 training records.
    """
    records = collect_full_s2_records(out, scenarios_by_id, args)
    return [r for r in records if r.get("record_type") == "S2_full"]


def build_full_retrain_dataset(
    scenarios_by_id: Dict[int, Dict[str, Any]],
    s2_full_records: List[Dict[str, Any]],
    out_npz: Path,
    args,
    reference_norm: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Build base replay + full S2 replay dataset.

    This calls the shared replay builder, but the record buffer contains only
    S2_full records. The builder still uses some historical "dagger" field
    names internally because train_nn_policy.py already consumes that NPZ
    format; this script reports them as S2 replay in its own CSV.
    """
    args.dagger_replay_size = int(args.s2_replay_size)
    args.dagger_sample_weight = float(args.s2_full_sample_weight)
    args.dagger_priority_sampling = bool(args.s2_priority_sampling)
    return build_continual_dataset_with_replay(
        scenarios_by_id=scenarios_by_id,
        dagger_records=s2_full_records,
        out_npz=out_npz,
        args=args,
        reference_norm=reference_norm,
    )


def make_full_retrain_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--scenarios", type=str, required=True)
    p.add_argument("--initial_model", type=str, required=True)
    p.add_argument("--base_dataset", type=str, default="Solvers/nn_dataset_maze_diverse_5k.npz")
    p.add_argument("--workdir", type=str, default="output/continual_full_retrain")
    p.add_argument("--train_script", type=str, default="Solvers/Base/train_nn_policy.py")

    p.add_argument("--batch_size_scenarios", type=int, default=200)
    p.add_argument("--max_scenarios", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # Rollout/control settings.
    p.add_argument("--s1_steps", type=int, default=120)
    p.add_argument("--s2_steps", type=int, default=800)
    p.add_argument("--s2_margin", type=float, default=0.35)
    p.add_argument("--s2_gamma", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--u_max", type=float, default=3.0)
    p.add_argument("--goal_tol", type=float, default=0.6)
    p.add_argument("--collision_margin", type=float, default=0.05)

    # Situation vector settings.
    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument("--n_steps_nom", type=int, default=200)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)
    p.add_argument("--L_c", type=int, default=20)

    # Confidence switching.
    p.add_argument("--enable_confidence_switch", action="store_true", default=True)
    p.add_argument("--no_confidence_switch", dest="enable_confidence_switch", action="store_false")
    p.add_argument("--confidence_method", choices=["heuristic", "mc_dropout"], default="heuristic",
                   help="heuristic is one NN forward per step; mc_dropout is slower but estimates variance.")
    p.add_argument("--mc_dropout_samples", type=int, default=4)
    p.add_argument("--local_uncertainty_scale", type=float, default=0.25)
    p.add_argument("--confidence_control_scale", type=float, default=1.0)
    p.add_argument("--confidence_safety_substeps", type=int, default=6)
    p.add_argument("--confidence_global_weight", type=float, default=0.35)
    p.add_argument("--confidence_threshold", type=float, default=0.35)
    p.add_argument("--confidence_patience", type=int, default=3)
    p.add_argument("--confidence_min_steps", type=int, default=5)
    p.add_argument("--initial_global_s1_accept_rate", type=float, default=0.5)

    # Episodic memory. Enabled by default because this is the part that already works.
    p.add_argument("--enable_episodic_memory", action="store_true", default=True)
    p.add_argument("--no_episodic_memory", dest="enable_episodic_memory", action="store_false")
    p.add_argument("--episodic_memory_path", type=str, default="")
    p.add_argument("--resume_episodic_memory", action="store_true")
    p.add_argument("--episodic_memory_try_before_neural", action="store_true", default=True)
    p.add_argument("--episodic_memory_try_after_neural", dest="episodic_memory_try_before_neural", action="store_false")
    p.add_argument("--episodic_memory_store_s2_success", action="store_true", default=True)
    p.add_argument("--no_episodic_memory_store_s2_success", dest="episodic_memory_store_s2_success", action="store_false")
    p.add_argument("--use_base_episodic_memory", action="store_true", default=True)
    p.add_argument("--no_base_episodic_memory", dest="use_base_episodic_memory", action="store_false")
    p.add_argument("--base_memory_traj_npz", type=str, default="")
    p.add_argument("--base_memory_scenarios", type=str, default="")
    p.add_argument("--base_memory_max_items", type=int, default=0)
    p.add_argument("--episodic_memory_top_k", type=int, default=5)
    p.add_argument("--episodic_memory_score_threshold", type=float, default=0.65)
    p.add_argument("--episodic_memory_map_threshold", type=float, default=0.45)
    p.add_argument("--episodic_memory_dyn_sigma", type=float, default=0.45)
    p.add_argument("--episodic_memory_replay_steps", type=int, default=0)
    p.add_argument("--episodic_memory_max_traj_steps", type=int, default=0)
    p.add_argument("--episodic_memory_max_per_bucket", type=int, default=25)
    p.add_argument("--episodic_memory_max_total", type=int, default=2000)

    # Full S2 trajectory collection. These replace DAgger.
    p.add_argument("--collect_full_s2_trajs", action="store_true", default=True)
    p.add_argument("--no_collect_full_s2_trajs", dest="collect_full_s2_trajs", action="store_false")
    p.add_argument("--s2_full_require_fallback", action="store_true", default=True)
    p.add_argument("--no_s2_full_require_fallback", dest="s2_full_require_fallback", action="store_false")
    p.add_argument("--s2_full_stride", type=int, default=2)
    p.add_argument("--s2_full_max_records_per_traj", type=int, default=24)
    p.add_argument("--s2_full_sample_weight", type=float, default=1.0)
    p.add_argument("--s2_full_novelty_weight", type=float, default=0.25)
    p.add_argument("--s2_full_allow_nonprogress", type=float, default=0.35)
    p.add_argument("--max_s2_full_records", type=int, default=60000)
    p.add_argument("--min_s2_full_records_for_update", type=int, default=200)
    p.add_argument("--s2_replay_size", type=int, default=12000)
    p.add_argument("--s2_priority_sampling", action="store_true", default=True)
    p.add_argument("--no_s2_priority_sampling", dest="s2_priority_sampling", action="store_false")

    # Base replay and sample weights.
    p.add_argument("--base_replay_size", type=int, default=5000)
    p.add_argument("--base_sample_weight", type=float, default=1.0)
    p.add_argument("--dagger_min_sample_weight", type=float, default=1.0)
    p.add_argument("--dagger_require_s2_success", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("--dagger_novelty_weight", type=float, default=0.0,
                   help=argparse.SUPPRESS)
    p.add_argument("--max_effective_dagger_fraction", type=float, default=0.35,
                   help="Historical name: caps total S2 replay loss fraction.")
    p.add_argument("--max_sample_weight", type=float, default=5.0)

    # Whole-model neural retraining/consolidation.
    p.add_argument("--update_every_blocks", type=int, default=1)
    p.add_argument("--skip_final_update", action="store_true")
    p.add_argument("--train_epochs", type=int, default=8)
    p.add_argument("--train_batch", type=int, default=128)
    p.add_argument("--train_lr", type=float, default=5e-6)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--lambda_next", type=float, default=0.1)
    p.add_argument("--lambda_dir", type=float, default=0.25)
    p.add_argument("--lambda_speed", type=float, default=0.0)
    p.add_argument("--lambda_progress", type=float, default=0.0)
    p.add_argument("--lambda_u_range", type=float, default=1.0)
    p.add_argument("--train_loss_type", type=str, choices=["mse", "huber"], default="huber")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--behavior_u_clip", type=float, default=3.0)
    p.add_argument("--speed_loss_min_step", type=float, default=0.2)
    p.add_argument("--speed_loss_clip", type=float, default=50.0)
    p.add_argument("--progress_fraction", type=float, default=0.5)
    p.add_argument("--near_goal_boost", type=float, default=1.0)
    p.add_argument("--progress_boost", type=float, default=1.0)
    p.add_argument("--train_head_only", action="store_true", default=False,
                   help="Off by default: this script is for whole-model retraining.")

    # Behavior gate.
    p.add_argument("--behavior_gate", action="store_true", default=True)
    p.add_argument("--no_behavior_gate", dest="behavior_gate", action="store_false")
    p.add_argument("--require_probe_accept_non_decrease", action="store_true", default=True)
    p.add_argument("--allow_probe_accept_drop", dest="require_probe_accept_non_decrease", action="store_false")
    p.add_argument("--min_probe_accept_gain", type=float, default=0.005)
    p.add_argument("--max_accept_drop", type=float, default=0.05)
    p.add_argument("--max_success_drop", type=float, default=0.03)
    p.add_argument("--max_collision_free_drop", type=float, default=0.0)
    p.add_argument("--max_goal_drop", type=float, default=0.03)
    p.add_argument("--min_probe_collision_free", type=float, default=0.0)
    p.add_argument("--probe_fixed_size", type=int, default=200)
    p.add_argument("--probe_recent_size", type=int, default=200)
    p.add_argument("--probe_random_seen_size", type=int, default=200)
    p.add_argument("--probe_include_current_block", action="store_true", default=True)
    p.add_argument("--no_probe_include_current_block", dest="probe_include_current_block", action="store_false")
    p.add_argument("--probe_evaluate_old", action="store_true", default=True)
    p.add_argument("--max_probe_eval", type=int, default=300)

    p.add_argument("--plot_curves", action="store_true", default=True)
    p.add_argument("--no_plot_curves", dest="plot_curves", action="store_false")
    return p


def main_full_retrain():
    args = make_full_retrain_parser().parse_args()
    args = resolve_full_retrain_inputs(args)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    learning_curve_csv = workdir / "learning_curve.csv"

    scenarios = json.loads(Path(args.scenarios).read_text())
    if args.max_scenarios > 0:
        scenarios = scenarios[:args.max_scenarios]
    scenarios_by_id = {int(sc.get("scenario_id", i)): sc for i, sc in enumerate(scenarios)}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    current_model_path = Path(args.initial_model)
    if not current_model_path.exists():
        raise FileNotFoundError(f"Initial S1 model not found: {current_model_path}")

    model, norm, L_c, _ = load_s1_model(current_model_path, device)
    print(f"[init] loaded S1 model: {current_model_path}")
    print("[mode] full neural retrain from base replay + successful full S2 trajectories only; no local DAgger.")

    memory_path = Path(args.episodic_memory_path) if args.episodic_memory_path else workdir / "episodic_s2_memory.json"
    if args.enable_episodic_memory and args.resume_episodic_memory:
        episodic_memory = load_episodic_memory(memory_path)
        print(f"[memory] loaded episodic S1 memory: {memory_path} | items={episodic_memory_size(episodic_memory)}")
    elif args.enable_episodic_memory:
        episodic_memory = empty_episodic_memory()
        print(f"[memory] starting empty episodic S1 memory: {memory_path}")
    else:
        episodic_memory = None
        print("[memory] episodic S1 memory disabled.")

    base_memory_added_initial = add_base_trajectories_to_episodic_memory(episodic_memory, args)
    if args.enable_episodic_memory and episodic_memory is not None:
        save_episodic_memory(episodic_memory, memory_path)

    all_results: List[Dict[str, Any]] = []
    s2_full_records: List[Dict[str, Any]] = []

    n = len(scenarios)
    Bsz = int(args.batch_size_scenarios)

    for block_id, start_idx in enumerate(range(0, n, Bsz), start=1):
        end_idx = min(start_idx + Bsz, n)
        block_scenarios = scenarios[start_idx:end_idx]
        print(f"\n========== FULL-RETRAIN BLOCK {block_id}: scenarios {start_idx}:{end_idx} ==========")

        block_results: List[Dict[str, Any]] = []
        block_memory_added = 0
        block_s2_full_added = 0

        for j, sc in enumerate(block_scenarios, start=start_idx):
            if all_results:
                global_s1_accept_rate = summarize_block(all_results)["s1_accept_rate"]
            else:
                global_s1_accept_rate = float(args.initial_global_s1_accept_rate)

            out = run_hybrid_on_scenario(
                sc=sc,
                model=model,
                norm=norm,
                L_c=L_c,
                device=device,
                args=args,
                global_s1_accept_rate=global_s1_accept_rate,
                episodic_memory=episodic_memory,
            )

            block_results.append(out)
            all_results.append(out)

            new_full = collect_successful_full_s2_records(out, scenarios_by_id, args)
            s2_full_records.extend(new_full)
            block_s2_full_added += len(new_full)
            if len(s2_full_records) > int(args.max_s2_full_records):
                s2_full_records = s2_full_records[-int(args.max_s2_full_records):]

            if add_s2_success_to_episodic_memory(episodic_memory, sc, out, args, block_id=block_id):
                block_memory_added += 1

            if (j - start_idx + 1) % 10 == 0 or j + 1 == end_idx:
                tmp = summarize_block(block_results)
                print(
                    f"[block {block_id}] {j-start_idx+1}/{len(block_scenarios)} | "
                    f"succ={100*tmp['success_rate']:.1f}% | "
                    f"cf={100*tmp['collision_free_rate']:.1f}% | "
                    f"goal={100*tmp['goal_reach_rate']:.1f}% | "
                    f"S1_accept={100*tmp['s1_accept_rate']:.1f}% | "
                    f"S1_mem={100*tmp['s1_memory_accept_rate']:.1f}% | "
                    f"S1_neural={100*tmp['s1_neural_accept_rate']:.1f}% | "
                    f"fallback={100*tmp['fallback_to_s2_rate']:.1f}% | "
                    f"S2_full_records={len(s2_full_records)} | "
                    f"memory={episodic_memory_size(episodic_memory)}"
                )

        block_summary = summarize_block(block_results)
        total_summary = summarize_block(all_results)

        (workdir / f"results_block_{block_id:03d}.json").write_text(json.dumps(block_results, indent=2))
        (workdir / "results_all.json").write_text(json.dumps(all_results, indent=2))
        if args.enable_episodic_memory and episodic_memory is not None:
            save_episodic_memory(episodic_memory, memory_path)

        print(
            f"[summary block {block_id}] "
            f"succ={100*block_summary['success_rate']:.1f}% | "
            f"cf={100*block_summary['collision_free_rate']:.1f}% | "
            f"goal={100*block_summary['goal_reach_rate']:.1f}% | "
            f"S1_accept={100*block_summary['s1_accept_rate']:.1f}% | "
            f"S1_mem={100*block_summary['s1_memory_accept_rate']:.1f}% | "
            f"S1_neural={100*block_summary['s1_neural_accept_rate']:.1f}% | "
            f"S2_full_records={len(s2_full_records)} (+{block_s2_full_added}) | "
            f"memory={episodic_memory_size(episodic_memory)} (+{block_memory_added})"
        )

        gate_info = {
            "accepted": False,
            "reason": "not_updated",
            "old_probe_s1_accept": np.nan,
            "new_probe_s1_accept": np.nan,
            "old_probe_success": np.nan,
            "new_probe_success": np.nan,
            "old_probe_collision_free": np.nan,
            "new_probe_collision_free": np.nan,
            "old_probe_goal": np.nan,
            "new_probe_goal": np.nan,
            "probe_size": 0,
        }
        dataset_meta = {
            "n_base_replay": 0,
            "n_dagger_replay": 0,
            "n_samples": 0,
            "dagger_weight_scale": np.nan,
        }

        should_update = (
            current_model_path is not None
            and norm is not None
            and len(s2_full_records) >= int(args.min_s2_full_records_for_update)
            and int(args.update_every_blocks) > 0
            and block_id % int(args.update_every_blocks) == 0
            and (not args.skip_final_update or end_idx < n)
        )

        if should_update:
            dataset_path = workdir / f"full_retrain_dataset_block_{block_id:03d}.npz"
            candidate_model = workdir / f"s1_policy_full_retrain_candidate_block_{block_id:03d}.pth"
            latest_model = workdir / "s1_policy_full_retrain_latest.pth"

            dataset_meta = build_full_retrain_dataset(
                scenarios_by_id=scenarios_by_id,
                s2_full_records=s2_full_records,
                out_npz=dataset_path,
                args=args,
                reference_norm=norm,
            )

            train_s1_model(dataset_path, candidate_model, args, current_model_path=current_model_path)

            accepted, model, norm, L_c, gate_info = maybe_accept_candidate_model(
                current_model_path=current_model_path,
                candidate_model_path=candidate_model,
                latest_model_path=latest_model,
                scenarios=scenarios[:end_idx],
                current_block=block_scenarios,
                old_block_s1_accept=block_summary["s1_accept_rate"],
                old_block_success=block_summary["success_rate"],
                device=device,
                args=args,
            )
            if accepted:
                current_model_path = latest_model
                print(f"[update] accepted full neural retrain candidate and loaded: {current_model_path}")
            else:
                print(f"[update] rejected full neural retrain candidate; keeping previous S1. reason={gate_info.get('reason')}")
        else:
            if len(s2_full_records) < int(args.min_s2_full_records_for_update):
                print(
                    f"[update] skipped: S2_full_records={len(s2_full_records)} < "
                    f"min_s2_full_records_for_update={args.min_s2_full_records_for_update}"
                )
            else:
                print(f"[update] skipped: update_every_blocks={args.update_every_blocks}; will update later.")

        row = {
            "block_id": block_id,
            "scenario_start": start_idx,
            "scenario_end": end_idx,
            "block_success_rate": block_summary["success_rate"],
            "block_collision_free_rate": block_summary["collision_free_rate"],
            "block_goal_reach_rate": block_summary["goal_reach_rate"],
            "block_s1_accept_rate": block_summary["s1_accept_rate"],
            "block_s1_neural_accept_rate": block_summary["s1_neural_accept_rate"],
            "block_s1_memory_accept_rate": block_summary["s1_memory_accept_rate"],
            "block_s1_attempt_rate": block_summary["s1_attempt_rate"],
            "block_s1_memory_attempt_rate": block_summary["s1_memory_attempt_rate"],
            "block_s1_raw_success_rate": block_summary["s1_raw_success_rate"],
            "block_s1_failed_switch_rate": block_summary["s1_failed_switch_rate"],
            "block_low_confidence_switch_rate": block_summary["low_confidence_switch_rate"],
            "block_fallback_to_s2_rate": block_summary["fallback_to_s2_rate"],
            "block_avg_runtime_sec": block_summary["avg_runtime_sec"],
            "block_avg_memory_score": block_summary["avg_memory_score"],
            "block_avg_memory_map_similarity": block_summary["avg_memory_map_similarity"],
            "block_avg_memory_dyn_distance": block_summary["avg_memory_dyn_distance"],
            "total_success_rate": total_summary["success_rate"],
            "total_collision_free_rate": total_summary["collision_free_rate"],
            "total_goal_reach_rate": total_summary["goal_reach_rate"],
            "total_s1_accept_rate": total_summary["s1_accept_rate"],
            "total_s1_neural_accept_rate": total_summary["s1_neural_accept_rate"],
            "total_s1_memory_accept_rate": total_summary["s1_memory_accept_rate"],
            "total_s1_raw_success_rate": total_summary["s1_raw_success_rate"],
            "episodic_memory_size": episodic_memory_size(episodic_memory),
            "episodic_memory_base_items_initial": base_memory_added_initial,
            "episodic_memory_added_block": block_memory_added,
            "n_s2_full_records_total": len(s2_full_records),
            "n_s2_full_records_added_block": block_s2_full_added,
            "n_base_replay_used": dataset_meta.get("n_base_replay", 0),
            "n_s2_replay_used": dataset_meta.get("n_dagger_replay", 0),
            "s2_weight_scale": dataset_meta.get("dagger_weight_scale", np.nan),
            "update_accepted": gate_info.get("accepted", False),
            "update_reason": gate_info.get("reason", ""),
            "old_probe_s1_accept": gate_info.get("old_probe_s1_accept", np.nan),
            "new_probe_s1_accept": gate_info.get("new_probe_s1_accept", np.nan),
            "old_probe_success": gate_info.get("old_probe_success", np.nan),
            "new_probe_success": gate_info.get("new_probe_success", np.nan),
            "old_probe_collision_free": gate_info.get("old_probe_collision_free", np.nan),
            "new_probe_collision_free": gate_info.get("new_probe_collision_free", np.nan),
            "old_probe_goal": gate_info.get("old_probe_goal", np.nan),
            "new_probe_goal": gate_info.get("new_probe_goal", np.nan),
            "probe_size": gate_info.get("probe_size", 0),
        }
        append_csv_row(learning_curve_csv, row)
        if args.plot_curves:
            plot_full_retrain_curves(learning_curve_csv, workdir)

    print("\n========== DONE ==========")
    print(f"[ok] all results: {workdir / 'results_all.json'}")
    print(f"[ok] learning curve: {learning_curve_csv}")
    if args.enable_episodic_memory:
        print(f"[ok] episodic S1 memory: {memory_path} | items={episodic_memory_size(episodic_memory)}")
    if args.plot_curves:
        print(f"[ok] plot: {workdir / 'full_retrain_learning_curve.png'}")



def main():
    main_full_retrain()


if __name__ == "__main__":
    main()
