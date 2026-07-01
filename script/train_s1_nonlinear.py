#!/usr/bin/env python3
"""Train the nonlinear neural System 1 from successful benchmark trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def configure_repo(root: Path, mplconfigdir: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", mplconfigdir)
    for path in (root, root / "sofai", root / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--dictionary", required=True)
    p.add_argument("--results_jsonl", nargs="+", required=True)
    p.add_argument("--out_model", required=True)
    p.add_argument("--out_dataset", default="")
    p.add_argument("--init_model", default="")
    p.add_argument("--source", choices=["s2", "selected", "all_success"], default="all_success")
    p.add_argument("--max_trajectories", type=int, default=200)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument("--dt_nom", type=float, default=0.05)
    p.add_argument("--n_steps_nom", type=int, default=200)
    p.add_argument("--u_max_nom", type=float, default=3.0)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--cnn_channels", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--lambda_next", type=float, default=1.0)
    p.add_argument("--lambda_dir", type=float, default=0.25)
    p.add_argument("--lambda_speed", type=float, default=0.25)
    p.add_argument("--lambda_progress", type=float, default=0.5)
    p.add_argument("--progress_fraction", type=float, default=0.9)
    p.add_argument("--lambda_u_range", type=float, default=0.0)
    p.add_argument("--loss_type", choices=["mse", "huber"], default="huber")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--train_head_only", action="store_true")
    p.add_argument("--near_goal_boost", type=float, default=2.0)
    p.add_argument("--progress_boost", type=float, default=2.0)
    p.add_argument("--max_sample_weight", type=float, default=25.0)
    p.add_argument("--mplconfigdir", default="/private/tmp/mpl")
    return p.parse_args()


def load_results(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def select_training_attempts(result: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    attempts = result.get("attempts", [])
    if not attempts:
        return []

    if source == "s2":
        if str(result.get("run_type")) == "s2":
            return [attempts[0]] if bool(attempts[0].get("success")) else []
        for attempt in attempts:
            if str(attempt.get("system")) == "s2" and bool(attempt.get("success")):
                return [attempt]
        return []

    if source == "all_success":
        return [attempt for attempt in attempts if bool(attempt.get("success"))]

    selected_name = str(result.get("selected_attempt") or "").strip()
    if selected_name:
        for attempt in attempts:
            if str(attempt.get("name")) == selected_name and bool(attempt.get("success")):
                return [attempt]

    for attempt in attempts:
        if bool(attempt.get("success")):
            return [attempt]
    return []


def _xy(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32)


def make_context_window(xy_prefix: np.ndarray, L_c: int) -> np.ndarray:
    xy = np.asarray(xy_prefix, dtype=np.float32)
    if xy.shape[0] == 0:
        return np.zeros((L_c, 2), dtype=np.float32)
    if xy.shape[0] >= L_c:
        return xy[-L_c:]

    first = xy[0]
    if xy.shape[0] >= 2:
        step = xy[1] - xy[0]
    else:
        step = np.array([0.5, 0.0], dtype=np.float32)
    pad = []
    missing = L_c - xy.shape[0]
    for i in range(missing):
        pad.append(first - step * float(missing - i))
    return np.vstack([np.asarray(pad, dtype=np.float32), xy]).astype(np.float32)


def build_samples(
    result_rows: List[Dict[str, Any]],
    *,
    root: Path,
    default_dictionary: Path,
    source: str,
    max_trajectories: int,
    L_c: int,
    dt_nom: float,
    grid_n: int,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
):
    from input.input_handler import load_scenarios
    from solvers import s1_nonlinear as nl

    scenario_cache: Dict[str, Dict[int, Any]] = {}

    def load_by_id(dictionary_path: Path) -> Dict[int, Any]:
        key = str(dictionary_path.resolve())
        if key not in scenario_cache:
            scenarios = load_scenarios(str(dictionary_path))
            scenario_cache[key] = {int(getattr(sc, "scenario_id", i)): sc for i, sc in enumerate(scenarios)}
        return scenario_cache[key]

    default_by_id = load_by_id(default_dictionary)

    def resolve_row_dictionary(row_value: Any) -> Path:
        candidate = Path(str(row_value)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for path in (
            root / candidate,
            root / "input" / candidate,
            root / "input" / "nl" / candidate,
        ):
            if path.exists():
                return path
        return default_dictionary

    ctx_list: List[np.ndarray] = []
    sit_list: List[np.ndarray] = []
    dyn_list: List[np.ndarray] = []
    goal_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    next_list: List[np.ndarray] = []
    traj_ids: List[int] = []

    traj_count = 0
    max_traj = int(max_trajectories)
    unlimited = max_traj <= 0
    for row in result_rows:
        if not unlimited and traj_count >= max_traj:
            break
        attempts = select_training_attempts(row, source)
        if not attempts:
            continue
        scenario_id = int(row.get("scenario_id", -1))
        row_dictionary = resolve_row_dictionary(row.get("dictionary_path") or row.get("dictionary") or str(default_dictionary))
        by_id = load_by_id(row_dictionary) if row_dictionary.exists() else default_by_id
        scenario = by_id.get(scenario_id)
        if scenario is None:
            continue

        goal = nl.scenario_goal(scenario)
        sit_vec = nl.compute_situation_vector(
            scenario,
            grid_n=grid_n,
            dt_nom=dt_nom,
            n_steps_nom=n_steps_nom,
            u_max_nom=u_max_nom,
            buffer_cells=buffer_cells,
            stop_tol=stop_tol,
        )

        for attempt in attempts:
            if not unlimited and traj_count >= max_traj:
                break
            states = _xy(attempt.get("states", []))
            inputs = np.asarray(attempt.get("inputs", []), dtype=np.float32)
            if states.shape[0] < 2:
                continue

            traj_idx = traj_count
            traj_count += 1
            for t in range(states.shape[0] - 1):
                prefix = states[: t + 1]
                ctx_global = make_context_window(prefix, L_c)
                ctx_local, goal_local, origin, heading = nl.transform_to_local(ctx_global, goal)
                dyn_feat = nl.nonlinear_dynamics_features(scenario, states[t], heading)
                next_local = (states[t + 1] - origin) @ nl.rotation_from_heading(heading).T
                if inputs.ndim == 2 and inputs.shape[0] > t and inputs.shape[1] >= 2:
                    u_global = np.asarray(inputs[t, :2], dtype=np.float32)
                else:
                    u_global = (states[t + 1] - states[t]) / float(dt_nom) - nl.nonlinear_drift_global(scenario, states[t])
                u_local = np.asarray(u_global, dtype=np.float32) @ nl.rotation_from_heading(heading).T

                ctx_list.append(ctx_local.astype(np.float32))
                sit_list.append(sit_vec.astype(np.float32))
                dyn_list.append(dyn_feat.astype(np.float32))
                goal_list.append(goal_local.astype(np.float32))
                u_list.append(u_local.astype(np.float32))
                next_list.append(next_local.astype(np.float32))
                traj_ids.append(traj_idx)

    if not ctx_list:
        raise RuntimeError("No successful benchmark trajectories found for S1 training.")

    ctx = np.stack(ctx_list, axis=0).astype(np.float32)
    sit = np.stack(sit_list, axis=0).astype(np.float32)
    dyn = np.stack(dyn_list, axis=0).astype(np.float32)
    goal = np.stack(goal_list, axis=0).astype(np.float32)
    u = np.stack(u_list, axis=0).astype(np.float32)
    next_local = np.stack(next_list, axis=0).astype(np.float32)
    traj_id = np.asarray(traj_ids, dtype=np.int32)

    return {
        "ctx": ctx,
        "sit": sit,
        "dyn": dyn,
        "goal": goal,
        "u": u,
        "next_local": next_local,
        "traj_id": traj_id,
        "meta": {
            "dynamics_mode": "nonlinear_point_policy",
            "ctx_shape": tuple(ctx.shape[1:]),
            "sit_dim": int(sit.shape[1]),
            "dyn_dim": int(dyn.shape[1]),
            "goal_dim": int(goal.shape[1]),
            "u_dim": int(u.shape[1]),
            "context_len": int(L_c),
            "dt_nom": float(dt_nom),
            "grid_n": int(grid_n),
            "n_steps_nom": int(n_steps_nom),
            "u_max_nom": float(u_max_nom),
            "buffer_cells": int(buffer_cells),
            "stop_tol": float(stop_tol),
            "trajectory_count": int(traj_count),
            "sample_count": int(ctx.shape[0]),
        },
    }


def grouped_train_val_split(traj_ids: np.ndarray, val_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(traj_ids)
    rng.shuffle(unique_ids)
    n_val_groups = max(1, int(len(unique_ids) * val_frac))
    val_ids = set(unique_ids[:n_val_groups].tolist())
    val_mask = np.array([tid in val_ids for tid in traj_ids], dtype=bool)
    return ~val_mask, val_mask


def make_sample_weights(goal_den: np.ndarray, next_den: np.ndarray, near_goal_boost: float, progress_boost: float) -> np.ndarray:
    eps = 1e-6
    goal_dist_now = np.linalg.norm(goal_den, axis=1)
    goal_dist_next = np.linalg.norm(goal_den - next_den, axis=1)
    expert_progress = np.maximum(goal_dist_now - goal_dist_next, 0.0)
    near_score = 1.0 / (goal_dist_now + 0.5)
    near_score = near_score / (near_score.mean() + eps)
    prog_score = expert_progress / (expert_progress.mean() + eps)
    w = 1.0 + near_goal_boost * near_score + progress_boost * prog_score
    return (w / (w.mean() + eps)).astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    configure_repo(root, args.mplconfigdir)

    dictionary = Path(args.dictionary).expanduser()
    if not dictionary.is_absolute():
        dictionary = root / "input" / dictionary

    from solvers import s1_nonlinear as nl

    rows = load_results(args.results_jsonl)
    data = build_samples(
        rows,
        root=root,
        default_dictionary=dictionary,
        source=str(args.source),
        max_trajectories=args.max_trajectories,
        L_c=int(args.context_len),
        dt_nom=float(args.dt_nom),
        grid_n=int(args.grid_n),
        n_steps_nom=int(args.n_steps_nom),
        u_max_nom=float(args.u_max_nom),
        buffer_cells=int(args.buffer_cells),
        stop_tol=float(args.stop_tol),
    )

    out_model = Path(args.out_model).expanduser()
    if not out_model.is_absolute():
        out_model = root / out_model
    out_dataset = Path(args.out_dataset).expanduser()
    if args.out_dataset:
        if not out_dataset.is_absolute():
            out_dataset = root / out_dataset
        out_dataset.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dataset, **{k: v for k, v in data.items() if k != "meta"})

    X_ctx = torch.from_numpy(data["ctx"]).float()
    X_sit = torch.from_numpy(data["sit"]).float()
    X_dyn = torch.from_numpy(data["dyn"]).float()
    X_goal = torch.from_numpy(data["goal"]).float()
    Y_u = torch.from_numpy(data["u"]).float()
    Y_next = torch.from_numpy(data["next_local"]).float()
    traj_ids = np.asarray(data["traj_id"], dtype=np.int32)
    meta = data["meta"]

    goal_den = data["goal"]
    next_den = data["next_local"]
    sample_weights_np = make_sample_weights(goal_den, next_den, args.near_goal_boost, args.progress_boost)
    if args.max_sample_weight > 0:
        sample_weights_np = np.minimum(sample_weights_np, float(args.max_sample_weight))
    sample_weights_np = sample_weights_np / (sample_weights_np.mean() + 1e-6)
    W = torch.from_numpy(sample_weights_np).float()

    train_mask, val_mask = grouped_train_val_split(traj_ids, args.val_frac, seed=42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = nl.NeuralSystem1ControlPolicyCNN(
        ctx_shape=X_ctx.shape[1:],
        sit_dim=X_sit.shape[1],
        dyn_dim=X_dyn.shape[1],
        goal_dim=X_goal.shape[1],
        u_dim=Y_u.shape[1],
        hidden=args.hidden,
        cnn_channels=args.cnn_channels,
        dropout=args.dropout,
    ).to(device)

    if args.init_model:
        init_model = Path(args.init_model).expanduser()
        if not init_model.is_absolute():
            init_model = root / init_model
        ckpt = torch.load(str(init_model), map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)

    if args.train_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    ctx_mean = torch.from_numpy(np.mean(data["ctx"], axis=0)).float().to(device)
    ctx_std = torch.from_numpy(np.std(data["ctx"], axis=0) + 1e-6).float().to(device)
    sit_mean = torch.from_numpy(np.mean(data["sit"], axis=0)).float().to(device)
    sit_std = torch.from_numpy(np.std(data["sit"], axis=0) + 1e-6).float().to(device)
    dyn_mean = torch.from_numpy(np.mean(data["dyn"], axis=0)).float().to(device)
    dyn_std = torch.from_numpy(np.std(data["dyn"], axis=0) + 1e-6).float().to(device)
    goal_mean = torch.from_numpy(np.mean(data["goal"], axis=0)).float().to(device)
    goal_std = torch.from_numpy(np.std(data["goal"], axis=0) + 1e-6).float().to(device)
    u_mean = torch.from_numpy(np.mean(data["u"], axis=0)).float().to(device)
    u_std = torch.from_numpy(np.std(data["u"], axis=0) + 1e-6).float().to(device)
    next_mean = torch.from_numpy(np.mean(data["next_local"], axis=0)).float().to(device)
    next_std = torch.from_numpy(np.std(data["next_local"], axis=0) + 1e-6).float().to(device)
    dt_nom = float(meta["dt_nom"])

    def norm_ctx(x): return (x - ctx_mean[None, :, :]) / ctx_std[None, :, :]
    def norm_vec(x, mean, std): return (x - mean[None, :]) / std[None, :]
    def denorm(x, mean, std): return x * std + mean

    def pointwise_loss(pred, target):
        if args.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none", beta=max(float(args.huber_delta), 1e-6)).mean(dim=1)
        return ((pred - target) ** 2).mean(dim=1)

    def weighted_mean(x, w):
        while w.ndim < x.ndim:
            w = w.unsqueeze(-1)
        return (x * w).mean()

    train_traj_ids = traj_ids[train_mask]
    val_traj_ids = traj_ids[val_mask]
    train_balance = np.ones_like(train_traj_ids, dtype=np.float32)
    if train_traj_ids.size:
        train_counts = np.bincount(train_traj_ids)
        train_counts = np.maximum(train_counts, 1)
        train_balance = (1.0 / train_counts[train_traj_ids]).astype(np.float32)
        train_balance = train_balance / (train_balance.mean() + 1e-6)

    val_balance = np.ones_like(val_traj_ids, dtype=np.float32)
    if val_traj_ids.size:
        val_counts = np.bincount(val_traj_ids)
        val_counts = np.maximum(val_counts, 1)
        val_balance = (1.0 / val_counts[val_traj_ids]).astype(np.float32)
        val_balance = val_balance / (val_balance.mean() + 1e-6)

    train_ds = TensorDataset(
        X_ctx[train_mask],
        X_sit[train_mask],
        X_dyn[train_mask],
        X_goal[train_mask],
        Y_u[train_mask],
        Y_next[train_mask],
        W[train_mask],
    )
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_balance).double(),
        num_samples=len(train_balance),
        replacement=True,
    )
    val_weights = torch.from_numpy((W[val_mask].cpu().numpy() * val_balance).astype(np.float32))
    val_weights = val_weights / (val_weights.mean() + 1e-6)
    val_ds = TensorDataset(
        X_ctx[val_mask],
        X_sit[val_mask],
        X_dyn[val_mask],
        X_goal[val_mask],
        Y_u[val_mask],
        Y_next[val_mask],
        val_weights,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=train_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    best_val = float("inf")
    out_model.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        for b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w in train_loader:
            b_ctx = b_ctx.to(device)
            b_sit = b_sit.to(device)
            b_dyn = b_dyn.to(device)
            b_goal = b_goal.to(device)
            b_u = b_u.to(device)
            b_next = b_next.to(device)
            b_w = b_w.to(device)

            optimizer.zero_grad()
            pred_u = model(norm_ctx(b_ctx), norm_vec(b_sit, sit_mean, sit_std), norm_vec(b_dyn, dyn_mean, dyn_std), norm_vec(b_goal, goal_mean, goal_std))
            loss_u_each = pointwise_loss(pred_u, b_u)
            loss_u = weighted_mean(loss_u_each, b_w)

            pred_u_den = denorm(pred_u, u_mean, u_std)
            goal_den = denorm(b_goal, goal_mean, goal_std)
            next_true = denorm(b_next, next_mean, next_std)
            drift_local = denorm(b_dyn[:, :2], dyn_mean[:2], dyn_std[:2])
            pred_next = dt_nom * (drift_local + pred_u_den)

            loss_next_each = pointwise_loss(pred_next, next_true)
            loss_next = weighted_mean(loss_next_each, b_w)

            pred_norm = torch.norm(pred_next, dim=1) + 1e-6
            true_norm = torch.norm(next_true, dim=1) + 1e-6
            cos_sim = (pred_next * next_true).sum(dim=1) / (pred_norm * true_norm)
            loss_dir = weighted_mean(1.0 - torch.clamp(cos_sim, -1.0, 1.0), b_w)

            denom = torch.clamp(true_norm.detach(), min=0.2)
            loss_speed = weighted_mean(((pred_norm - true_norm) / denom) ** 2, b_w)

            dist_now = torch.norm(goal_den, dim=1)
            dist_pred = torch.norm(goal_den - pred_next, dim=1)
            dist_true = torch.norm(goal_den - next_true, dim=1)
            true_progress = torch.relu(dist_now - dist_true)
            pred_progress = dist_now - dist_pred
            loss_progress = weighted_mean(torch.relu(args.progress_fraction * true_progress - pred_progress), b_w)

            total = (
                args.lambda_u * loss_u
                + args.lambda_next * loss_next
                + args.lambda_dir * loss_dir
                + args.lambda_speed * loss_speed
                + args.lambda_progress * loss_progress
            )
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(total.detach().cpu())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w in val_loader:
                b_ctx = b_ctx.to(device)
                b_sit = b_sit.to(device)
                b_dyn = b_dyn.to(device)
                b_goal = b_goal.to(device)
                b_u = b_u.to(device)
                b_next = b_next.to(device)
                b_w = b_w.to(device)

                pred_u = model(norm_ctx(b_ctx), norm_vec(b_sit, sit_mean, sit_std), norm_vec(b_dyn, dyn_mean, dyn_std), norm_vec(b_goal, goal_mean, goal_std))
                loss_u = weighted_mean(pointwise_loss(pred_u, b_u), b_w)
                pred_u_den = denorm(pred_u, u_mean, u_std)
                next_true = denorm(b_next, next_mean, next_std)
                drift_local = denorm(b_dyn[:, :2], dyn_mean[:2], dyn_std[:2])
                pred_next = dt_nom * (drift_local + pred_u_den)
                loss_next = weighted_mean(pointwise_loss(pred_next, next_true), b_w)
                total = args.lambda_u * loss_u + args.lambda_next * loss_next
                val_loss += float(total.detach().cpu())

        scheduler.step(val_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | train={train_loss/ max(1, len(train_loader)):.6f} | val={val_loss/ max(1, len(val_loader)):.6f}")

    if val_loss < best_val:
            best_val = val_loss
            nl.save_s1_checkpoint(
                model,
                out_model,
                meta={
                    "ctx_shape": tuple(X_ctx.shape[1:]),
                    "sit_dim": int(X_sit.shape[1]),
                    "dyn_dim": int(X_dyn.shape[1]),
                    "goal_dim": int(X_goal.shape[1]),
                    "u_dim": int(Y_u.shape[1]),
                    "hidden": int(args.hidden),
                    "cnn_channels": int(args.cnn_channels),
                    "dropout": float(args.dropout),
                    "dataset_meta": meta,
                },
                norm={
                    "ctx_mean": ctx_mean.cpu().numpy(),
                    "ctx_std": ctx_std.cpu().numpy(),
                    "sit_mean": sit_mean.cpu().numpy(),
                    "sit_std": sit_std.cpu().numpy(),
                    "dyn_mean": dyn_mean.cpu().numpy(),
                    "dyn_std": dyn_std.cpu().numpy(),
                    "goal_mean": goal_mean.cpu().numpy(),
                    "goal_std": goal_std.cpu().numpy(),
                    "u_mean": u_mean.cpu().numpy(),
                    "u_std": u_std.cpu().numpy(),
                    "next_mean": next_mean.cpu().numpy(),
                    "next_std": next_std.cpu().numpy(),
                },
            )

    print(f"[done] model saved to {out_model}")


if __name__ == "__main__":
    main()
