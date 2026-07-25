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
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler


def configure_repo(root: Path, mplconfigdir: str) -> None:
    for path in (root, root / "sofai", root / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from solvers._s2_common import resolve_mplconfigdir

    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, mplconfigdir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--dictionary", required=True)
    p.add_argument("--results_jsonl", nargs="+", required=True)
    p.add_argument("--audit_json", default="", help="Optional JSON audit written for the training inputs.")
    p.add_argument("--out_model", required=True)
    p.add_argument("--out_dataset", default="")
    p.add_argument("--init_model", default="")
    p.add_argument("--source", choices=["s2", "selected", "all_success", "fallback_success"], default="all_success")
    p.add_argument("--max_trajectories", type=int, default=200)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--grid_n", type=int, default=25)
    p.add_argument(
        "--dt_nom",
        type=float,
        default=0.075,
        help=(
            "Integrator step the policy is trained and executed at. Keep this equal to the "
            "step the S2 teachers record (SOFAI_MPC_DT and SOFAI_CBF_DT), otherwise the "
            "control and next-state targets disagree by their ratio."
        ),
    )
    p.add_argument(
        "--n_steps_nom",
        type=int,
        default=900,
        help=(
            "Steps of the nominal goal-seeking rollout that paints the corridor in the "
            "situation vector. Too few truncates the corridor before the goal, which "
            "leaves the policy with no map feature for the far end of long maps."
        ),
    )
    p.add_argument("--u_max_nom", type=float, default=3.0)
    p.add_argument("--buffer_cells", type=int, default=2)
    p.add_argument("--stop_tol", type=float, default=0.6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--device",
        default="cpu",
        help="PyTorch training device, for example cpu, cuda, cuda:0, or mps.",
    )
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
    p.add_argument("--lambda_rollout", type=float, default=0.5)
    p.add_argument("--lambda_smooth", type=float, default=0.10)
    p.add_argument("--lambda_obstacle", type=float, default=0.25)
    p.add_argument("--progress_fraction", type=float, default=0.9)
    p.add_argument("--lambda_u_range", type=float, default=0.0)
    p.add_argument("--loss_type", choices=["mse", "huber"], default="huber")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--train_head_only", action="store_true")
    p.add_argument("--near_goal_boost", type=float, default=2.0)
    p.add_argument("--progress_boost", type=float, default=2.0)
    p.add_argument("--fallback_success_weight", type=float, default=10.0)
    p.add_argument("--s1_success_weight", type=float, default=1.0)
    p.add_argument("--bootstrap_success_weight", type=float, default=5.0)
    p.add_argument("--dagger_success_weight", type=float, default=10.0)
    p.add_argument("--max_sample_weight", type=float, default=25.0)
    p.add_argument("--action_mode", choices=["delta_u", "absolute_u"], default="delta_u")
    p.add_argument("--rollout_horizon", type=int, default=8)
    p.add_argument("--rollout_stride", type=int, default=4)
    p.add_argument("--rollout_every", type=int, default=8)
    p.add_argument("--rollout_collision_margin", type=float, default=0.25)
    p.add_argument("--mplconfigdir", default="")
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
                    row = json.loads(line)
                    row["_training_jsonl"] = str(path.resolve())
                    rows.append(row)
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

    if source == "all_success": ## doing this now 
        return [attempt for attempt in attempts if bool(attempt.get("success"))]

    if source == "fallback_success": 
        s1_failed = any(str(attempt.get("system")) == "s1" and not bool(attempt.get("success")) for attempt in attempts)
        if not s1_failed:
            return []
        return [
            attempt
            for attempt in attempts
            if str(attempt.get("system")) == "s2" and bool(attempt.get("success"))
        ]

    selected_name = str(result.get("selected_attempt") or "").strip()
    if selected_name:
        for attempt in attempts:
            if str(attempt.get("name")) == selected_name and bool(attempt.get("success")):
                return [attempt]

    for attempt in attempts:
        if bool(attempt.get("success")):
            return [attempt]
    return []


def attempt_weight(
    result: Dict[str, Any],
    attempt: Dict[str, Any],
    *,
    fallback_success_weight: float,
    s1_success_weight: float,
    bootstrap_success_weight: float,
    dagger_success_weight: float,
) -> float:
    system = str(attempt.get("system"))
    run_type = str(result.get("run_type"))
    attempts = result.get("attempts", []) or []
    s1_failed = any(str(a.get("system")) == "s1" and not bool(a.get("success")) for a in attempts)

    if bool(result.get("dagger")) and system == "s2":
        return max(float(dagger_success_weight), 0.0)
    if run_type == "s2" and system == "s2":
        return max(float(bootstrap_success_weight), 0.0)
    if system == "s2" and s1_failed and bool(attempt.get("success")):
        return max(float(fallback_success_weight), 0.0)
    if system == "s1" and bool(attempt.get("success")):
        return max(float(s1_success_weight), 0.0)
    return 1.0


def _xy(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32)


def attempt_dt(attempt: Dict[str, Any], dt_nom: float) -> float:
    """Integrator step the demonstration was generated with.

    Every solver records its own step, so a mismatch with `dt_nom` is what makes
    the recorded control and the recorded state delta inconsistent targets.
    Result files written before the field existed fall back to `dt_nom`.
    """
    try:
        dt = float(attempt.get("dt"))
    except (TypeError, ValueError):
        return float(dt_nom)
    return dt if dt > 1e-9 else float(dt_nom)


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
    fallback_success_weight: float,
    s1_success_weight: float,
    bootstrap_success_weight: float,
    dagger_success_weight: float,
    action_mode: str,
):
    from input.input_handler import load_scenarios
    from solvers import s1_nonlinear as nl

    scenario_cache: Dict[str, Dict[int, Any]] = {}

    def load_by_id(dictionary_path: Path) -> Dict[int, Any]:
        key = str(dictionary_path.resolve())
        if key not in scenario_cache:
            scenarios = load_scenarios(str(dictionary_path))
            by_id: Dict[int, Any] = {}
            for i, sc in enumerate(scenarios):
                by_id[i] = sc
                by_id[int(getattr(sc, "scenario_id", i))] = sc
            scenario_cache[key] = by_id
        return scenario_cache[key]

    default_by_id = load_by_id(default_dictionary)

    def resolve_row_dictionary(row_value: Any) -> Path | None:
        if row_value in (None, ""):
            return default_dictionary
        candidate = Path(str(row_value)).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend((root / candidate, root / "input" / candidate, root / "input" / "nl" / candidate))
        # Results copied from another machine retain absolute paths.  Rebase by
        # filename instead of silently substituting the evaluation dictionary.
        candidates.extend((root / "input" / candidate.name, root / "input" / "nl" / candidate.name))
        for path in candidates:
            if path.exists():
                return path
        return None

    ctx_list: List[np.ndarray] = []
    sit_list: List[np.ndarray] = []
    dyn_list: List[np.ndarray] = []
    goal_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    du_list: List[np.ndarray] = []
    prev_u_list: List[np.ndarray] = []
    next_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    next_pos_list: List[np.ndarray] = []
    heading_list: List[float] = []
    drift_global_list: List[np.ndarray] = []
    rect_list: List[np.ndarray] = []
    traj_ids: List[int] = []
    source_weight_list: List[float] = []
    trajectory_count_by_jsonl: Dict[str, int] = {}
    row_count_by_jsonl: Dict[str, int] = {}
    trajectory_count_by_type: Dict[str, int] = {
        "bootstrap_s2": 0,
        "s1_success": 0,
        "s2_fallback_success": 0,
        "other_success": 0,
    }
    selected_success_count = 0
    skipped_missing_dictionary = 0
    skipped_missing_scenario = 0
    skipped_invalid_states = 0
    sample_count_by_demo_dt: Dict[str, int] = {}
    rescaled_dt_trajectories = 0
    for row in result_rows:
        source_file = str(row.get("_training_jsonl", ""))
        row_count_by_jsonl[source_file] = row_count_by_jsonl.get(source_file, 0) + 1
        trajectory_count_by_jsonl.setdefault(source_file, 0)

    traj_count = 0
    max_traj = int(max_trajectories)
    unlimited = max_traj <= 0
    for row in result_rows:
        if not unlimited and traj_count >= max_traj:
            break
        attempts = select_training_attempts(row, source)
        if not attempts:
            continue
        selected_success_count += len(attempts)
        scenario_id = int(row.get("scenario_index", row.get("scenario_id", -1)))
        row_dictionary = resolve_row_dictionary(row.get("dictionary_path") or row.get("dictionary") or str(default_dictionary))
        if row_dictionary is None:
            skipped_missing_dictionary += len(attempts)
            continue
        by_id = load_by_id(row_dictionary)
        scenario_override = row.get("scenario_override")
        scenario = scenario_override if isinstance(scenario_override, dict) else by_id.get(scenario_id)
        if scenario is None:
            skipped_missing_scenario += len(attempts)
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
                skipped_invalid_states += 1
                continue
            # The policy always integrates at dt_nom, so state targets have to be
            # expressed on that step. Demonstrations recorded at another step are
            # resampled along each segment, which keeps them consistent with the
            # recorded control but only reproduces their geometry to first order.
            demo_dt = attempt_dt(attempt, dt_nom)
            step_scale = float(dt_nom) / demo_dt
            dt_key = f"{demo_dt:.6g}"
            sample_count_by_demo_dt[dt_key] = sample_count_by_demo_dt.get(dt_key, 0) + int(states.shape[0] - 1)
            if abs(step_scale - 1.0) > 1e-6:
                rescaled_dt_trajectories += 1
            traj_weight = attempt_weight(
                row,
                attempt,
                fallback_success_weight=fallback_success_weight,
                s1_success_weight=s1_success_weight,
                bootstrap_success_weight=bootstrap_success_weight,
                dagger_success_weight=dagger_success_weight,
            )

            traj_idx = traj_count
            traj_count += 1
            source_file = str(row.get("_training_jsonl", ""))
            trajectory_count_by_jsonl[source_file] = trajectory_count_by_jsonl.get(source_file, 0) + 1
            system = str(attempt.get("system", ""))
            run_type = str(row.get("run_type", ""))
            s1_failed = any(
                str(item.get("system")) == "s1" and not bool(item.get("success"))
                for item in (row.get("attempts", []) or [])
            )
            if run_type == "s2" and system == "s2":
                trajectory_count_by_type["bootstrap_s2"] += 1
            elif system == "s1":
                trajectory_count_by_type["s1_success"] += 1
            elif system == "s2" and s1_failed:
                trajectory_count_by_type["s2_fallback_success"] += 1
            else:
                trajectory_count_by_type["other_success"] += 1
            for t in range(states.shape[0] - 1):
                prefix = states[: t + 1]
                ctx_global = make_context_window(prefix, L_c)
                ctx_local, goal_local, origin, heading = nl.transform_to_local(ctx_global, goal)
                dyn_feat = nl.nonlinear_dynamics_features(scenario, states[t], heading)
                next_state = states[t] + step_scale * (states[t + 1] - states[t])
                next_local = (next_state - origin) @ nl.rotation_from_heading(heading).T
                if inputs.ndim == 2 and inputs.shape[0] > t and inputs.shape[1] >= 2:
                    u_global = np.asarray(inputs[t, :2], dtype=np.float32)
                else:
                    u_global = (states[t + 1] - states[t]) / demo_dt - nl.nonlinear_drift_global(scenario, states[t])
                if t == 0:
                    prev_u_global = np.zeros(2, dtype=np.float32)
                elif inputs.ndim == 2 and inputs.shape[0] >= t and inputs.shape[1] >= 2:
                    prev_u_global = np.asarray(inputs[t - 1, :2], dtype=np.float32)
                else:
                    prev_u_global = (states[t] - states[t - 1]) / demo_dt - nl.nonlinear_drift_global(scenario, states[t - 1])
                u_local = np.asarray(u_global, dtype=np.float32) @ nl.rotation_from_heading(heading).T
                prev_u_local = np.asarray(prev_u_global, dtype=np.float32) @ nl.rotation_from_heading(heading).T
                du_local = u_local - prev_u_local
                if action_mode == "delta_u":
                    dyn_feat = np.concatenate([dyn_feat, prev_u_local], axis=0)

                ctx_list.append(ctx_local.astype(np.float32))
                sit_list.append(sit_vec.astype(np.float32))
                dyn_list.append(dyn_feat.astype(np.float32))
                goal_list.append(goal_local.astype(np.float32))
                u_list.append(u_local.astype(np.float32))
                du_list.append(du_local.astype(np.float32))
                prev_u_list.append(prev_u_local.astype(np.float32))
                next_list.append(next_local.astype(np.float32))
                pos_list.append(states[t].astype(np.float32))
                next_pos_list.append(next_state.astype(np.float32))
                heading_list.append(float(heading))
                drift_global_list.append(nl.nonlinear_drift_global(scenario, states[t]).astype(np.float32))
                rect_list.append(np.asarray(nl.scenario_rects(scenario), dtype=np.float32).reshape(-1, 4))
                traj_ids.append(traj_idx)
                source_weight_list.append(float(traj_weight))

    if not ctx_list:
        raise RuntimeError("No successful benchmark trajectories found for S1 training.")

    if rescaled_dt_trajectories:
        print(
            f"[warn] {rescaled_dt_trajectories} demonstration trajectories were generated at a timestep "
            f"other than dt_nom={dt_nom}; samples per demonstration timestep: {sample_count_by_demo_dt}. "
            "Their state targets were resampled onto dt_nom. Run the S2 teachers at dt_nom to avoid it."
        )

    ctx = np.stack(ctx_list, axis=0).astype(np.float32)
    sit = np.stack(sit_list, axis=0).astype(np.float32)
    dyn = np.stack(dyn_list, axis=0).astype(np.float32)
    goal = np.stack(goal_list, axis=0).astype(np.float32)
    u = np.stack(u_list, axis=0).astype(np.float32)
    du = np.stack(du_list, axis=0).astype(np.float32)
    prev_u = np.stack(prev_u_list, axis=0).astype(np.float32)
    next_local = np.stack(next_list, axis=0).astype(np.float32)
    pos = np.stack(pos_list, axis=0).astype(np.float32)
    next_pos = np.stack(next_pos_list, axis=0).astype(np.float32)
    heading = np.asarray(heading_list, dtype=np.float32)
    drift_global = np.stack(drift_global_list, axis=0).astype(np.float32)
    max_rects = max((rect.shape[0] for rect in rect_list), default=0)
    rects = np.zeros((len(rect_list), max(1, max_rects), 4), dtype=np.float32)
    rect_mask = np.zeros((len(rect_list), max(1, max_rects)), dtype=np.float32)
    for index, rect in enumerate(rect_list):
        if rect.size:
            rects[index, : rect.shape[0]] = rect
            rect_mask[index, : rect.shape[0]] = 1.0
    traj_id = np.asarray(traj_ids, dtype=np.int32)
    source_weight = np.asarray(source_weight_list, dtype=np.float32)

    return {
        "ctx": ctx,
        "sit": sit,
        "dyn": dyn,
        "goal": goal,
        "u": u,
        "du": du,
        "prev_u": prev_u,
        "next_local": next_local,
        "pos": pos,
        "next_pos": next_pos,
        "heading": heading,
        "drift_global": drift_global,
        "rects": rects,
        "rect_mask": rect_mask,
        "traj_id": traj_id,
        "source_weight": source_weight,
        "meta": {
            "dynamics_mode": "nonlinear_point_policy",
            "action_mode": str(action_mode),
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
            "input_jsonls": sorted(row_count_by_jsonl),
            "input_row_count_by_jsonl": row_count_by_jsonl,
            "trajectory_count_by_jsonl": trajectory_count_by_jsonl,
            "trajectory_count_by_type": trajectory_count_by_type,
            "sample_count_by_demo_dt": sample_count_by_demo_dt,
            "rescaled_dt_trajectories": int(rescaled_dt_trajectories),
            "selected_success_count": int(selected_success_count),
            "skipped_missing_dictionary": int(skipped_missing_dictionary),
            "skipped_missing_scenario": int(skipped_missing_scenario),
            "skipped_invalid_states": int(skipped_invalid_states),
        },
    }


def grouped_train_val_split(traj_ids: np.ndarray, val_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(traj_ids)
    if unique_ids.size <= 1 or val_frac <= 0:
        train_mask = np.ones_like(traj_ids, dtype=bool)
        return train_mask, ~train_mask
    rng.shuffle(unique_ids)
    n_val_groups = max(1, int(len(unique_ids) * val_frac))
    n_val_groups = min(n_val_groups, len(unique_ids) - 1)
    val_ids = set(unique_ids[:n_val_groups].tolist())
    val_mask = np.array([tid in val_ids for tid in traj_ids], dtype=bool)
    return ~val_mask, val_mask


## sample weights for the current objective function 
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


def rollout_window_starts(traj_ids: np.ndarray, horizon: int, stride: int) -> np.ndarray:
    """Return contiguous trajectory windows; never cross a trajectory boundary."""
    if horizon <= 1 or traj_ids.size < horizon:
        return np.zeros(0, dtype=np.int64)
    starts: List[int] = []
    begin = 0
    while begin < traj_ids.size:
        end = begin + 1
        while end < traj_ids.size and traj_ids[end] == traj_ids[begin]:
            end += 1
        starts.extend(range(begin, max(begin, end - horizon + 1), max(1, stride)))
        begin = end
    return np.asarray(starts, dtype=np.int64)


class RolloutWindowDataset(Dataset):
    def __init__(self, tensors: Sequence[torch.Tensor], starts: np.ndarray, horizon: int):
        self.tensors = tuple(tensors)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        stop = start + self.horizon
        return tuple(tensor[start:stop] for tensor in self.tensors)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    configure_repo(root, args.mplconfigdir)

    dictionary = Path(args.dictionary).expanduser()
    if not dictionary.is_absolute():
        candidates = (root / dictionary, root / "input" / dictionary)
        dictionary = next((path for path in candidates if path.exists()), candidates[0])
    if not dictionary.is_file():
        raise FileNotFoundError(dictionary)

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
        fallback_success_weight=float(args.fallback_success_weight),
        s1_success_weight=float(args.s1_success_weight),
        bootstrap_success_weight=float(args.bootstrap_success_weight),
        dagger_success_weight=float(args.dagger_success_weight),
        action_mode=str(args.action_mode),
    )

    if data["meta"]["trajectory_count"] != data["meta"]["selected_success_count"]:
        raise RuntimeError(
            "Training input audit failed: not every selected successful trajectory could be used. "
            f"selected={data['meta']['selected_success_count']} "
            f"used={data['meta']['trajectory_count']} "
            f"missing_dictionary={data['meta']['skipped_missing_dictionary']} "
            f"missing_scenario={data['meta']['skipped_missing_scenario']} "
            f"invalid_states={data['meta']['skipped_invalid_states']}"
        )

    if args.audit_json:
        audit_path = Path(args.audit_json).expanduser()
        if not audit_path.is_absolute():
            audit_path = root / audit_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(data["meta"], indent=2, sort_keys=True) + "\n")
        print(f"[write] {audit_path}")

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
    X_prev_u = torch.from_numpy(data["prev_u"]).float()
    Y_u = torch.from_numpy(data["u"]).float()
    Y_du = torch.from_numpy(data["du"]).float()
    Y_next = torch.from_numpy(data["next_local"]).float()
    X_pos = torch.from_numpy(data["pos"]).float()
    X_next_pos = torch.from_numpy(data["next_pos"]).float()
    X_heading = torch.from_numpy(data["heading"]).float()
    X_drift_global = torch.from_numpy(data["drift_global"]).float()
    X_rects = torch.from_numpy(data["rects"]).float()
    X_rect_mask = torch.from_numpy(data["rect_mask"]).float()
    traj_ids = np.asarray(data["traj_id"], dtype=np.int32)
    meta = data["meta"]

    goal_den = data["goal"]
    next_den = data["next_local"]
    sample_weights_np = make_sample_weights(goal_den, next_den, args.near_goal_boost, args.progress_boost)
    if args.max_sample_weight > 0:
        sample_weights_np = np.minimum(sample_weights_np, float(args.max_sample_weight))
    sample_weights_np = sample_weights_np / (sample_weights_np.mean() + 1e-6)
    source_weights_np = np.asarray(data["source_weight"], dtype=np.float32)
    source_mean = float(source_weights_np.mean()) if source_weights_np.size else 0.0
    source_weights_np = source_weights_np / source_mean if source_mean > 0 else np.ones_like(source_weights_np)
    W = torch.from_numpy(sample_weights_np).float()

    train_mask, val_mask = grouped_train_val_split(traj_ids, args.val_frac, seed=42)

    try:
        device = torch.device(str(args.device))
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid training device {args.device!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested training device {device}, but CUDA is unavailable in this PyTorch environment."
        )
    mps_backend = getattr(torch.backends, "mps", None)
    if device.type == "mps" and (mps_backend is None or not mps_backend.is_available()):
        raise RuntimeError(
            f"Requested training device {device}, but MPS is unavailable in this PyTorch environment."
        )
    print(f"[train] device={device}")

    #### training model 
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
        old_meta = ckpt.get("meta", {})
        compatible = (
            old_meta.get("action_mode", "absolute_u") == args.action_mode
            and int(old_meta.get("dyn_dim", -1)) == int(X_dyn.shape[1])
        )
        if "state_dict" in ckpt and compatible:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif "state_dict" in ckpt:
            raise RuntimeError(
                "Incompatible init model: action_mode/dyn_dim do not match the training dataset. "
                "Regenerate the base S1 checkpoint before controlled continual retraining."
            )

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
    du_mean = torch.from_numpy(np.mean(data["du"], axis=0)).float().to(device)
    du_std = torch.from_numpy(np.std(data["du"], axis=0) + 1e-6).float().to(device)
    next_mean = torch.from_numpy(np.mean(data["next_local"], axis=0)).float().to(device)
    next_std = torch.from_numpy(np.std(data["next_local"], axis=0) + 1e-6).float().to(device)
    dt_nom = float(meta["dt_nom"])

    def norm_ctx(x): return (x - ctx_mean[None, :, :]) / ctx_std[None, :, :]
    def norm_vec(x, mean, std): return (x - mean[None, :]) / std[None, :]
    def denorm(x, mean, std): return x * std + mean

    Y_action = Y_du if args.action_mode == "delta_u" else Y_u
    action_mean = du_mean if args.action_mode == "delta_u" else u_mean
    action_std = du_std if args.action_mode == "delta_u" else u_std
    Y_action_norm = (Y_action - action_mean.cpu()) / action_std.cpu()

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
        Y_action_norm[train_mask],
        X_prev_u[train_mask],
        Y_next[train_mask],
        W[train_mask],
    )
    train_sampler_weights = train_balance * source_weights_np[train_mask]
    train_sampler_weights = train_sampler_weights / (train_sampler_weights.mean() + 1e-6)
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_sampler_weights).double(),
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
        Y_action_norm[val_mask],
        X_prev_u[val_mask],
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

    horizon = max(0, int(args.rollout_horizon))
    starts = rollout_window_starts(traj_ids, horizon, int(args.rollout_stride))
    train_starts = starts[train_mask[starts]] if starts.size else starts
    val_starts = starts[val_mask[starts]] if starts.size else starts
    window_tensors = (
        X_ctx,
        X_sit,
        X_dyn,
        X_goal,
        X_prev_u,
        X_pos,
        X_next_pos,
        X_heading,
        X_drift_global,
        X_rects,
        X_rect_mask,
    )
    train_rollout_loader = DataLoader(
        RolloutWindowDataset(window_tensors, train_starts, horizon),
        batch_size=args.batch,
        shuffle=True,
    ) if train_starts.size else None
    val_rollout_loader = DataLoader(
        RolloutWindowDataset(window_tensors, val_starts, horizon),
        batch_size=args.batch,
        shuffle=False,
    ) if val_starts.size else None

    best_val = float("inf")
    out_model.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint() -> None:
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
                "action_mode": str(args.action_mode),
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
                "du_mean": du_mean.cpu().numpy(),
                "du_std": du_std.cpu().numpy(),
                "next_mean": next_mean.cpu().numpy(),
                "next_std": next_std.cpu().numpy(),
            },
        )

    def local_to_global(u_local: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        return torch.stack(
            [u_local[:, 0] * cos_h - u_local[:, 1] * sin_h, u_local[:, 0] * sin_h + u_local[:, 1] * cos_h],
            dim=1,
        )

    def global_to_local(u_global: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        return torch.stack(
            [u_global[:, 0] * cos_h + u_global[:, 1] * sin_h, -u_global[:, 0] * sin_h + u_global[:, 1] * cos_h],
            dim=1,
        )

    def nonlinear_drift_global(x: torch.Tensor, dyn: torch.Tensor) -> torch.Tensor:
        """Torch version of the benchmark nonlinear drift for rollout supervision."""
        a, b, shear, damp, omega = (dyn[:, 6], dyn[:, 7], dyn[:, 8], dyn[:, 9], dyn[:, 10])
        x1, x2 = x[:, 0], x[:, 1]
        sink = torch.stack([-a * torch.tanh(x1) + shear * torch.sin(x2), -b * torch.tanh(x2) - 0.5 * shear * torch.sin(x1)], dim=1)
        rotate_cw = torch.stack([-damp * torch.tanh(x1) + omega * torch.sin(x2), -damp * torch.tanh(x2) - omega * torch.sin(x1)], dim=1)
        rotate_ccw = torch.stack([-damp * torch.tanh(x1) - omega * torch.sin(x2), -damp * torch.tanh(x2) + omega * torch.sin(x1)], dim=1)
        weak_shear = torch.stack([-a * torch.tanh(x1) + 2.0 * shear * x2, -b * torch.tanh(x2) + 0.25 * shear * x1], dim=1)
        regime = dyn[:, 2:6]
        return (
            regime[:, 0:1] * sink
            + regime[:, 1:2] * rotate_cw
            + regime[:, 2:3] * rotate_ccw
            + regime[:, 3:4] * weak_shear
        )

    def obstacle_penalty(pos: torch.Tensor, rects: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, y = pos[:, 0:1], pos[:, 1:2]
        xmin, ymin, xmax, ymax = (rects[:, :, 0], rects[:, :, 1], rects[:, :, 2], rects[:, :, 3])
        dx = torch.maximum(torch.maximum(xmin - x, x - xmax), torch.zeros_like(xmin))
        dy = torch.maximum(torch.maximum(ymin - y, y - ymax), torch.zeros_like(ymin))
        outside = torch.sqrt(dx.square() + dy.square() + 1e-12)
        inside_depth = torch.minimum(torch.minimum(x - xmin, xmax - x), torch.minimum(y - ymin, ymax - y))
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        signed_distance = torch.where(inside, -inside_depth, outside)
        margin = max(float(args.rollout_collision_margin), 1e-6)
        penalty = (torch.relu(margin - signed_distance) / margin).square() * mask
        return penalty.max(dim=1).values.mean()

    def rollout_losses(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            b_ctx,
            b_sit,
            b_dyn,
            b_goal,
            b_prev,
            b_pos,
            b_next_pos,
            b_heading,
            _b_drift,
            b_rects,
            b_rect_mask,
        ) = (item.to(device) for item in batch)
        pos = b_pos[:, 0].clone()
        prev_u_global = local_to_global(b_prev[:, 0], b_heading[:, 0])
        rollout_loss = torch.zeros((), device=device)
        smooth_loss = torch.zeros((), device=device)
        clearance_loss = torch.zeros((), device=device)
        for step in range(b_ctx.shape[1]):
            heading = b_heading[:, step]
            dyn_raw = b_dyn[:, step].clone()
            prev_u_local = global_to_local(prev_u_global, heading)
            if args.action_mode == "delta_u":
                dyn_raw[:, -2:] = prev_u_local
            pred_action_norm = model(
                norm_ctx(b_ctx[:, step]),
                norm_vec(b_sit[:, step], sit_mean, sit_std),
                norm_vec(dyn_raw, dyn_mean, dyn_std),
                norm_vec(b_goal[:, step], goal_mean, goal_std),
            )
            pred_action = denorm(pred_action_norm, action_mean, action_std)
            u_local = prev_u_local + pred_action if args.action_mode == "delta_u" else pred_action
            u_global = local_to_global(u_local, heading)
            pos = pos + dt_nom * (nonlinear_drift_global(pos, dyn_raw) + u_global)
            rollout_loss = rollout_loss + F.smooth_l1_loss(pos, b_next_pos[:, step], beta=max(float(args.huber_delta), 1e-6))
            if args.action_mode == "delta_u":
                smooth_loss = smooth_loss + (pred_action / max(float(args.u_max_nom), 1e-6)).square().mean()
            clearance_loss = clearance_loss + obstacle_penalty(pos, b_rects[:, step], b_rect_mask[:, step])
            prev_u_global = u_global
        divisor = max(int(b_ctx.shape[1]), 1)
        return rollout_loss / divisor, smooth_loss / divisor, clearance_loss / divisor

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        train_rollout_iter = iter(train_rollout_loader) if train_rollout_loader is not None else None
        for batch_index, (b_ctx, b_sit, b_dyn, b_goal, b_action, b_prev_u, b_next, b_w) in enumerate(train_loader):
            b_ctx = b_ctx.to(device)
            b_sit = b_sit.to(device)
            b_dyn = b_dyn.to(device)
            b_goal = b_goal.to(device)
            b_action = b_action.to(device)
            b_prev_u = b_prev_u.to(device)
            b_next = b_next.to(device)
            b_w = b_w.to(device)

            optimizer.zero_grad()
            pred_action_norm = model(norm_ctx(b_ctx), norm_vec(b_sit, sit_mean, sit_std), norm_vec(b_dyn, dyn_mean, dyn_std), norm_vec(b_goal, goal_mean, goal_std))
            loss_u_each = pointwise_loss(pred_action_norm, b_action)
            loss_u = weighted_mean(loss_u_each, b_w)

            pred_action = denorm(pred_action_norm, action_mean, action_std)
            pred_u_den = b_prev_u + pred_action if args.action_mode == "delta_u" else pred_action
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

            loss_rollout = torch.zeros((), device=device)
            loss_smooth = torch.zeros((), device=device)
            loss_obstacle = torch.zeros((), device=device)
            if train_rollout_iter is not None and batch_index % max(1, int(args.rollout_every)) == 0:
                try:
                    rollout_batch = next(train_rollout_iter)
                except StopIteration:
                    train_rollout_iter = iter(train_rollout_loader)
                    rollout_batch = next(train_rollout_iter)
                loss_rollout, loss_smooth, loss_obstacle = rollout_losses(rollout_batch)

            total = (
                args.lambda_u * loss_u
                + args.lambda_next * loss_next
                + args.lambda_dir * loss_dir
                + args.lambda_speed * loss_speed
                + args.lambda_progress * loss_progress
                + args.lambda_rollout * loss_rollout
                + args.lambda_smooth * loss_smooth
                + args.lambda_obstacle * loss_obstacle
            )
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(total.detach().cpu())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_ctx, b_sit, b_dyn, b_goal, b_action, b_prev_u, b_next, b_w in val_loader:
                b_ctx = b_ctx.to(device)
                b_sit = b_sit.to(device)
                b_dyn = b_dyn.to(device)
                b_goal = b_goal.to(device)
                b_action = b_action.to(device)
                b_prev_u = b_prev_u.to(device)
                b_next = b_next.to(device)
                b_w = b_w.to(device)

                pred_action_norm = model(norm_ctx(b_ctx), norm_vec(b_sit, sit_mean, sit_std), norm_vec(b_dyn, dyn_mean, dyn_std), norm_vec(b_goal, goal_mean, goal_std))
                loss_u = weighted_mean(pointwise_loss(pred_action_norm, b_action), b_w)
                pred_action = denorm(pred_action_norm, action_mean, action_std)
                pred_u_den = b_prev_u + pred_action if args.action_mode == "delta_u" else pred_action
                next_true = denorm(b_next, next_mean, next_std)
                drift_local = denorm(b_dyn[:, :2], dyn_mean[:2], dyn_std[:2])
                pred_next = dt_nom * (drift_local + pred_u_den)
                loss_next = weighted_mean(pointwise_loss(pred_next, next_true), b_w)
                total = args.lambda_u * loss_u + args.lambda_next * loss_next
                val_loss += float(total.detach().cpu())

            if val_rollout_loader is not None:
                rollout_values = []
                for rollout_batch in val_rollout_loader:
                    loss_rollout, loss_smooth, loss_obstacle = rollout_losses(rollout_batch)
                    rollout_values.append(
                        float((args.lambda_rollout * loss_rollout + args.lambda_smooth * loss_smooth + args.lambda_obstacle * loss_obstacle).cpu())
                    )
                if rollout_values:
                    val_loss += float(np.mean(rollout_values))

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) else float("nan")
        monitor_loss = avg_val_loss if len(val_loader) else avg_train_loss
        scheduler.step(monitor_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | train={avg_train_loss:.6f} | val={avg_val_loss:.6f}")

        if monitor_loss < best_val:
            best_val = monitor_loss
            save_checkpoint()

    print(f"[done] model saved to {out_model}")


if __name__ == "__main__":
    main()
