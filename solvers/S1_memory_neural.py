"""Pure neural System 1 wrapper for the nonlinear maze setup.

The public entrypoint keeps the historical name `solveMemoryNeural` so the
existing runners do not need to change, but the implementation no longer uses
episodic memory. It loads a neural policy trained on trajectories collected
from System 2 and rolls out a nonlinear point-robot model.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from solvers import s1_nonlinear as nl

SOLVER_DIR = Path(__file__).resolve().parent

_CACHE: Optional[Dict[str, Any]] = None
_CACHE_LOCK = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else float(default)


def _first_existing(candidates, *, env_name: str = "", required: bool = False) -> Path:
    if env_name and os.environ.get(env_name):
        path = Path(os.environ[env_name]).expanduser()
        if required and not path.exists():
            raise FileNotFoundError(f"{env_name} points to a missing path: {path}")
        return path

    paths = [Path(p).expanduser() for p in candidates]
    for path in paths:
        if path.exists():
            return path
    if required:
        tried = "\n  ".join(str(p) for p in paths)
        raise FileNotFoundError(f"Could not find required S1 asset. Tried:\n  {tried}")
    return paths[0]


def _default_model_path() -> Path:
    return _first_existing(
        [
            SOLVER_DIR / "s1_policy_nonlinear.pth",
            SOLVER_DIR / "s1_policy_nonlinear_dense_clutter.pth",
            SOLVER_DIR / "s1_policy_nonlinear_latest.pth",
            Path("/Users/apple/Desktop/sofai/db/by_env/dense_clutter_nl/s1_policy_nonlinear.pth"),
            Path("/Users/apple/Desktop/sofai/db/by_env/dense_clutter/s1_policy_nonlinear.pth"),
        ],
        env_name="SOFAI_NEW_S1_MODEL",
        required=True,
    )


def _make_args(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # The integrator step and the situation-vector geometry are properties of the
    # trained policy: rolling out with different values than training used makes
    # the learned controls mean something else. Take them from the checkpoint and
    # let the environment override only when set explicitly.
    trained = dict((meta or {}).get("dataset_meta", {}) or {})

    def trained_float(env_name: str, key: str, default: float) -> float:
        return _env_float(env_name, float(trained.get(key, default)))

    def trained_int(env_name: str, key: str, default: int) -> int:
        return _env_int(env_name, int(trained.get(key, default)))

    return {
        # Same closed-loop budget as the S2 solvers (SOFAI_MPC_STEPS and
        # SOFAI_CBF_STEPS), so S1 is allowed to reach as far as its teachers
        # rather than being cut off part-way through a long corridor.
        "total_steps": _env_int("SOFAI_NEW_S1_STEPS", 900),
        # End failed S1 attempts early without limiting trajectories that are
        # still either moving or making meaningful progress to the goal.
        "stall_patience": _env_int("SOFAI_NEW_S1_STALL_PATIENCE", 40),
        "stall_tol": _env_float("SOFAI_NEW_S1_STALL_TOL", 0.05),
        "progress_patience": _env_int("SOFAI_NEW_S1_PROGRESS_PATIENCE", 200),
        "progress_tol": _env_float("SOFAI_NEW_S1_PROGRESS_TOL", 0.5),
        # Reuse a policy prediction briefly. This keeps S1 an approximate,
        # low-latency controller without changing the checkpoint format.
        "action_hold": _env_int("SOFAI_NEW_S1_ACTION_HOLD", 4),
        "dt_nom": trained_float("SOFAI_NEW_S1_DT", "dt_nom", 0.075),
        "u_max_nom": trained_float("SOFAI_NEW_S1_U_MAX", "u_max_nom", 3.0),
        "goal_tol": _env_float("SOFAI_NEW_S1_GOAL_TOL", 0.5),
        "collision_margin": _env_float("SOFAI_NEW_S1_COLLISION_MARGIN", 0.05),
        "grid_n": trained_int("SOFAI_NEW_S1_GRID_N", "grid_n", 25),
        "n_steps_nom": trained_int("SOFAI_NEW_S1_N_STEPS_NOM", "n_steps_nom", 900),
        "buffer_cells": trained_int("SOFAI_NEW_S1_BUFFER_CELLS", "buffer_cells", 2),
        "stop_tol": trained_float("SOFAI_NEW_S1_STOP_TOL", "stop_tol", 0.6),
        "debug": _env_bool("SOFAI_NEW_S1_DEBUG", False),
    }


def _init():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE

        import torch

        requested_device = os.environ.get("SOFAI_NEW_S1_DEVICE", "").strip().lower()
        if requested_device:
            device = torch.device(requested_device)
        else:
            # S1 is a tiny policy network; on Apple Silicon the CPU path is usually
            # faster than paying MPS transfer overhead for these small tensors.
            device = torch.device("cpu")

        if device.type == "cpu":
            # The policy uses tiny single-item convolutions. Let the benchmark
            # workers provide parallelism instead of oversubscribing CPU cores.
            torch.set_num_threads(max(1, _env_int("SOFAI_NEW_S1_TORCH_THREADS", 1)))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        model_path = _default_model_path()
        model, norm, meta = nl.load_s1_checkpoint(model_path, device)
        args = _make_args(meta)
        _CACHE = {
            "model": model,
            "norm": norm,
            "meta": meta,
            "device": device,
            "model_path": model_path,
            "args": args,
        }
        print(
            f"[S1_nonlinear] ready: model={model_path} device={device} "
            f"dt_nom={args['dt_nom']} total_steps={args['total_steps']}"
        )
    return _CACHE


def _states_from_traj(traj: np.ndarray) -> Optional[np.ndarray]:
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return None
    return traj[:, :2]


def _confidence(info: Dict[str, Any]) -> float:
    if info.get("solved", False):
        return float(np.clip(1.0 - 0.05 * info.get("final_dist", 0.0), 0.75, 1.0))
    return float(np.clip(0.45 - 0.1 * info.get("final_dist", 10.0), 0.0, 0.45))


def solveMemoryNeural(
    scenario: Any,
    *,
    global_s1_accept_rate: Optional[float] = None,
    return_info: bool = False,
) -> Tuple[Optional[np.ndarray], float] | Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    bundle = _init()
    args = bundle["args"]
    model = bundle["model"]
    norm = bundle["norm"]
    device = bundle["device"]
    goal_tol = nl.scenario_goal_tol(scenario, float(args["goal_tol"]))

    traj, controls, info = nl.rollout_policy(
        model,
        scenario,
        norm,
        device,
        total_steps=int(args["total_steps"]),
        action_hold=int(args["action_hold"]),
        stall_patience=int(args["stall_patience"]),
        stall_tol=float(args["stall_tol"]),
        progress_patience=int(args["progress_patience"]),
        progress_tol=float(args["progress_tol"]),
        dt_nom=float(args["dt_nom"]),
        u_max_nom=float(args["u_max_nom"]),
        collision_margin=float(args["collision_margin"]),
        goal_tol=goal_tol,
        grid_n=int(args["grid_n"]),
        n_steps_nom=int(args["n_steps_nom"]),
        buffer_cells=int(args["buffer_cells"]),
        stop_tol=float(args["stop_tol"]),
        debug=bool(args["debug"]),
    )

    goal = nl.scenario_goal(scenario)
    patch_tol = float(os.environ.get("SOFAI_S1_GOAL_PATCH_TOL", max(goal_tol, 0.75)))
    if traj is not None and len(traj) > 0:
        final_dist = float(np.linalg.norm(traj[-1, :2] - goal[:2]))
        if final_dist <= patch_tol:
            goal_state = np.array(traj[-1], copy=True)
            goal_state[:2] = goal[:2]
            traj = np.vstack([traj, goal_state[None, :]]).astype(np.float32)
            info["solved"] = bool(nl.goal_reached(traj[:, :2], goal, goal_tol))
            info["collision_free"] = bool(nl.collision_free_rectangles(traj[:, :2], nl.scenario_rects(scenario)))
            info["final_dist"] = float(np.linalg.norm(traj[-1, :2] - goal[:2]))

    states = _states_from_traj(traj)
    confidence = _confidence(info)
    out = {
        "source": "S1_neural",
        "used_system": "S1_neural",
        "success": bool(info.get("solved", False)),
        "collision_free": bool(info.get("collision_free", False)),
        "goal_reached": bool(info.get("solved", False)),
        "states": traj.tolist(),
        "inputs": controls.tolist(),
        "dt": float(args["dt_nom"]),
        "confidence": confidence,
        "final_dist": float(info.get("final_dist", float("inf"))),
        "runtime_sec": None,
    }
    return (states, confidence, out) if return_info else (states, confidence)


def solveNeural(scenario: Any, *, return_info: bool = False):
    out = solveMemoryNeural(scenario, return_info=return_info)
    return out


def resetMemoryNeuralCache():
    global _CACHE
    _CACHE = None
