"""
Solver-facing wrapper for the memory + neural System 1.

Keep this file small and System-2 independent, matching the role of
S1_motion_primitives.py. The heavy neural rollout and episodic-memory retrieval
helpers are imported from Solvers/Base/S1_S2_continual_maze.py, but this wrapper
does not call that module's S2 fallback or continual-retraining entry points.

"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from solvers.base import S1_S2_continual_maze as base


SOLVER_DIR = Path(__file__).resolve().parent
LEGACY_S1_DIR = Path(
    os.environ.get("SOFAI_LEGACY_S1_DIR", "/Users/apple/Desktop/S1:2 codes/maze/n_s1")
).expanduser()

_CACHE: Optional[Dict[str, Any]] = None


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
            SOLVER_DIR / "s1_policy_full_retrain_latest.pth",
            LEGACY_S1_DIR / "paper_1199_full_retrain_s2only" / "s1_policy_full_retrain_latest.pth",
            SOLVER_DIR / "s1_policy_control_cnn_diverse_5k.pth",
            LEGACY_S1_DIR / "s1_policy_control_cnn_diverse_5k.pth",
        ],
        env_name="SOFAI_NEW_S1_MODEL",
        required=True,
    )


def _default_base_dataset() -> Path:
    return _first_existing(
        [
            SOLVER_DIR / "nn_dataset_maze_diverse_5k.npz",
            LEGACY_S1_DIR / "nn_dataset_maze_diverse_5k.npz",
        ],
        env_name="SOFAI_NEW_S1_BASE_DATASET",
    )


def _default_base_memory_traj() -> Path:
    return _first_existing(
        [
            SOLVER_DIR / "s1_sfcbf_success_trajs_diverse_5k.npz",
            LEGACY_S1_DIR / "s1_sfcbf_success_trajs_diverse_5k.npz",
        ],
        env_name="SOFAI_NEW_S1_BASE_MEMORY_TRAJ",
    )


def _default_base_memory_scenarios() -> Path:
    return _first_existing(
        [
            SOLVER_DIR / "benchmark_scenarios_maze_diverse_5k.json",
            LEGACY_S1_DIR / "benchmark_scenarios_maze_diverse_5k.json",
        ],
        env_name="SOFAI_NEW_S1_BASE_MEMORY_SCENARIOS",
    )


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        s1_steps=_env_int("SOFAI_NEW_S1_STEPS", 120),
        dt=_env_float("SOFAI_NEW_S1_DT", 0.05),
        u_max=_env_float("SOFAI_NEW_S1_U_MAX", 3.0),
        goal_tol=_env_float("SOFAI_NEW_S1_GOAL_TOL", 0.5),
        collision_margin=_env_float("SOFAI_NEW_S1_COLLISION_MARGIN", 0.05),
        grid_n=_env_int("SOFAI_NEW_S1_GRID_N", 25),
        n_steps_nom=_env_int("SOFAI_NEW_S1_N_STEPS_NOM", 200),
        buffer_cells=_env_int("SOFAI_NEW_S1_BUFFER_CELLS", 2),
        stop_tol=_env_float("SOFAI_NEW_S1_STOP_TOL", 0.6),
        enable_confidence_switch=_env_bool("SOFAI_NEW_S1_ENABLE_CONFIDENCE_SWITCH", True),
        mc_dropout_samples=_env_int("SOFAI_NEW_S1_MC_DROPOUT_SAMPLES", 4),
        local_uncertainty_scale=_env_float("SOFAI_NEW_S1_LOCAL_UNCERTAINTY_SCALE", 0.25),
        confidence_global_weight=_env_float("SOFAI_NEW_S1_CONFIDENCE_GLOBAL_WEIGHT", 0.35),
        confidence_threshold=_env_float("SOFAI_NEW_S1_CONFIDENCE_THRESHOLD", 0.35),
        confidence_patience=_env_int("SOFAI_NEW_S1_CONFIDENCE_PATIENCE", 3),
        confidence_min_steps=_env_int("SOFAI_NEW_S1_CONFIDENCE_MIN_STEPS", 5),
        initial_global_s1_accept_rate=_env_float("SOFAI_NEW_S1_GLOBAL_ACCEPT_RATE", 0.5),
        enable_episodic_memory=_env_bool("SOFAI_NEW_S1_ENABLE_MEMORY", True),
        episodic_memory_path=os.environ.get("SOFAI_NEW_S1_MEMORY_PATH", ""),
        resume_episodic_memory=_env_bool("SOFAI_NEW_S1_RESUME_MEMORY", False),
        episodic_memory_try_before_neural=_env_bool("SOFAI_NEW_S1_MEMORY_BEFORE_NN", True),
        # This solver only consumes memory. Storing new S2 trajectories belongs
        # to the outer SOFAI solver or an offline continual-learning script.
        episodic_memory_store_s2_success=False,
        use_base_episodic_memory=_env_bool("SOFAI_NEW_S1_USE_BASE_MEMORY", True),
        base_dataset=str(_default_base_dataset()),
        base_memory_traj_npz=str(_default_base_memory_traj()),
        base_memory_scenarios=str(_default_base_memory_scenarios()),
        base_memory_max_items=_env_int("SOFAI_NEW_S1_BASE_MEMORY_MAX_ITEMS", 0),
        episodic_memory_top_k=_env_int("SOFAI_NEW_S1_MEMORY_TOP_K", 5),
        episodic_memory_score_threshold=_env_float("SOFAI_NEW_S1_MEMORY_SCORE_THRESHOLD", 0.65),
        episodic_memory_map_threshold=_env_float("SOFAI_NEW_S1_MEMORY_MAP_THRESHOLD", 0.45),
        episodic_memory_dyn_sigma=_env_float("SOFAI_NEW_S1_MEMORY_DYN_SIGMA", 0.45),
        episodic_memory_replay_steps=_env_int("SOFAI_NEW_S1_MEMORY_REPLAY_STEPS", 0),
        episodic_memory_max_traj_steps=_env_int("SOFAI_NEW_S1_MEMORY_MAX_TRAJ_STEPS", 0),
        episodic_memory_max_per_bucket=_env_int("SOFAI_NEW_S1_MEMORY_MAX_PER_BUCKET", 25),
        episodic_memory_max_total=_env_int("SOFAI_NEW_S1_MEMORY_MAX_TOTAL", 2000),
    )


def _init():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    import torch

    args = _make_args()
    model_path = _default_model_path()

    requested_device = os.environ.get("SOFAI_NEW_S1_DEVICE", "").strip()
    if requested_device:
        device = torch.device(requested_device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, norm, L_c, meta = base.load_s1_model(model_path, device)

    episodic_memory = None
    if args.enable_episodic_memory:
        memory_path = Path(args.episodic_memory_path).expanduser() if args.episodic_memory_path else None
        if memory_path is not None and args.resume_episodic_memory:
            episodic_memory = base.load_episodic_memory(memory_path)
        else:
            episodic_memory = base.empty_episodic_memory()
        base.add_base_trajectories_to_episodic_memory(episodic_memory, args)

    _CACHE = {
        "args": args,
        "device": device,
        "model": model,
        "norm": norm,
        "L_c": L_c,
        "meta": meta,
        "model_path": model_path,
        "episodic_memory": episodic_memory,
    }
    print(f"[S1_memory_neural] ready: model={model_path} device={device}")
    return _CACHE


def _scenario_to_dict(scenario: Any) -> Dict[str, Any]:
    if isinstance(scenario, dict):
        sc = dict(scenario)
    else:
        sc = {
            "scenario_id": int(getattr(scenario, "scenario_id", -1)),
            "A_query": getattr(scenario, "A"),
            "B_query": getattr(scenario, "B"),
            "rectangles": getattr(scenario, "rects"),
            "start": getattr(scenario, "start"),
            "goal": getattr(scenario, "goal"),
            "bounds": getattr(scenario, "bounds"),
            "u_max": getattr(scenario, "u_max", 3.0),
            "goal_tol": getattr(scenario, "goal_tol", 0.5),
        }

    if "A_query" not in sc and "A" in sc:
        sc["A_query"] = sc["A"]
    if "B_query" not in sc and "B" in sc:
        sc["B_query"] = sc["B"]
    return sc


def _states_from_output(out: Dict[str, Any]) -> Optional[np.ndarray]:
    states = np.asarray(out.get("states", []), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        return None
    return states[:, :2]


def _clip01(value: Any) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.0


def _confidence_from_output(out: Dict[str, Any]) -> float:
    if out.get("source") == "S1_memory":
        score = _clip01(out.get("episodic_memory_score", 0.0))
        return max(0.75, score) if out.get("success", False) else min(0.35, score)

    conf = _clip01(out.get("combined_confidence_mean", out.get("local_confidence_mean", 0.0)))
    if out.get("confidence_triggered", False):
        return min(conf, 0.35)
    if out.get("success", False):
        return max(0.75, conf)
    return min(0.45, conf)


def _better_failed(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return new
    old_dist = float(old.get("final_dist", float("inf")))
    new_dist = float(new.get("final_dist", float("inf")))
    return new if new_dist < old_dist else old


def solveMemoryNeural(
    scenario: Any,
    *,
    global_s1_accept_rate: Optional[float] = None,
    return_info: bool = False,
) -> Tuple[Optional[np.ndarray], float] | Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    """
    Run memory-assisted neural System 1 on one maze scenario.

    Returns:
        states, confidence
        or states, confidence, info when return_info=True.
    """
    bundle = _init()
    args = bundle["args"]
    sc = _scenario_to_dict(scenario)
    global_accept = (
        float(global_s1_accept_rate)
        if global_s1_accept_rate is not None
        else float(args.initial_global_s1_accept_rate)
    )

    best_failed = None
    info: Dict[str, Any] = {"source": "none", "success": False}

    if args.enable_episodic_memory and args.episodic_memory_try_before_neural:
        mem_out = base.run_episodic_memory_on_scenario(sc, bundle["episodic_memory"], args)
        if mem_out.get("attempted", False):
            best_failed = _better_failed(best_failed, mem_out)
            if mem_out.get("success", False):
                states = _states_from_output(mem_out)
                confidence = _confidence_from_output(mem_out)
                return (states, confidence, mem_out) if return_info else (states, confidence)

    s1_out = base.run_s1_on_scenario(
        sc=sc,
        model=bundle["model"],
        norm=bundle["norm"],
        L_c=bundle["L_c"],
        device=bundle["device"],
        args=args,
        global_s1_accept_rate=global_accept,
    )
    best_failed = _better_failed(best_failed, s1_out)
    if s1_out.get("success", False):
        states = _states_from_output(s1_out)
        confidence = _confidence_from_output(s1_out)
        return (states, confidence, s1_out) if return_info else (states, confidence)

    if args.enable_episodic_memory and not args.episodic_memory_try_before_neural:
        mem_out = base.run_episodic_memory_on_scenario(sc, bundle["episodic_memory"], args)
        if mem_out.get("attempted", False):
            best_failed = _better_failed(best_failed, mem_out)
            if mem_out.get("success", False):
                states = _states_from_output(mem_out)
                confidence = _confidence_from_output(mem_out)
                return (states, confidence, mem_out) if return_info else (states, confidence)

    if best_failed is not None:
        info = best_failed
        states = _states_from_output(best_failed)
        confidence = _confidence_from_output(best_failed)
    else:
        states = None
        confidence = 0.0

    return (states, confidence, info) if return_info else (states, confidence)


def resetMemoryNeuralCache():
    global _CACHE
    _CACHE = None
