#!/usr/bin/env python3
"""Run motion-planning benchmarks and save JSONL/CSV metrics.

Run from the repo root:


PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 run_motion_planning_benchmarks.py \
  --patterns benchmark_dualmp_dense_clutter.json \
  --scenario_ids 2 \
  --s1 neural \
  --s2 mpc \
  --run_type s1 \
  --timeout_sec 60 \
  --out_dir output/benchmark_runs/check \
  --out_prefix dense_clutter_sc2_s1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_PATTERNS = [
    "benchmark_dualmp_small_open.json",
    "benchmark_dualmp_large_sparse.json",
    "benchmark_dualmp_dense_clutter.json",
    "benchmark_dualmp_wall_gap.json",
    "benchmark_dualmp_serial_walls.json",
    "benchmark_dualmp_maze_branching.json",
    "benchmark_dualmp_bugtrap.json",
]

CSV_FIELDS = [
    "dictionary", "scenario_index", "scenario_id", "run_type", "s1", "s2", "status", "timed_out",
    "selected_attempt", "success", "collision_free", "goal_reached",
    "final_goal_error", "path_length", "num_states", "runtime_sec", "planning_runtime_sec",
    "selected_runtime_sec", "wall_runtime_sec",
    "quality_score", "quality_path_efficiency", "quality_smoothness", "quality_clearance",
    "quality_path_length", "quality_reference_path_length", "quality_min_clearance",
    "quality_sparc", "quality_ldlj", "quality_smoothness_ldlj",
    "quality_duration_sec", "quality_mean_speed",
    "quality_peak_control_ratio", "quality_control_saturation_frac",
    "quality_definition", "quality_family",
    "s1_attempted", "s1_success", "s1_collision_free", "s1_goal_reached",
    "s1_confidence", "s1_runtime_sec", "s1_final_goal_error",
    "s1_path_length", "s1_num_states",
    "s2_attempted", "s2_success", "s2_collision_free", "s2_goal_reached",
    "s2_confidence", "s2_runtime_sec", "s2_final_goal_error",
    "s2_path_length", "s2_num_states", "error_message",
]


def configure_repo(root: Path, mplconfigdir: str) -> None:
    for path in (root, root / "sofai", root / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from solvers._s2_common import resolve_mplconfigdir

    os.environ["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, mplconfigdir))


def jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return str(x)


def parse_ids(raw: str, count: int) -> List[int]:
    raw = str(raw).strip()
    if raw.lower() in {"", "all"}:
        return list(range(count))

    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(v.strip()) for v in part.split("-", 1)]
            lo, hi = sorted((a, b))
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(part))

    seen = set()
    return [i for i in ids if 0 <= i < count and not (i in seen or seen.add(i))]


def discover(input_dir: Path, patterns: Sequence[str]) -> List[Path]:
    excluded = ("_manifest.json", "_query.json", "_results.json", "_summary.json")
    found: List[Path] = []
    for pattern in patterns:
        p = Path(pattern).expanduser()
        if p.is_absolute():
            matches = sorted(p.parent.glob(p.name))
        else:
            matches = sorted(input_dir.rglob(pattern))
            matches.extend(sorted((input_dir / "benchmarks").rglob(pattern)))
            matches.extend(sorted((input_dir / "nl").rglob(pattern)))
        found.extend(m.resolve() for m in matches if not m.name.endswith(excluded))
    return list(dict.fromkeys(found))


def load_count(path: Path) -> int:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} does not contain a scenario list")
    return len(data)


def error_result(exc: BaseException, opts: Dict[str, Any], runtime: float = 0.0) -> Dict[str, Any]:
    problem_name = f"{Path(opts['dictionary']).stem}_sc_{int(opts['scenario_id'])}"
    return {
        "status": "error",
        "problem_name": problem_name,
        "dictionary": Path(opts["dictionary"]).name,
        "dictionary_path": str(opts["dictionary"]),
        "scenario_index": int(opts["scenario_id"]),
        "scenario_id": int(opts["scenario_id"]),
        "run_type": str(opts["run_type"]),
        "s1": str(opts["s1"]),
        "s2": str(opts["s2"]),
        "scenario": None,
        "attempts": [],
        "selected_attempt": None,
        "success": False,
        "collision_free": False,
        "goal_reached": False,
        "final_goal_error": None,
        "path_length": None,
        "num_states": 0,
        "selected_runtime_sec": None,
        "planning_runtime_sec": runtime,
        "runtime_sec": runtime,
        "running_time": runtime,
        "timed_out": False,
        "error_message": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "solution_raw": None,
        "solution": "noSolution",
        "confidence": 0.0,
        "correctness": 0.0,
        "meta": {
            "problem_name": problem_name,
            "scenario": None,
            "solution_raw": None,
            "solution": "noSolution",
            "confidence": 0.0,
            "correctness": 0.0,
            "running_time": runtime,
        },
    }

## load scenarios and run
def run_case(opts: Dict[str, Any]) -> Dict[str, Any]:
    configure_repo(Path(opts["root"]), str(opts["mplconfigdir"]))
    from motion_planning_solver import solve_benchmark_case

    result = solve_benchmark_case(
        opts["dictionary"],
        int(opts["scenario_id"]),
        s1=str(opts["s1"]),
        s2=str(opts["s2"]),
        run_type=str(opts["run_type"]),
        run_all_attempts=bool(opts["run_all_attempts"]),
    )
    result["scenario_index"] = int(opts["scenario_id"])
    return result


def worker(opts: Dict[str, Any], queue: Any) -> None:
    try:
        queue.put(run_case(opts))
    except BaseException as exc:
        queue.put(error_result(exc, opts))


def timeout_result(opts: Dict[str, Any], timeout_sec: float, wall: float) -> Dict[str, Any]:
    problem_name = f"{Path(opts['dictionary']).stem}_sc_{int(opts['scenario_id'])}"
    return {
        "status": "timeout",
        "problem_name": problem_name,
        "dictionary": Path(opts["dictionary"]).name,
        "dictionary_path": str(opts["dictionary"]),
        "scenario_index": int(opts["scenario_id"]),
        "scenario_id": int(opts["scenario_id"]),
        "run_type": str(opts["run_type"]),
        "s1": str(opts["s1"]),
        "s2": str(opts["s2"]),
        "scenario": None,
        "attempts": [],
        "selected_attempt": None,
        "success": False,
        "collision_free": False,
        "goal_reached": False,
        "final_goal_error": None,
        "path_length": None,
        "num_states": 0,
        "selected_runtime_sec": None,
        "planning_runtime_sec": wall,
        "runtime_sec": wall,
        "running_time": wall,
        "wall_runtime_sec": wall,
        "timed_out": True,
        "error_message": f"Timed out after {timeout_sec:.1f}s",
        "traceback": "",
        "solution_raw": None,
        "solution": "noSolution",
        "confidence": 0.0,
        "correctness": 0.0,
        "meta": {
            "problem_name": problem_name,
            "scenario": None,
            "solution_raw": None,
            "solution": "noSolution",
            "confidence": 0.0,
            "correctness": 0.0,
            "running_time": wall,
        },
    }


def persistent_worker(worker_id: int, task_queue: Any, result_queue: Any) -> None:
    """Run benchmark cases serially so solver module caches survive each case."""
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id, opts = task
        try:
            result = run_case(opts)
        except BaseException as exc:
            result = error_result(exc, opts)
        result_queue.put((worker_id, task_id, result))


def run_cases_persistent(
    opts_list: Sequence[Dict[str, Any]],
    *,
    timeout_sec: float,
    workers: int,
    on_result: Callable[[Dict[str, Any], int], None],
) -> List[Dict[str, Any]]:
    """Use persistent spawned workers with parent-enforced per-case timeouts.

    Each worker imports and caches the S1 model only once. If a case times out,
    the parent kills just that worker and starts a replacement, preserving hard
    timeout semantics while limiting a reload to the replacement worker.
    """
    if not opts_list:
        return []

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    worker_count = min(max(1, int(workers)), len(opts_list))
    states: List[Dict[str, Any]] = []

    def start_worker(worker_id: int) -> Dict[str, Any]:
        task_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=persistent_worker, args=(worker_id, task_queue, result_queue))
        proc.start()
        return {"id": worker_id, "process": proc, "task_queue": task_queue, "active": None}

    def stop_worker(state: Dict[str, Any]) -> None:
        proc = state["process"]
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)
        state["task_queue"].close()

    states = [start_worker(worker_id) for worker_id in range(worker_count)]
    next_task_id = 0
    completed = 0
    results: List[Dict[str, Any]] = []

    def dispatch(state: Dict[str, Any]) -> bool:
        nonlocal next_task_id
        if next_task_id >= len(opts_list):
            return False
        task_id = next_task_id
        opts = opts_list[task_id]
        state["task_queue"].put((task_id, opts))
        state["active"] = (task_id, opts, time.perf_counter())
        next_task_id += 1
        return True

    for state in states:
        dispatch(state)

    try:
        while completed < len(opts_list):
            now = time.perf_counter()
            for index, state in enumerate(states):
                active = state["active"]
                proc = state["process"]
                if active is None:
                    if not proc.is_alive():
                        states[index] = start_worker(state["id"])
                    dispatch(states[index])
                    continue

                task_id, opts, started = active
                wall = now - started
                if proc.is_alive() and wall < timeout_sec:
                    continue

                if proc.is_alive():
                    stop_worker(state)
                    result = timeout_result(opts, timeout_sec, wall)
                else:
                    state["task_queue"].close()
                    result = error_result(
                        RuntimeError(f"persistent worker exited with code {proc.exitcode}"),
                        opts,
                        wall,
                    )
                result["wall_runtime_sec"] = wall
                results.append(result)
                completed += 1
                on_result(result, completed)
                states[index] = start_worker(state["id"])
                dispatch(states[index])

            try:
                worker_id, task_id, result = result_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            state = states[int(worker_id)]
            active = state["active"]
            if active is None or int(active[0]) != int(task_id):
                # Result from a worker that was terminated after its deadline.
                continue
            wall = time.perf_counter() - float(active[2])
            result["wall_runtime_sec"] = wall
            state["active"] = None
            results.append(result)
            completed += 1
            on_result(result, completed)
            dispatch(state)
    finally:
        for state in states:
            proc = state["process"]
            if proc.is_alive():
                state["task_queue"].put(None)
                proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            state["task_queue"].close()
        result_queue.close()

    return results


def run_case_timed(opts: Dict[str, Any], timeout_sec: float, same_process: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if same_process or timeout_sec <= 0:
        try:
            result = run_case(opts)
        except BaseException as exc:
            result = error_result(exc, opts, time.perf_counter() - t0)
        result["wall_runtime_sec"] = time.perf_counter() - t0
        return result

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=worker, args=(opts, queue))
    proc.start()
    proc.join(float(timeout_sec))
    wall = time.perf_counter() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return timeout_result(opts, timeout_sec, wall)

    try:
        result = queue.get_nowait()
    except Exception as exc:
        result = error_result(RuntimeError(f"child exited with code {proc.exitcode}: {exc}"), opts, wall)
    result["wall_runtime_sec"] = wall
    return result


def find_attempt(result: Dict[str, Any], system: str) -> Optional[Dict[str, Any]]:
    return next((a for a in result.get("attempts", []) if a.get("system") == system), None)


def flat(result: Dict[str, Any]) -> Dict[str, Any]:
    s1 = find_attempt(result, "s1")
    s2 = find_attempt(result, "s2")

    def get(attempt: Optional[Dict[str, Any]], key: str) -> Any:
        if attempt is None:
            return ""
        value = attempt.get(key, "")
        return "" if value is None else value

    return {
        "dictionary": result.get("dictionary", ""),
        "scenario_index": result.get("scenario_index", result.get("scenario_id", "")),
        "scenario_id": result.get("scenario_id", ""),
        "run_type": result.get("run_type", ""),
        "s1": result.get("s1", ""),
        "s2": result.get("s2", ""),
        "status": result.get("status", ""),
        "timed_out": bool(result.get("timed_out", False)),
        "selected_attempt": result.get("selected_attempt") or "",
        "success": bool(result.get("success", False)),
        "collision_free": bool(result.get("collision_free", False)),
        "goal_reached": bool(result.get("goal_reached", False)),
        "final_goal_error": "" if result.get("final_goal_error") is None else result.get("final_goal_error"),
        "path_length": "" if result.get("path_length") is None else result.get("path_length"),
        "num_states": result.get("num_states", 0),
        "runtime_sec": result.get("runtime_sec", ""),
        "planning_runtime_sec": result.get("planning_runtime_sec", ""),
        "selected_runtime_sec": "" if result.get("selected_runtime_sec") is None else result.get("selected_runtime_sec"),
        "wall_runtime_sec": result.get("wall_runtime_sec", result.get("runtime_sec", "")),
        **{
            column: "" if result.get(column) is None else result.get(column)
            for column in (
                "quality_score", "quality_path_efficiency", "quality_smoothness", "quality_clearance",
                "quality_path_length", "quality_reference_path_length", "quality_min_clearance",
                "quality_sparc", "quality_ldlj", "quality_smoothness_ldlj",
                "quality_duration_sec", "quality_mean_speed",
                "quality_peak_control_ratio", "quality_control_saturation_frac",
            )
        },
        "quality_definition": result.get("quality_definition", ""),
        "quality_family": result.get("quality_family", ""),
        "s1_attempted": s1 is not None,
        "s1_success": get(s1, "success"),
        "s1_collision_free": get(s1, "collision_free"),
        "s1_goal_reached": get(s1, "goal_reached"),
        "s1_confidence": get(s1, "confidence"),
        "s1_runtime_sec": get(s1, "runtime_sec"),
        "s1_final_goal_error": get(s1, "final_goal_error"),
        "s1_path_length": get(s1, "path_length"),
        "s1_num_states": get(s1, "num_states"),
        "s2_attempted": s2 is not None,
        "s2_success": get(s2, "success"),
        "s2_collision_free": get(s2, "collision_free"),
        "s2_goal_reached": get(s2, "goal_reached"),
        "s2_confidence": get(s2, "confidence"),
        "s2_runtime_sec": get(s2, "runtime_sec"),
        "s2_final_goal_error": get(s2, "final_goal_error"),
        "s2_path_length": get(s2, "path_length"),
        "s2_num_states": get(s2, "num_states"),
        "error_message": result.get("error_message", ""),
    }


def write_outputs(out_dir: Path, prefix: str, results: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{prefix}_runs.jsonl"
    csv_path = out_dir / f"{prefix}_summary.csv"

    with jsonl_path.open("w") as f:
        for result in results:
            f.write(json.dumps(jsonable(result)) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(flat(result))

    return jsonl_path, csv_path


def annotate_quality(results: List[Dict[str, Any]]) -> Dict[str, float]:
    from solvers._s2_common import (
        QUALITY_DEFINITION_VERSION,
        benchmark_family_from_dictionary,
        quality_score,
        trajectory_quality_components,
    )

    # Index components first, then the diagnostics that stay outside the index.
    quality_fields = (
        ("quality_score", None),
        ("quality_path_efficiency", "path_efficiency"),
        ("quality_smoothness", "smoothness"),
        ("quality_clearance", "clearance_score"),
        ("quality_path_length", "path_length"),
        ("quality_reference_path_length", "reference_path_length"),
        ("quality_min_clearance", "min_clearance"),
        ("quality_sparc", "sparc"),
        ("quality_ldlj", "ldlj"),
        ("quality_smoothness_ldlj", "smoothness_ldlj"),
        ("quality_duration_sec", "duration_sec"),
        ("quality_mean_speed", "mean_speed"),
        ("quality_peak_control_ratio", "peak_control_ratio"),
        ("quality_control_saturation_frac", "control_saturation_frac"),
    )

    family = benchmark_family_from_dictionary(results[0].get("dictionary", "")) if results else ""
    for result in results:
        sample = trajectory_quality_components(result) if bool(result.get("success", False)) else None
        result["quality_family"] = family
        result["quality_definition"] = QUALITY_DEFINITION_VERSION
        for column, key in quality_fields:
            if sample is None:
                result[column] = None
            elif key is None:
                result[column] = quality_score(sample)
            else:
                value = sample.get(key)
                result[column] = float(value) if value is not None and math.isfinite(float(value)) else None

    return {}


def print_aggregate(results: List[Dict[str, Any]]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]

    def count(key: str) -> int:
        return sum(bool(r.get(key)) for r in ok)

    def rate(num: int) -> float:
        return num / len(ok) if ok else 0.0

    runtimes = [float(r["planning_runtime_sec"]) for r in ok if r.get("planning_runtime_sec") is not None]
    qualities = [float(r["quality_score"]) for r in ok if r.get("success") and r.get("quality_score") is not None]
    s1_success = 0
    s2_only_success = 0
    for result in ok:
        attempts = result.get("attempts", []) or []
        if any(a.get("system") == "s1" and a.get("success") for a in attempts):
            s1_success += 1
        elif any(a.get("system") == "s2" and a.get("success") for a in attempts):
            s2_only_success += 1

    print(
        "[aggregate] "
        f"cases={len(results)} ok={len(ok)} "
        f"timeout={sum(bool(r.get('timed_out')) for r in results)} "
        f"error={sum(r.get('status') == 'error' for r in results)} "
        f"success_rate={rate(count('success')):.3f} "
        f"collision_free_rate={rate(count('collision_free')):.3f} "
        f"goal_reach_rate={rate(count('goal_reached')):.3f} "
        f"mean_planning_runtime_sec={float(np.mean(runtimes)) if runtimes else float('nan'):.3f} "
        f"mean_quality={float(np.mean(qualities)) if qualities else float('nan'):.3f} "
        f"s1_success={s1_success} s2_only_success={s2_only_success}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run motion-planning benchmark dictionaries.")
    p.add_argument("--root", default=Path(__file__).resolve().parent)
    p.add_argument("--input_dir", default="input")
    p.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    p.add_argument("--scenario_ids", default="all")
    p.add_argument("--limit_per_dictionary", type=int, default=0)
    p.add_argument("--stop_after_successes", type=int, default=0, help="Stop sequential execution after this many successful cases; 0 runs every selected case.")
    p.add_argument("--s1", choices=["neural", "primitives"], default="primitives")
    p.add_argument("--s2", choices=["cbf", "mpc", "mpc_warm", "mpc_do"], default="mpc")
    p.add_argument("--run_type", choices=["sofai", "s1", "s2"], default="sofai")
    p.add_argument("--run_all_attempts", action="store_true")
    p.add_argument("--timeout_sec", type=float, default=60.0)
    p.add_argument(
        "--same_process",
        action="store_true",
        help="Run cases directly in this process. This disables hard timeout enforcement; omit it to use cached persistent workers.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent persistent benchmark workers; each loads and caches solver assets once.",
    )
    p.add_argument("--mplconfigdir", default="")
    p.add_argument("--out_dir", default="output/benchmark_runs")
    p.add_argument("--out_prefix", default="benchmark_dualmp")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--new_run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--python", default=sys.executable, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    input_dir = input_dir if input_dir.is_absolute() else root / input_dir
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir

    configure_repo(root, args.mplconfigdir)

    dictionaries = discover(input_dir, args.patterns)
    if not dictionaries:
        raise SystemExit(f"No dictionaries matched {args.patterns} in {input_dir}")

    planned: List[Tuple[Path, int]] = []
    for dictionary in dictionaries:
        ids = parse_ids(args.scenario_ids, load_count(dictionary))
        ids = ids[: args.limit_per_dictionary] if args.limit_per_dictionary > 0 else ids
        print(f"[dict] {dictionary.name} scenarios_to_run={len(ids)}")
        planned.extend((dictionary, sid) for sid in ids)

    if args.dry_run:
        for dictionary, sid in planned:
            print(f"[dry_run] {dictionary.name} scenario={sid} run_type={args.run_type} s1={args.s1} s2={args.s2}")
        print(f"[dry_run] total_cases={len(planned)}")
        return

    if args.stop_after_successes > 0 and int(args.workers) > 1:
        raise SystemExit("--stop_after_successes requires --workers 1 so no extra cases are submitted.")

    def make_opts(dictionary: Path, sid: int) -> Dict[str, Any]:
        return {
            "root": str(root),
            "dictionary": str(dictionary),
            "scenario_id": sid,
            "s1": args.s1,
            "s2": args.s2,
            "run_type": args.run_type,
            "run_all_attempts": bool(args.run_all_attempts),
            "mplconfigdir": args.mplconfigdir,
        }

    def print_result(result: Dict[str, Any], i: int) -> None:
        wall = float(result.get("wall_runtime_sec", result.get("runtime_sec", 0.0)))
        run_type = str(result.get("run_type", ""))
        if run_type == "sofai":
            if result.get("s2_skipped", False):
                s2_state = "skipped"
            elif result.get("s2_attempted", False):
                s2_state = "used"
            else:
                s2_state = "not_run"
        elif run_type == "s2":
            s2_state = "selected"
        else:
            s2_state = "n/a"
        print(
            f"[done {i}/{len(planned)}] {result.get('dictionary')} scenario={result.get('scenario_id')} "
            f"status={result.get('status')} selected={result.get('selected_attempt') or 'none'} "
            f"s2={s2_state} success={result.get('success', False)} wall_runtime={wall:.2f}s"
        )
        if result.get("status") != "ok" and result.get("error_message"):
            print(f"[message] {result['error_message']}")

    results: List[Dict[str, Any]] = []
    use_persistent_workers = not args.same_process and args.timeout_sec > 0 and args.stop_after_successes == 0
    if use_persistent_workers:
        worker_count = min(max(1, int(args.workers)), len(planned))
        print(
            f"[persistent] workers={worker_count} cases={len(planned)} "
            f"timeout_sec={args.timeout_sec:g}; solver caches persist per worker"
        )
        results = run_cases_persistent(
            [make_opts(dictionary, sid) for dictionary, sid in planned],
            timeout_sec=float(args.timeout_sec),
            workers=worker_count,
            on_result=print_result,
        )
    elif int(args.workers) <= 1:
        for i, (dictionary, sid) in enumerate(planned, start=1):
            print(f"[run {i}/{len(planned)}] {dictionary.name} scenario={sid}")
            result = run_case_timed(make_opts(dictionary, sid), args.timeout_sec, args.same_process)
            results.append(result)
            print_result(result, i)
            if args.stop_after_successes > 0 and sum(bool(row.get("success")) for row in results) >= args.stop_after_successes:
                print(f"[stop] reached {args.stop_after_successes} successful cases after {i} attempted cases")
                break
    else:
        print(f"[parallel] workers={args.workers} cases={len(planned)}")
        with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {
                pool.submit(run_case_timed, make_opts(dictionary, sid), args.timeout_sec, args.same_process): (dictionary, sid)
                for dictionary, sid in planned
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                print_result(result, i)

    annotate_quality(results)
    jsonl_path, csv_path = write_outputs(out_dir, args.out_prefix, results)
    print_aggregate(results)
    print(f"[write] {jsonl_path}")
    print(f"[write] {csv_path}")


if __name__ == "__main__":
    main()
