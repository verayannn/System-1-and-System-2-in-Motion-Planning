#!/usr/bin/env python3
"""Run motion-planning benchmarks and save JSONL/CSV metrics.

Run from the repo root:

    python3 run_motion_planning_benchmarks.py \
      --patterns benchmark_dualmp_dense_clutter.json \
      --scenario_ids 0-9 \
      --s1 primitives \
      --s2 mpc \
      --run_type sofai

cd /Users/apple/Desktop/sofai

PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 run_motion_planning_benchmarks.py \
  --patterns benchmark_dualmp_dense_clutter.json \
  --scenario_ids 2 \
  --s1 primitives \
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
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
    "dictionary", "scenario_id", "run_type", "s1", "s2", "status", "timed_out",
    "selected_attempt", "success", "collision_free", "goal_reached",
    "final_goal_error", "path_length", "num_states", "runtime_sec",
    "selected_runtime_sec", "wall_runtime_sec",
    "s1_attempted", "s1_success", "s1_collision_free", "s1_goal_reached",
    "s1_confidence", "s1_runtime_sec", "s1_final_goal_error",
    "s1_path_length", "s1_num_states",
    "s2_attempted", "s2_success", "s2_collision_free", "s2_goal_reached",
    "s2_confidence", "s2_runtime_sec", "s2_final_goal_error",
    "s2_path_length", "s2_num_states", "error_message",
]


def configure_repo(root: Path, mplconfigdir: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", mplconfigdir)
    for path in (root, root / "sofai", root / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


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
            matches = sorted(input_dir.glob(pattern))
            matches.extend(sorted((input_dir / "benchmarks").glob(pattern)))
        found.extend(m.resolve() for m in matches if not m.name.endswith(excluded))
    return list(dict.fromkeys(found))


def load_count(path: Path) -> int:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} does not contain a scenario list")
    return len(data)


def scenario_json(scenario: Any, fallback_id: int) -> Dict[str, Any]:
    return {
        "scenario_id": int(getattr(scenario, "scenario_id", fallback_id)),
        "A_query": jsonable(getattr(scenario, "A", None)),
        "B_query": jsonable(getattr(scenario, "B", None)),
        "rectangles": jsonable([list(r) for r in getattr(scenario, "rects", [])]),
        "start": jsonable(getattr(scenario, "start", None)),
        "goal": jsonable(getattr(scenario, "goal", None)),
        "bounds": jsonable(getattr(scenario, "bounds", None)),
        "u_max": jsonable(getattr(scenario, "u_max", None)),
        "goal_tol": float(getattr(scenario, "goal_tol", 0.5)),
    }


def run_s1(scenario: Any, mode: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if mode == "neural":
        from solvers.S1_memory_neural import solveMemoryNeural
        states, confidence = solveMemoryNeural(scenario, return_info=False)
    else:
        from solvers.S1_motion_primitives import solveMotionPrimitives
        states, confidence = solveMotionPrimitives(scenario)
    return {
        "name": f"s1_{mode}",
        "system": "s1",
        "mode": mode,
        "states": None if states is None else jsonable(states),
        "confidence": float(confidence),
        "runtime_sec": time.perf_counter() - t0,
    }


def run_s2(scenario: Any, mode: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if mode == "cbf":
        from solvers.S2_cbf import solve_CBF_with_info
        out = solve_CBF_with_info(scenario)
    else:
        from solvers.S2_mpc import solve_MPC_with_info
        out = solve_MPC_with_info(scenario)
    states = None if out is None else out.get("states")
    return {
        "name": f"s2_{mode}",
        "system": "s2",
        "mode": mode,
        "states": None if states is None else jsonable(states),
        "inputs": None if out is None or out.get("inputs") is None else jsonable(out.get("inputs")),
        "confidence": 1.0 if states is not None else 0.0,
        "runtime_sec": float(out.get("runtime_sec", time.perf_counter() - t0)) if out is not None else time.perf_counter() - t0,
    }


def add_metrics(attempt: Dict[str, Any], scenario: Any) -> Dict[str, Any]:
    import numpy as np
    from solvers.base.S2_mpc_maze import collision_free_rectangles, goal_reached

    states_raw = attempt.get("states")
    if states_raw is None:
        attempt.update(success=False, collision_free=False, goal_reached=False,
                       final_goal_error=None, path_length=None, num_states=0, correctness=0.0)
        return attempt

    states = np.asarray(states_raw, dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        attempt["states"] = None
        return add_metrics(attempt, scenario)

    xy = states[:, :2]
    collision_free = bool(collision_free_rectangles(xy, scenario.rects))
    reached = bool(goal_reached(xy, scenario.goal, getattr(scenario, "goal_tol", 0.5)))
    path_length = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()) if len(xy) > 1 else 0.0
    final_error = float(math.dist(xy[-1].tolist(), list(scenario.goal)))
    success = bool(collision_free and reached)

    attempt.update(
        success=success,
        collision_free=collision_free,
        goal_reached=reached,
        final_goal_error=final_error,
        path_length=path_length,
        num_states=int(states.shape[0]),
        correctness=1.0 if success else 0.0,
    )
    return attempt


def select_attempt(attempts: List[Dict[str, Any]], run_type: str) -> Optional[Dict[str, Any]]:
    if not attempts:
        return None
    if run_type in {"s1", "s2"}:
        return attempts[0]
    return next((a for a in attempts if a.get("success")), attempts[-1])


def error_result(exc: BaseException, opts: Dict[str, Any], runtime: float = 0.0) -> Dict[str, Any]:
    return {
        "status": "error",
        "dictionary": Path(opts["dictionary"]).name,
        "dictionary_path": str(opts["dictionary"]),
        "scenario_id": int(opts["scenario_id"]),
        "run_type": str(opts["run_type"]),
        "s1": str(opts["s1"]),
        "s2": str(opts["s2"]),
        "attempts": [],
        "selected_attempt": None,
        "success": False,
        "collision_free": False,
        "goal_reached": False,
        "final_goal_error": None,
        "path_length": None,
        "num_states": 0,
        "selected_runtime_sec": None,
        "runtime_sec": runtime,
        "timed_out": False,
        "error_message": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def run_case(opts: Dict[str, Any]) -> Dict[str, Any]:
    configure_repo(Path(opts["root"]), str(opts["mplconfigdir"]))
    from input.input_handler import load_scenarios

    dictionary = Path(opts["dictionary"])
    scenario_id = int(opts["scenario_id"])
    scenario = load_scenarios(str(dictionary))[scenario_id]
    run_type = str(opts["run_type"])
    attempts: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    if run_type in {"s1", "sofai"}:
        s1_attempt = add_metrics(run_s1(scenario, str(opts["s1"])), scenario)
        attempts.append(s1_attempt)
    if run_type == "s2" or (run_type == "sofai" and (opts["run_all_attempts"] or not attempts[0].get("success"))):
        attempts.append(add_metrics(run_s2(scenario, str(opts["s2"])), scenario))

    selected = select_attempt(attempts, run_type)
    return {
        "status": "ok",
        "dictionary": dictionary.name,
        "dictionary_path": str(dictionary),
        "scenario_id": scenario_id,
        "run_type": run_type,
        "s1": str(opts["s1"]),
        "s2": str(opts["s2"]),
        "scenario": scenario_json(scenario, scenario_id),
        "attempts": attempts,
        "selected_attempt": None if selected is None else selected["name"],
        "success": bool(selected and selected.get("success")),
        "collision_free": bool(selected and selected.get("collision_free")),
        "goal_reached": bool(selected and selected.get("goal_reached")),
        "final_goal_error": None if selected is None else selected.get("final_goal_error"),
        "path_length": None if selected is None else selected.get("path_length"),
        "num_states": 0 if selected is None else int(selected.get("num_states", 0)),
        "selected_runtime_sec": None if selected is None else selected.get("runtime_sec"),
        "runtime_sec": time.perf_counter() - t0,
        "timed_out": False,
        "error_message": "",
        "traceback": "",
    }


def worker(opts: Dict[str, Any], queue: Any) -> None:
    try:
        queue.put(run_case(opts))
    except BaseException as exc:
        queue.put(error_result(exc, opts))


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
        return {
            "status": "timeout",
            "dictionary": Path(opts["dictionary"]).name,
            "dictionary_path": str(opts["dictionary"]),
            "scenario_id": int(opts["scenario_id"]),
            "run_type": str(opts["run_type"]),
            "s1": str(opts["s1"]),
            "s2": str(opts["s2"]),
            "attempts": [],
            "selected_attempt": None,
            "success": False,
            "collision_free": False,
            "goal_reached": False,
            "final_goal_error": None,
            "path_length": None,
            "num_states": 0,
            "selected_runtime_sec": None,
            "runtime_sec": wall,
            "wall_runtime_sec": wall,
            "timed_out": True,
            "error_message": f"Timed out after {timeout_sec:.1f}s",
            "traceback": "",
        }

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
        "selected_runtime_sec": "" if result.get("selected_runtime_sec") is None else result.get("selected_runtime_sec"),
        "wall_runtime_sec": result.get("wall_runtime_sec", result.get("runtime_sec", "")),
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


def print_aggregate(results: List[Dict[str, Any]]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]

    def count(key: str) -> int:
        return sum(bool(r.get(key)) for r in ok)

    def rate(num: int) -> float:
        return num / len(ok) if ok else 0.0

    print(
        "[aggregate] "
        f"cases={len(results)} ok={len(ok)} "
        f"timeout={sum(bool(r.get('timed_out')) for r in results)} "
        f"error={sum(r.get('status') == 'error' for r in results)} "
        f"success_rate={rate(count('success')):.3f} "
        f"collision_free_rate={rate(count('collision_free')):.3f} "
        f"goal_reach_rate={rate(count('goal_reached')):.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run motion-planning benchmark dictionaries.")
    p.add_argument("--root", default=Path(__file__).resolve().parent)
    p.add_argument("--input_dir", default="input")
    p.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    p.add_argument("--scenario_ids", default="all")
    p.add_argument("--limit_per_dictionary", type=int, default=0)
    p.add_argument("--s1", choices=["neural", "primitives"], default="primitives")
    p.add_argument("--s2", choices=["cbf", "mpc"], default="mpc")
    p.add_argument("--run_type", choices=["sofai", "s1", "s2"], default="sofai")
    p.add_argument("--run_all_attempts", action="store_true")
    p.add_argument("--timeout_sec", type=float, default=300.0)
    p.add_argument("--same_process", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="Number of benchmark cases to run concurrently.")
    p.add_argument("--mplconfigdir", default="/private/tmp/mpl")
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
        print(
            f"[done {i}/{len(planned)}] {result.get('dictionary')} scenario={result.get('scenario_id')} "
            f"status={result.get('status')} selected={result.get('selected_attempt') or 'none'} "
            f"success={result.get('success', False)} wall_runtime={wall:.2f}s"
        )
        if result.get("status") != "ok" and result.get("error_message"):
            print(f"[message] {result['error_message']}")

    results: List[Dict[str, Any]] = []
    if int(args.workers) <= 1:
        for i, (dictionary, sid) in enumerate(planned, start=1):
            print(f"[run {i}/{len(planned)}] {dictionary.name} scenario={sid}")
            result = run_case_timed(make_opts(dictionary, sid), args.timeout_sec, args.same_process)
            results.append(result)
            print_result(result, i)
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

    jsonl_path, csv_path = write_outputs(out_dir, args.out_prefix, results)
    print_aggregate(results)
    print(f"[write] {jsonl_path}")
    print(f"[write] {csv_path}")


if __name__ == "__main__":
    main()
