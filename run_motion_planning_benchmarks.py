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
    "quality_path_length", "quality_control_effort", "quality_smoothness",
    "quality_j", "quality_score",
    "quality_path_length_ref", "quality_control_effort_ref", "quality_smoothness_ref",
    "quality_weight_path_length", "quality_weight_control_effort", "quality_weight_smoothness",
    "quality_family",
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
        "quality_path_length": "" if result.get("quality_path_length") is None else result.get("quality_path_length"),
        "quality_control_effort": "" if result.get("quality_control_effort") is None else result.get("quality_control_effort"),
        "quality_smoothness": "" if result.get("quality_smoothness") is None else result.get("quality_smoothness"),
        "quality_j": "" if result.get("quality_j") is None else result.get("quality_j"),
        "quality_score": "" if result.get("quality_score") is None else result.get("quality_score"),
        "quality_path_length_ref": "" if result.get("quality_path_length_ref") is None else result.get("quality_path_length_ref"),
        "quality_control_effort_ref": "" if result.get("quality_control_effort_ref") is None else result.get("quality_control_effort_ref"),
        "quality_smoothness_ref": "" if result.get("quality_smoothness_ref") is None else result.get("quality_smoothness_ref"),
        "quality_weight_path_length": "" if result.get("quality_weight_path_length") is None else result.get("quality_weight_path_length"),
        "quality_weight_control_effort": "" if result.get("quality_weight_control_effort") is None else result.get("quality_weight_control_effort"),
        "quality_weight_smoothness": "" if result.get("quality_weight_smoothness") is None else result.get("quality_weight_smoothness"),
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
        benchmark_family_from_dictionary,
        quality_refs_for_result,
        quality_score,
        quality_weights_for_family,
        trajectory_quality_components,
    )

    per_result: List[Dict[str, float] | None] = []
    family = benchmark_family_from_dictionary(results[0].get("dictionary", "")) if results else ""
    weights = quality_weights_for_family(family)
    for result in results:
        sample = trajectory_quality_components(result) if bool(result.get("success", False)) else None
        per_result.append(sample)

    for result, sample in zip(results, per_result):
        refs = quality_refs_for_result(result)
        result["quality_path_length_ref"] = refs["path_length"]
        result["quality_control_effort_ref"] = refs["control_effort"]
        result["quality_smoothness_ref"] = refs["smoothness"]
        result["quality_weight_path_length"] = weights["path_length"]
        result["quality_weight_control_effort"] = weights["control_effort"]
        result["quality_weight_smoothness"] = weights["smoothness"]
        result["quality_family"] = family
        if sample is None:
            result["quality_path_length"] = None
            result["quality_control_effort"] = None
            result["quality_smoothness"] = None
            result["quality_j"] = None
            result["quality_score"] = None
            continue
        j = (
            weights["path_length"] * float(sample["path_length"]) / float(refs["path_length"])
            + weights["control_effort"] * float(sample["control_effort"]) / float(refs["control_effort"])
            + weights["smoothness"] * float(sample["smoothness"]) / float(refs["smoothness"])
        )
        result["quality_path_length"] = float(sample["path_length"])
        result["quality_control_effort"] = float(sample["control_effort"])
        result["quality_smoothness"] = float(sample["smoothness"])
        result["quality_j"] = j
        result["quality_score"] = quality_score(sample, refs, weights)

    return quality_refs_for_result(results[0]) if results else {
        "path_length": 1.0,
        "control_effort": 1.0,
        "smoothness": 1.0,
    }


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
    p.add_argument("--timeout_sec", type=float, default=300.0)
    p.add_argument("--same_process", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="Number of benchmark cases to run concurrently.")
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
    if int(args.workers) <= 1:
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
