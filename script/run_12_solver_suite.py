#!/usr/bin/env python3
"""Run the seven-environment, twelve-solver benchmark suite.

This script assumes `script/prepare_environment_assets.py` has already created:
  db/by_env/<environment>/{S1_database_maze.json,s1_sfcbf_success_trajs.npz,
  nn_dataset_maze.npz,s1_policy_control_cnn.pth}

For continual-learning variants, 
each successful S2 trajectory is inserted into memory before the next scenario is solved, 
and neural S1 is retrained every `--retrain_every` successful S2 trajectories by default.

Parallelism convention:
  --workers controls parallelism across environments/families.
  --case_workers controls optional parallelism inside one non-CL benchmark run.
  For CL runs, cases inside one environment are intentionally executed in scenario order
  so that memory updates and neural retraining are applied immediately.


intended large run:

cd /Users/apple/Desktop/sofai

PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/run_12_solver_suite.py \
  --families bugtrap \
  --configs s1_neural s1_primitives s2_cbf s2_mpc sofai_cbf_neural sofai_cbf_primitives sofai_cbf_neural_cl sofai_cbf_primitives_cl\
  --assets_dir db/by_env \
  --benchmark_dir input/benchmarks_10k \
  --out_dir output/benchmark_runs/twelve_solver_suite_bugtrap \
  --scenario_ids all \
  --limit_per_environment 0 \
  --workers 1 \
  --case_workers 1 \
  --timeout_sec 300 \
  --retrain_every 25 \
  --train_epochs_cl 25 \
  --mplconfigdir /private/tmp/mpl


sample small run:

cd /Users/apple/Desktop/sofai

PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/run_12_solver_suite.py \
  --families large_sparse \
  --configs sofai_cbf_neural sofai_cbf_neural_cl \
  --limit_per_environment 200 \
  --workers 1 \
  --case_workers 1 \
  --retrain_every 50 \
  --timeout_sec 120

limit_per_environment 100: the first 100 scenarios of the benchmark
workers: parallel environments/families
case_workers: optional within-benchmark workers, defaulting to 1.


cd /Users/apple/Desktop/sofai

PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/run_12_solver_suite.py \
  --families bugtrap \
  --configs \
    sofai_mpc_primitives sofai_mpc_primitives_cl \
    sofai_mpc_neural sofai_mpc_neural_cl \
  --limit_per_environment 40 \
  --workers 1 \
  --case_workers 1 \
  --retrain_every 3 \
  --train_epochs_cl 2 \
  --timeout_sec 90 \
  --out_dir output/benchmark_runs/quick_cl_signal_bugtrap



PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/run_12_solver_suite.py \
  --families bugtrap \
  --configs \
    sofai_mpc_neural_cl \
  --limit_per_environment 10 \
  --workers 1 \
  --case_workers 1 \
  --retrain_every 2 \
  --train_epochs_cl 20 \
  --timeout_sec 90 \
  --out_dir output/benchmark_runs/quick_cl_signal_bugtrap

  

  
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FAMILIES = [
    "small_open",
    "large_sparse",
    "dense_clutter",
    "wall_gap",
    "serial_walls",
    "maze_branching",
    "bugtrap",
]

CONFIGS = [
    {"label": "s1_primitives", "run_type": "s1", "s1": "primitives", "s2": "mpc", "cl": False},
    {"label": "s1_neural", "run_type": "s1", "s1": "neural", "s2": "mpc", "cl": False},
    {"label": "s2_mpc", "run_type": "s2", "s1": "primitives", "s2": "mpc", "cl": False},
    {"label": "s2_cbf", "run_type": "s2", "s1": "primitives", "s2": "cbf", "cl": False},
    {"label": "sofai_mpc_primitives", "run_type": "sofai", "s1": "primitives", "s2": "mpc", "cl": False},
    {"label": "sofai_mpc_neural", "run_type": "sofai", "s1": "neural", "s2": "mpc", "cl": False},
    {"label": "sofai_cbf_primitives", "run_type": "sofai", "s1": "primitives", "s2": "cbf", "cl": False},
    {"label": "sofai_cbf_neural", "run_type": "sofai", "s1": "neural", "s2": "cbf", "cl": False},
    {"label": "sofai_mpc_primitives_cl", "run_type": "sofai", "s1": "primitives", "s2": "mpc", "cl": True},
    {"label": "sofai_mpc_neural_cl", "run_type": "sofai", "s1": "neural", "s2": "mpc", "cl": True},
    {"label": "sofai_cbf_primitives_cl", "run_type": "sofai", "s1": "primitives", "s2": "cbf", "cl": True},
    {"label": "sofai_cbf_neural_cl", "run_type": "sofai", "s1": "neural", "s2": "cbf", "cl": True},
]


def bool_from_csv(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def float_or_none(value: object) -> Optional[float]:
    try:
        out = float(str(value).strip())
    except Exception:
        return None
    return out if math.isfinite(out) else None


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def selected(raw: Iterable[str], valid: List[str], kind: str) -> List[str]:
    vals = list(raw)
    if not vals or vals == ["all"]:
        return list(valid)
    missing = [v for v in vals if v not in valid]
    if missing:
        raise SystemExit(f"Unknown {kind}: {', '.join(missing)}")
    return vals


def env_for_assets(base_env: Dict[str, str], assets_dir: Path, cl_dir: Optional[Path] = None) -> Dict[str, str]:
    source = cl_dir if cl_dir is not None else assets_dir
    env = dict(base_env)
    env["SOFAI_S1_DB_PATH"] = str(source / "S1_database_maze.json")
    env["SOFAI_S1_TRAJ_PATH"] = str(source / "s1_sfcbf_success_trajs.npz")
    env["SOFAI_NEW_S1_MODEL"] = str(source / "s1_policy_control_cnn.pth")
    env["SOFAI_NEW_S1_BASE_DATASET"] = str(source / "nn_dataset_maze.npz")
    env["SOFAI_NEW_S1_BASE_MEMORY_TRAJ"] = str(source / "s1_sfcbf_success_trajs.npz")
    env["SOFAI_NEW_S1_BASE_MEMORY_SCENARIOS"] = str(source / "benchmark_scenarios_maze_s1_db.json")
    return env


def resolve_benchmark_file(root: Path, benchmark_dir: str, family: str) -> Path:
    filename = f"benchmark_dualmp_{family}.json"
    preferred = root / benchmark_dir / filename
    fallback = root / "input" / filename
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find benchmark dictionary for {family}. Tried:\n"
        f"  {preferred}\n"
        f"  {fallback}"
    )


def run_cmd(cmd: List[str], *, cwd: Path, env: Dict[str, str], dry_run: bool = False) -> None:
    print("\n[cmd]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def config_by_label(label: str) -> Dict[str, Any]:
    return next((c for c in CONFIGS if c["label"] == label), {})


def normalize_selected_system(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "null", "nan"}:
        return ""
    if text in {"s1", "system1", "system_1", "primitives", "primitive", "neural"}:
        return "s1"
    if text in {"s2", "system2", "system_2", "mpc", "cbf"}:
        return "s2"
    if "s1" in text or "system 1" in text or "system_1" in text:
        return "s1"
    if "s2" in text or "system 2" in text or "system_2" in text:
        return "s2"
    return ""


def selected_system_from_row(row: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Return the selected solving system for a CSV row.

    Important: this is deliberately different from raw attempt columns such as
    `s1_success` and `s2_success` in older diagnostic runs. The table requested
    for the paper should count the selected/used successful solver.
    """
    for key in (
        "selected_attempt",
        "selected_system",
        "used_system",
        "solver_system",
        "system",
    ):
        system = normalize_selected_system(row.get(key))
        if system:
            return system

    # Fallback for older summary CSVs that did not write a selected-system column.
    # In S1-only/S2-only runs, the selected system is unambiguous if the row succeeded.
    if not bool_from_csv(row.get("success")):
        return ""
    if cfg.get("run_type") == "s1":
        return "s1"
    if cfg.get("run_type") == "s2":
        return "s2"

    # Last-resort inference for hybrid rows. Prefer S1 if it succeeded, because
    # the hybrid policy should accept S1 before falling back to S2.
    if bool_from_csv(row.get("s1_success")):
        return "s1"
    if bool_from_csv(row.get("s2_success")):
        return "s2"
    return ""


def summarize_csv(path: Path, label: str, family: str) -> Dict[str, Any]:
    cfg = config_by_label(label)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r.get("status") == "ok"]
    runtimes = [v for v in (float_or_none(r.get("runtime_sec")) for r in ok) if v is not None]
    total = len(rows)
    successful_rows = [r for r in rows if bool_from_csv(r.get("success"))]
    selected_systems = [selected_system_from_row(r, cfg) for r in successful_rows]
    s1_solved = sum(system == "s1" for system in selected_systems)
    s2_solved = sum(system == "s2" for system in selected_systems)
    return {
        "environment": family,
        "solver": label,
        "success_rate": (len(successful_rows) / total) if total else 0.0,
        "average_runtime": (sum(runtimes) / len(runtimes)) if runtimes else None,
        "mean_runtime": statistics.mean(runtimes) if runtimes else None,
        "p90_runtime": percentile(runtimes, 90),
        "ok": len(ok),
        "total": total,
        "s1_solved": s1_solved,
        "s2_solved": s2_solved,
    }


def write_table_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "environment",
        "solver",
        "success_rate",
        "average_runtime",
        "mean_runtime",
        "p90_runtime",
        "s1_solved",
        "s2_solved",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_markdown_tables(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["environment"]), []).append(row)

    def fmt_float(v: Any) -> str:
        if v is None or v == "":
            return ""
        return f"{float(v):.4f}"

    lines = ["# Twelve-Solver Benchmark Tables", ""]
    for family in list(FAMILIES) + ["average"]:
        if family not in by_family:
            continue
        lines += [
            f"## {family}",
            "",
            "| solver | success rate | average runtime | mean runtime | p90 runtime | s1 solved | s2 solved |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in by_family[family]:
            lines.append(
                "| {solver} | {success_rate} | {average_runtime} | {mean_runtime} | {p90_runtime} | {s1_solved} | {s2_solved} |".format(
                    solver=row["solver"],
                    success_rate=fmt_float(row["success_rate"]),
                    average_runtime=fmt_float(row["average_runtime"]),
                    mean_runtime=fmt_float(row["mean_runtime"]),
                    p90_runtime=fmt_float(row["p90_runtime"]),
                    s1_solved=row["s1_solved"],
                    s2_solved=row["s2_solved"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def average_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for cfg in CONFIGS:
        label = cfg["label"]
        group = [r for r in rows if r["solver"] == label and r["environment"] != "average"]
        if not group:
            continue
        avg_runtime = [float(r["average_runtime"]) for r in group if r["average_runtime"] is not None]
        mean_runtime = [float(r["mean_runtime"]) for r in group if r["mean_runtime"] is not None]
        p90_runtime = [float(r["p90_runtime"]) for r in group if r["p90_runtime"] is not None]
        out.append({
            "environment": "average",
            "solver": label,
            "success_rate": statistics.mean(float(r["success_rate"]) for r in group),
            "average_runtime": statistics.mean(avg_runtime) if avg_runtime else None,
            "mean_runtime": statistics.mean(mean_runtime) if mean_runtime else None,
            "p90_runtime": statistics.mean(p90_runtime) if p90_runtime else None,
            "s1_solved": statistics.mean(float(r["s1_solved"]) for r in group),
            "s2_solved": statistics.mean(float(r["s2_solved"]) for r in group),
        })
    return out


def copy_cl_assets(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "S1_database_maze.json",
        "s1_sfcbf_success_trajs.npz",
        "nn_dataset_maze.npz",
        "s1_policy_control_cnn.pth",
        "benchmark_scenarios_maze_s1_db.json",
    ]:
        target = dst / name
        if not target.exists():
            shutil.copy2(src / name, target)


def ensure_assets_exist(src: Path) -> None:
    required = [
        "S1_database_maze.json",
        "s1_sfcbf_success_trajs.npz",
        "nn_dataset_maze.npz",
        "s1_policy_control_cnn.pth",
        "benchmark_scenarios_maze_s1_db.json",
    ]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared assets in {src}. Missing: {', '.join(missing)}"
        )


def find_attempt(result: Dict[str, Any], system: str) -> Optional[Dict[str, Any]]:
    return next((a for a in result.get("attempts", []) if a.get("system") == system), None)


def update_cl_memory_and_records(
    *,
    root: Path,
    memory_path: Path,
    results: List[Dict[str, Any]],
    block_id: int,
    s2_records: List[Dict[str, Any]],
    cl_args: Any,
) -> int:
    sys.path.insert(0, str(root))
    from solvers.base import S1_S2_continual_maze as cl

    memory = cl.load_episodic_memory(memory_path)
    scenarios_by_id = {int(r["scenario"]["scenario_id"]): r["scenario"] for r in results if "scenario" in r}
    added = 0

    for result in results:
        sc = result.get("scenario")
        if not sc:
            continue
        s2 = find_attempt(result, "s2")
        if not s2 or not s2.get("success"):
            continue
        out = {
            "scenario_id": int(result["scenario_id"]),
            "used_system": "S2",
            "success": bool(s2.get("success")),
            "collision_free": bool(s2.get("collision_free")),
            "goal_reached": bool(s2.get("goal_reached")),
            "states": s2.get("states") or [],
            "inputs": s2.get("inputs") or [],
            "runtime_sec": float(s2.get("runtime_sec") or 0.0),
            "final_dist": float(s2.get("final_goal_error") or 0.0),
            "s1_attempt": find_attempt(result, "s1"),
            "switch_reason": "online_s2_success",
        }
        item = cl.make_episodic_memory_item(sc, out, cl_args, block_id=block_id)
        if item is not None:
            memory.setdefault("items", []).append(item)
            added += 1
        s2_records.extend(cl.collect_full_s2_records(out, scenarios_by_id, cl_args))

    cl.cap_episodic_memory(memory, cl_args)
    cl.save_episodic_memory(memory, memory_path)
    return added


def update_cl_memory_and_records_for_result(
    *,
    root: Path,
    memory_path: Path,
    result: Dict[str, Any],
    block_id: int,
    s2_records: List[Dict[str, Any]],
    cl_args: Any,
) -> int:
    return update_cl_memory_and_records(
        root=root,
        memory_path=memory_path,
        results=[result],
        block_id=block_id,
        s2_records=s2_records,
        cl_args=cl_args,
    )


def maybe_retrain_neural(
    *,
    root: Path,
    cl_dir: Path,
    cl_args: Any,
    s2_records: List[Dict[str, Any]],
    block_id: int,
) -> None:
    if not s2_records:
        return
    sys.path.insert(0, str(root))
    import torch
    from solvers.base import S1_S2_continual_maze as cl

    current_model = cl_dir / "s1_policy_control_cnn.pth"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, norm, _, _ = cl.load_s1_model(current_model, device)

    dataset = cl_dir / f"continual_dataset_block_{block_id:03d}.npz"
    candidate = cl_dir / f"s1_policy_candidate_block_{block_id:03d}.pth"
    scenarios_by_id = {
        int(sc.get("scenario_id", i)): sc
        for i, sc in enumerate(json.loads(Path(cl_args.scenarios).read_text()))
    }
    cl.build_full_retrain_dataset(
        scenarios_by_id=scenarios_by_id,
        s2_full_records=s2_records,
        out_npz=dataset,
        args=cl_args,
        reference_norm=norm,
    )
    cl.train_s1_model(dataset, candidate, cl_args, current_model_path=current_model)
    shutil.copy2(candidate, current_model)
    print(f"[cl] retrained neural S1 at block {block_id}: {current_model}")


def make_cl_args(root: Path, family: str, benchmark_file: Path, cl_dir: Path, retrain_every: int, train_epochs: int) -> Any:
    sys.path.insert(0, str(root))
    from solvers.base import S1_S2_continual_maze as cl

    parser = cl.make_full_retrain_parser()
    args = parser.parse_args([
        "--scenarios", str(benchmark_file),
        "--initial_model", str(cl_dir / "s1_policy_control_cnn.pth"),
        "--base_dataset", str(cl_dir / "nn_dataset_maze.npz"),
        "--base_memory_traj_npz", str(cl_dir / "s1_sfcbf_success_trajs.npz"),
        "--base_memory_scenarios", str(cl_dir / "benchmark_scenarios_maze_s1_db.json"),
        "--train_script", str(root / "solvers/base/train_nn_policy.py"),
        "--workdir", str(cl_dir),
        "--batch_size_scenarios", str(retrain_every),
        "--train_epochs", str(train_epochs),
        "--no_plot_curves",
    ])
    args.episodic_memory_store_s2_success = True
    args.s2_full_require_fallback = False
    args.episodic_memory_max_per_bucket = 0
    args.episodic_memory_max_total = 0
    return args


def run_cl_cases_immediately(
    *,
    root: Path,
    family: str,
    benchmark_file: Path,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    memory_path: Path,
    cl_dir: Path,
    cl_args: Any,
) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(root))
    import run_motion_planning_benchmarks as runner_mod

    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("MPLCONFIGDIR", args.mplconfigdir)
    env = env_for_assets(env, root / args.assets_dir / family, cl_dir=cl_dir)
    env["SOFAI_NEW_S1_MEMORY_PATH"] = str(memory_path)
    env["SOFAI_NEW_S1_RESUME_MEMORY"] = "1"
    env["SOFAI_S1_EPISODIC_MEMORY_PATH"] = str(memory_path)
    os.environ.update(env)

    total_count = len(json.loads(benchmark_file.read_text()))
    scenario_ids = runner_mod.parse_ids(args.scenario_ids, total_count)
    if args.limit_per_environment > 0:
        scenario_ids = scenario_ids[: int(args.limit_per_environment)]

    s2_records: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    completed = 0
    retrain_block_id = 0
    s2_success_count = 0
    next_retrain_at = max(1, int(args.retrain_every))

    def make_opts(sid: int) -> Dict[str, Any]:
        return {
            "root": str(root),
            "dictionary": str(benchmark_file),
            "scenario_id": int(sid),
            "s1": cfg["s1"],
            "s2": cfg["s2"],
            "run_type": cfg["run_type"],
            "run_all_attempts": False,
            "mplconfigdir": args.mplconfigdir,
        }

    # CL cases must be sequential within one environment: each successful S2
    # trajectory is inserted into memory before the next scenario is solved, and
    # neural S1 is retrained exactly at the configured interval. Parallelism for
    # the full suite is therefore across environments, handled in main().
    for sid in scenario_ids:
        result = runner_mod.run_case_timed(
            make_opts(sid),
            float(args.timeout_sec),
            False,
        )
        all_results.append(result)
        completed += 1
        block_id = 1 + (completed - 1) // max(1, int(args.retrain_every))
        added_memory = update_cl_memory_and_records_for_result(
            root=root,
            memory_path=memory_path,
            result=result,
            block_id=block_id,
            s2_records=s2_records,
            cl_args=cl_args,
        )
        s2_success_count += int(added_memory)
        print(
            f"[cl case {completed}/{len(scenario_ids)}] "
            f"scenario={result.get('scenario_id')} "
            f"selected={result.get('selected_attempt') or 'none'} "
            f"success={result.get('success', False)} "
            f"s2_memory_added={added_memory} "
            f"s2_success_total={s2_success_count}"
        )
        if cfg["s1"] == "neural" and s2_success_count >= next_retrain_at:
            retrain_block_id += 1
            maybe_retrain_neural(
                root=root,
                cl_dir=cl_dir,
                cl_args=cl_args,
                s2_records=s2_records,
                block_id=retrain_block_id,
            )
            next_retrain_at += max(1, int(args.retrain_every))

    all_results.sort(key=lambda r: int(r.get("scenario_id", -1)))
    return all_results


def run_one_config(args: argparse.Namespace, family: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    out_root = root / args.out_dir / family / cfg["label"]
    out_root.mkdir(parents=True, exist_ok=True)
    assets = root / args.assets_dir / family
    benchmark_file = resolve_benchmark_file(root, args.benchmark_dir, family)
    runner = root / "run_motion_planning_benchmarks.py"

    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("MPLCONFIGDIR", args.mplconfigdir)
    env = env_for_assets(env, assets)

    if args.dry_run:
        placeholder = {
            "environment": family,
            "solver": cfg["label"],
            "success_rate": 0.0,
            "average_runtime": None,
            "mean_runtime": None,
            "p90_runtime": None,
            "ok": 0,
            "total": 0,
            "s1_solved": 0,
            "s2_solved": 0,
        }
    else:
        placeholder = None

    if not cfg["cl"]:
        cmd = [
            args.python,
            str(runner),
            "--input_dir",
            str(benchmark_file.parent),
            "--patterns",
            benchmark_file.name,
            "--scenario_ids",
            args.scenario_ids,
            "--limit_per_dictionary",
            str(args.limit_per_environment),
            "--s1",
            cfg["s1"],
            "--s2",
            cfg["s2"],
            "--run_type",
            cfg["run_type"],
            "--timeout_sec",
            str(args.timeout_sec),
            "--workers",
            str(args.case_workers),
            "--out_dir",
            str(out_root),
            "--out_prefix",
            cfg["label"],
            "--mplconfigdir",
            args.mplconfigdir,
        ]
        run_cmd(cmd, cwd=root, env=env, dry_run=args.dry_run)
        if placeholder is not None:
            return placeholder
        return summarize_csv(out_root / f"{cfg['label']}_summary.csv", cfg["label"], family)

    cl_dir = out_root / "cl_assets"
    if args.dry_run:
        cl_dir.mkdir(parents=True, exist_ok=True)
    else:
        ensure_assets_exist(assets)
        copy_cl_assets(assets, cl_dir)
    memory_path = cl_dir / "episodic_s2_memory.json"
    env = env_for_assets(env, assets, cl_dir=cl_dir)
    env["SOFAI_NEW_S1_MEMORY_PATH"] = str(memory_path)
    env["SOFAI_NEW_S1_RESUME_MEMORY"] = "1"
    env["SOFAI_S1_EPISODIC_MEMORY_PATH"] = str(memory_path)

    count = len(json.loads(benchmark_file.read_text()))
    if args.limit_per_environment > 0:
        count = min(count, int(args.limit_per_environment))
    if args.scenario_ids != "all":
        blocks = [args.scenario_ids]
    else:
        blocks = [
            f"{start}-{min(start + args.retrain_every - 1, count - 1)}"
            for start in range(0, count, args.retrain_every)
        ]

    all_results: List[Dict[str, Any]] = []
    s2_records: List[Dict[str, Any]] = []
    cl_args = make_cl_args(root, family, benchmark_file, cl_dir, args.retrain_every, args.train_epochs_cl)
    if args.dry_run:
        for block_id, scenario_ids in enumerate(blocks, start=1):
            prefix = f"{cfg['label']}_block_{block_id:03d}"
            cmd = [
                args.python,
                str(runner),
                "--input_dir",
                str(benchmark_file.parent),
                "--patterns",
                benchmark_file.name,
                "--scenario_ids",
                scenario_ids,
                "--s1",
                cfg["s1"],
                "--s2",
                cfg["s2"],
                "--run_type",
                cfg["run_type"],
                "--timeout_sec",
                str(args.timeout_sec),
                "--workers",
                str(args.workers),
                "--out_dir",
                str(out_root / "blocks"),
                "--out_prefix",
                prefix,
                "--mplconfigdir",
                args.mplconfigdir,
            ]
            run_cmd(cmd, cwd=root, env=env, dry_run=True)
    else:
        all_results = run_cl_cases_immediately(
            root=root,
            family=family,
            benchmark_file=benchmark_file,
            cfg=cfg,
            args=args,
            memory_path=memory_path,
            cl_dir=cl_dir,
            cl_args=cl_args,
        )

    if not args.dry_run:
        sys.path.insert(0, str(root))
        import run_motion_planning_benchmarks as runner_mod
        runner_mod.write_outputs(out_root, cfg["label"], all_results)
    elif placeholder is not None:
        return placeholder
    return summarize_csv(out_root / f"{cfg['label']}_summary.csv", cfg["label"], family)


def run_one_family(args: argparse.Namespace, family: str, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    print(f"\n========== ENVIRONMENT: {family} ==========")
    for cfg in configs:
        print(f"\n----- {family} / {cfg['label']} -----")
        rows.append(run_one_config(args, family, cfg))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--families", nargs="+", default=["all"])
    p.add_argument("--configs", nargs="+", default=["all"])
    p.add_argument("--assets_dir", default="db/by_env")
    p.add_argument("--benchmark_dir", default="input/benchmarks_10k")
    p.add_argument("--out_dir", default="output/benchmark_runs/twelve_solver_suite")
    p.add_argument("--scenario_ids", default="all")
    p.add_argument("--limit_per_environment", type=int, default=0)
    p.add_argument("--workers", type=int, default=4, help="Number of environments/families to run in parallel.")
    p.add_argument("--case_workers", type=int, default=1, help="Optional workers inside one non-CL benchmark run. Keep at 1 for benchmark-level parallelism only.")
    p.add_argument("--timeout_sec", type=float, default=300.0)
    p.add_argument("--retrain_every", type=int, default=500)
    p.add_argument("--train_epochs_cl", type=int, default=25)
    p.add_argument("--mplconfigdir", default="/private/tmp/mpl")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    families = selected(args.families, FAMILIES, "environment")
    labels = [c["label"] for c in CONFIGS]
    config_labels = selected(args.configs, labels, "config")
    configs = [c for c in CONFIGS if c["label"] in config_labels]

    family_rows: Dict[str, List[Dict[str, Any]]] = {}
    env_workers = max(1, min(int(args.workers), len(families)))

    if env_workers == 1:
        for family in families:
            family_rows[family] = run_one_family(args, family, configs)
    else:
        print(f"[parallel] running {len(families)} environment(s) with {env_workers} environment worker(s)")
        with ProcessPoolExecutor(max_workers=env_workers) as pool:
            futures = {pool.submit(run_one_family, args, family, configs): family for family in families}
            for future in as_completed(futures):
                family = futures[future]
                family_rows[family] = future.result()

    rows: List[Dict[str, Any]] = []
    for family in families:
        rows.extend(family_rows.get(family, []))

    rows.extend(average_rows(rows))
    out_root = root / args.out_dir
    write_table_csv(out_root / "twelve_solver_tables.csv", rows)
    write_markdown_tables(out_root / "twelve_solver_tables.md", rows)
    print(f"\n[write] {out_root / 'twelve_solver_tables.csv'}")
    print(f"[write] {out_root / 'twelve_solver_tables.md'}")


if __name__ == "__main__":
    main()
