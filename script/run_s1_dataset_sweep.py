#!/usr/bin/env python3
"""Measure whether S1's success rate responds to the size of its training set.

The sweep trains one S1 checkpoint per demonstration budget from a single
bootstrap collection, so the budgets are nested (5 demos are the first 5 of the
30, and so on) and nothing but dataset size changes between arms. Every arm is
then benchmarked under both safety filters:

  greedy  the original filter, which ranks candidate controls by one-step
          distance to the goal. Its progress penalty is larger than the spread of
          the rest of the score, so it never moves away from the goal while any
          collision-free action moves toward it, and the policy only breaks ties.
  policy  the deviation-minimising filter, which executes the policy action and
          deflects it only as far as safety requires.

The expected reading is a flat success curve under `greedy` regardless of budget,
and a curve that actually responds to the budget under `policy`.

Example:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python script/run_s1_dataset_sweep.py \
      --family bugtrap --train_n 140 --sizes 5 30 60 100 --workers 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--family", default="bugtrap")
    p.add_argument("--train_n", type=int, default=140,
                   help="Scenarios offered to the S2 teacher. Only its successes become demonstrations.")
    p.add_argument("--train_seed", type=int, default=7)
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 30, 60, 100],
                   help="Demonstration budgets to train, nested in bootstrap file order.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Independent training runs per budget. Weight initialisation is not seeded, "
                        "so repeats measure how much of a difference between budgets is just run-to-run "
                        "variance. With 100 evaluation scenarios one standard error is already ~4.5 "
                        "points, so single runs cannot separate small gaps.")
    p.add_argument("--s2_solver", default="mpc", choices=["mpc", "cbf", "mpc_warm", "mpc_do"])
    p.add_argument("--filters", nargs="+", default=["greedy", "policy"], choices=["greedy", "policy"])
    p.add_argument("--eval_scenario_ids", default="0-99")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout_sec", type=int, default=300)
    p.add_argument("--target_grad_steps", type=int, default=3000,
                   help="Epochs per arm are chosen to hit roughly this many optimizer steps, so a "
                        "small budget is not also penalised by receiving less optimisation.")
    p.add_argument("--min_epochs", type=int, default=25)
    p.add_argument("--max_epochs", type=int, default=900)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_source", default="all_success", choices=["s2", "selected", "all_success"])
    p.add_argument("--dt_nom", type=float, default=0.075)
    p.add_argument("--n_steps_nom", type=int, default=900)
    p.add_argument("--train_device", default="cpu")
    p.add_argument("--out_dir", default="output/s1_dataset_sweep")
    p.add_argument("--input_dir", default="input/nl")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_bench", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Redo training and benchmarking for arms whose outputs already exist. "
                        "Without it the sweep resumes and only fills in what is missing.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def acados_env(root: Path) -> Dict[str, str]:
    """Mirror .env.acados so the MPC teacher can load its shared libraries."""
    acados = root / "safe_control" / "acados"
    lib = str(acados / "lib")
    env = {
        "ACADOS_SOURCE_DIR": str(acados),
        "ACADOS_INSTALL_DIR": str(acados),
        "ACADOS_PYTHON_INTERFACE_PATH": str(acados / "interfaces" / "acados_template" / "acados_template"),
    }
    for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = os.environ.get(key, "")
        env[key] = f"{lib}:{existing}" if existing else lib
    return env


def run(cmd: Sequence[str], *, root: Path, dry_run: bool, extra_env: Optional[Dict[str, str]] = None) -> None:
    print("\n[cmd]", " ".join(str(c) for c in cmd), flush=True)
    if extra_env:
        shown = {k: v for k, v in extra_env.items() if not k.startswith(("DYLD", "LD_LIBRARY"))}
        if shown:
            print("[env]", " ".join(f"{k}={v}" for k, v in shown.items()), flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH", "")]))
    env.update(extra_env or {})
    subprocess.run([str(c) for c in cmd], cwd=str(root), env=env, check=True)


def successful_rows(jsonl: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("success"):
            rows.append(row)
    return rows


def estimate_samples(jsonl: Path, budget: int, context_len: int = 20) -> int:
    """Approximate the supervised sample count for the first `budget` demos."""
    total = 0
    for row in successful_rows(jsonl)[: budget or None]:
        for attempt in row.get("attempts", []):
            states = attempt.get("states")
            if attempt.get("success") and isinstance(states, list):
                total += max(0, len(states) - 1)
                break
    return total


def epochs_for(samples: int, args: argparse.Namespace) -> int:
    steps_per_epoch = max(1, int(0.9 * samples) // int(args.batch))
    epochs = math.ceil(int(args.target_grad_steps) / steps_per_epoch)
    return int(min(max(epochs, args.min_epochs), args.max_epochs))


def aggregate(runs_jsonl: Path) -> Dict[str, Any]:
    n = 0
    success = 0
    collision_free = 0
    qualities: List[float] = []
    steps: List[int] = []
    terminations: Dict[str, int] = {}
    solved: List[Any] = []
    for line in runs_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        ok = bool(row.get("success"))
        success += ok
        collision_free += bool(row.get("collision_free"))
        if ok:
            solved.append(row.get("scenario_id"))
            q = row.get("quality_score")
            if isinstance(q, (int, float)) and math.isfinite(float(q)):
                qualities.append(float(q))
        for attempt in row.get("attempts", []):
            if attempt.get("system") == "S1_neural" or attempt.get("name") == "s1_neural":
                states = attempt.get("states")
                if isinstance(states, list):
                    steps.append(len(states))
                break
        term = str(row.get("s1_termination") or "")
        for attempt in row.get("attempts", []):
            term = term or str(attempt.get("s1_termination") or "")
        if term:
            terminations[term] = terminations.get(term, 0) + 1
    return {
        "cases": n,
        "success": success,
        "success_rate": success / n if n else float("nan"),
        "collision_free_rate": collision_free / n if n else float("nan"),
        "mean_quality": sum(qualities) / len(qualities) if qualities else float("nan"),
        "quality_n": len(qualities),
        "mean_steps": sum(steps) / len(steps) if steps else float("nan"),
        "terminations": terminations,
        "solved": sorted(x for x in solved if x is not None),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    family = args.family
    out_dir = root / args.out_dir
    input_dir = root / args.input_dir
    bootstrap_dir = out_dir / "bootstrap"
    model_dir = out_dir / "models"
    bench_dir = out_dir / "bench"
    for d in (out_dir, bootstrap_dir, model_dir, bench_dir):
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)

    train_prefix = f"benchmark_dualmp_nl_{family}_sweeptrain"
    train_dict = input_dir / f"{train_prefix}_{family}.json"
    eval_dict = input_dir / f"benchmark_dualmp_nl_{family}_eval_{family}.json"
    if not eval_dict.exists():
        raise SystemExit(f"missing evaluation dictionary {eval_dict}")

    # 1. A dedicated training dictionary, so the held-out evaluation set the
    #    earlier numbers were measured on is left untouched.
    if not args.skip_generate:
        run([args.python, "input/generate_nl_dict.py",
             "--root", str(root), "--output_dir", str(input_dir),
             "--prefix", train_prefix, "--n_per_family", str(args.train_n),
             "--seed", str(args.train_seed), "--families", family],
            root=root, dry_run=args.dry_run)

    bootstrap_jsonl = bootstrap_dir / f"{train_dict.stem}_{args.s2_solver}_bootstrap_runs.jsonl"

    # 2. One S2 collection shared by every arm.
    if not args.skip_collect:
        run([args.python, "run_motion_planning_benchmarks.py",
             "--root", str(root), "--input_dir", str(input_dir),
             "--patterns", train_dict.name,
             "--scenario_ids", f"0-{args.train_n - 1}",
             "--run_type", "s2", "--s2", args.s2_solver,
             "--timeout_sec", str(args.timeout_sec),
             "--workers", str(args.workers),
             "--out_dir", str(bootstrap_dir),
             "--out_prefix", f"{train_dict.stem}_{args.s2_solver}_bootstrap"],
            root=root, dry_run=args.dry_run, extra_env=acados_env(root))

    available = 0
    if not args.dry_run and bootstrap_jsonl.exists():
        available = len(successful_rows(bootstrap_jsonl))
        print(f"\n[bootstrap] {available} successful {args.s2_solver.upper()} demonstrations in {bootstrap_jsonl}")
        short = [s for s in args.sizes if s > available]
        if short:
            print(f"[warn] budgets {short} exceed the {available} demonstrations collected; "
                  f"those arms will silently train on {available}.")

    # 3. One checkpoint per (budget, repeat), all from the same demonstrations.
    plan: List[Dict[str, Any]] = []
    for size in args.sizes:
        samples = estimate_samples(bootstrap_jsonl, size) if (not args.dry_run and bootstrap_jsonl.exists()) else 0
        epochs = epochs_for(samples, args) if samples else args.min_epochs
        for repeat in range(1, max(1, int(args.repeats)) + 1):
            suffix = f"n{size}" if args.repeats <= 1 else f"n{size}_r{repeat}"
            plan.append({"size": size, "repeat": repeat, "tag": suffix,
                         "model": model_dir / f"s1_{family}_{suffix}.pth",
                         "samples": samples, "epochs": epochs})

    if plan and not args.dry_run:
        print("\n[plan] budget -> approx samples -> epochs (matched optimizer steps)")
        for item in plan:
            print(f"  {item['tag']:<12s} samples~{item['samples']:<6d} epochs={item['epochs']}")

    if not args.skip_train:
        for item in plan:
            if item["model"].exists() and not args.force:
                print(f"\n[skip] {item['model']} already trained")
                continue
            run([args.python, "script/train_s1_nonlinear.py",
                 "--root", str(root),
                 "--dictionary", str(train_dict),
                 "--results_jsonl", str(bootstrap_jsonl),
                 "--out_model", str(item["model"]),
                 "--out_dataset", str(model_dir / f"dataset_{family}_{item['tag']}.npz"),
                 "--audit_json", str(model_dir / f"audit_{family}_{item['tag']}.json"),
                 "--source", args.train_source,
                 "--max_trajectories", str(item["size"]),
                 "--epochs", str(item["epochs"]),
                 "--batch", str(args.batch),
                 "--lr", str(args.lr),
                 "--device", args.train_device,
                 "--dt_nom", str(args.dt_nom),
                 "--n_steps_nom", str(args.n_steps_nom)],
                root=root, dry_run=args.dry_run)

    # 4. Benchmark every (budget, filter) pair on the untouched evaluation set.
    results: Dict[str, Dict[str, Any]] = {}
    if not args.skip_bench:
        for filter_mode in args.filters:
            for item in plan:
                tag = f"{filter_mode}_{item['tag']}"
                case_dir = bench_dir / tag
                if (case_dir / "s1_neural_runs.jsonl").exists() and not args.force:
                    print(f"\n[skip] {tag} already benchmarked")
                    results[tag] = {"filter": filter_mode, "size": item["size"],
                                    "repeat": item["repeat"],
                                    **aggregate(case_dir / "s1_neural_runs.jsonl")}
                    continue
                run([args.python, "run_motion_planning_benchmarks.py",
                     "--root", str(root), "--input_dir", str(input_dir),
                     "--patterns", eval_dict.name,
                     "--scenario_ids", args.eval_scenario_ids,
                     "--run_type", "s1", "--s1", "neural",
                     "--timeout_sec", str(args.timeout_sec),
                     "--workers", str(args.workers),
                     "--out_dir", str(case_dir),
                     "--out_prefix", "s1_neural"],
                    root=root, dry_run=args.dry_run,
                    extra_env={
                        "SOFAI_NEW_S1_MODEL": str(item["model"]),
                        "SOFAI_S1_FILTER_MODE": filter_mode,
                    })
                runs_jsonl = case_dir / "s1_neural_runs.jsonl"
                if not args.dry_run and runs_jsonl.exists():
                    results[tag] = {"filter": filter_mode, "size": item["size"],
                                    "repeat": item["repeat"], **aggregate(runs_jsonl)}

    if args.dry_run:
        return

    # 5. Report.
    for filter_mode in args.filters:
        for item in plan:
            tag = f"{filter_mode}_{item['tag']}"
            if tag in results:
                continue
            runs_jsonl = bench_dir / tag / "s1_neural_runs.jsonl"
            if runs_jsonl.exists():
                results[tag] = {"filter": filter_mode, "size": item["size"],
                                "repeat": item["repeat"], **aggregate(runs_jsonl)}

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(
        {"family": family, "demonstrations_available": available,
         "eval_scenario_ids": args.eval_scenario_ids, "plan":
             [{k: (str(v) if isinstance(v, Path) else v) for k, v in item.items()} for item in plan],
         "results": results}, indent=2))

    print("\n" + "=" * 82)
    print(f"S1 dataset-size sweep | family={family} | eval scenarios {args.eval_scenario_ids}")
    print("=" * 82)
    header = (f"{'filter':8s} {'demos':>6s} {'runs':>5s} {'success rate':>22s} "
              f"{'mean':>7s} {'quality':>8s} {'steps':>7s}")
    for filter_mode in args.filters:
        print(f"\n{header}")
        print("-" * len(header))
        for size in args.sizes:
            arms = [r for r in results.values() if r["filter"] == filter_mode and r["size"] == size]
            if not arms:
                continue
            rates = [a["success_rate"] for a in arms]
            mean_rate = sum(rates) / len(rates)
            per_run = " ".join(f"{a['success']:d}" for a in sorted(arms, key=lambda a: a["repeat"]))
            quality = [a["mean_quality"] for a in arms if math.isfinite(a["mean_quality"])]
            steps = [a["mean_steps"] for a in arms if math.isfinite(a["mean_steps"])]
            print(f"{filter_mode:8s} {size:6d} {len(arms):5d} {per_run:>22s} "
                  f"{mean_rate:7.3f} "
                  f"{(sum(quality) / len(quality) if quality else float('nan')):8.3f} "
                  f"{(sum(steps) / len(steps) if steps else float('nan')):7.1f}")

    # One standard error on a rate estimated from `cases` Bernoulli trials, as a
    # yardstick for whether a gap between budgets is worth reading.
    cases = next((r["cases"] for r in results.values() if r.get("cases")), 100)
    print(f"\n[noise] one standard error at rate 0.3 with {cases} scenarios is "
          f"{math.sqrt(0.3 * 0.7 / cases):.3f}; gaps smaller than about twice that are not readable")

    for filter_mode in args.filters:
        points = []
        for size in args.sizes:
            arms = [r for r in results.values() if r["filter"] == filter_mode and r["size"] == size]
            if arms:
                points.append((size, sum(a["success_rate"] for a in arms) / len(arms)))
        if len(points) >= 2:
            span = max(r for _, r in points) - min(r for _, r in points)
            print(f"[{filter_mode}] mean success rate {points[0][1]:.3f} -> {points[-1][1]:.3f} "
                  f"across {points[0][0]}..{points[-1][0]} demos (spread {span:.3f})")

    print(f"\n[summary] {summary_path}")


if __name__ == "__main__":
    main()
