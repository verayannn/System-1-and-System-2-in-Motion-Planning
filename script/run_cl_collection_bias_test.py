#!/usr/bin/env python3
"""Is the flat continual-learning curve caused by the collection rule or by noise?

A continual-learning block only adds demonstrations from scenarios where S1 just
failed, so the accumulated set drifts toward trajectories the policy cannot fit.
That is one candidate explanation for a probe curve that does not rise. The other
is simply that consecutive blocks grow the dataset by far less than the spread
between two independent retrainings on identical data.

Those two cannot be separated from a single continual-learning run, so this
script retrains at each block's cumulative dataset *size* under two collection
rules and compares:

  biased    the demonstrations the continual-learning loop actually accumulated,
            i.e. only scenarios where S1 had failed
  unbiased  the same number of demonstrations drawn at random from a pool of
            scenarios sampled independently of S1

Everything else is held fixed: same teacher, same training configuration, same
number of optimizer steps, same fixed probe set, and several repeats per point so
the retraining spread is measured rather than assumed. If the unbiased curve
rises where the biased one does not, the collection rule is responsible. If
neither rises and the repeats overlap heavily, the increments are simply below
the noise floor.

Example:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python script/run_cl_collection_bias_test.py \
      --pool_jsonl output/cl_collection_test/pool/bugtrap_pool_mpc_runs.jsonl \
      --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    p.add_argument("--suite_dir", default="output/benchmark_runs/nl_bugtrap_suite")
    p.add_argument("--config", default="sofai_mpc_cl",
                   help="Continual-learning arm whose accumulated demonstrations are the biased condition.")
    p.add_argument("--pool_jsonl", default="output/cl_collection_test/pool/bugtrap_pool_mpc_runs.jsonl",
                   help="Unbiased demonstrations: the same teacher run over independently sampled scenarios.")
    p.add_argument("--dagger_jsonl", default="",
                   help="Optional third condition: MPC actions labelled on states S1 itself visited, "
                        "as written by collect_mpc_dagger.py. Unlike the other two conditions these "
                        "demonstrations lie on the policy's own state distribution.")
    p.add_argument("--probe_dictionary", default="input/nl/benchmark_dualmp_nl_bugtrap_probe_bugtrap.json")
    p.add_argument("--train_dictionary", default="input/nl/benchmark_dualmp_nl_bugtrap_train_bugtrap.json")
    p.add_argument("--probe_scenario_ids", default="0-99")
    p.add_argument("--sizes", type=int, nargs="+", default=[],
                   help="Demonstration counts. Defaults to the arm's cumulative per-block counts.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pool_seed", type=int, default=20260727)
    p.add_argument("--target_grad_steps", type=int, default=2500,
                   help="Epochs are chosen per size to hit this many optimizer steps, so a larger "
                        "dataset is not also given more optimisation.")
    p.add_argument("--min_epochs", type=int, default=12)
    p.add_argument("--max_epochs", type=int, default=600)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dt_nom", type=float, default=0.075)
    p.add_argument("--n_steps_nom", type=int, default=900)
    p.add_argument("--train_device", default="cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout_sec", type=int, default=300)
    p.add_argument("--filter_mode", default="policy", choices=["policy", "greedy"])
    p.add_argument("--out_dir", default="output/cl_collection_test")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_bench", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run(cmd: Sequence[str], *, root: Path, dry_run: bool, extra_env: Optional[Dict[str, str]] = None) -> None:
    print("\n[cmd]", " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH", "")]))
    env.update(extra_env or {})
    subprocess.run([str(c) for c in cmd], cwd=str(root), env=env, check=True)


def successful_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("success"):
            rows.append(row)
    return rows


def has_s2_demo(row: Dict[str, Any]) -> bool:
    return any(a.get("success") and a.get("system") == "s2" for a in row.get("attempts", []) or [])


def demo_sample_count(row: Dict[str, Any]) -> int:
    for a in row.get("attempts", []) or []:
        if a.get("success") and a.get("system") == "s2":
            states = a.get("states")
            return max(0, len(states) - 1) if isinstance(states, list) else 0
    return 0


def biased_rows(suite_dir: Path, config: str) -> List[Dict[str, Any]]:
    """Demonstrations in the order the continual-learning loop accumulated them."""
    rows: List[Dict[str, Any]] = []
    for path in sorted((suite_dir / config / "runs").glob("*_block*_runs.jsonl")):
        rows.extend(r for r in successful_rows(path) if has_s2_demo(r))
    return rows


def block_cumulative_sizes(suite_dir: Path, config: str) -> List[int]:
    sizes, total = [], 0
    for path in sorted((suite_dir / config / "runs").glob("*_block*_runs.jsonl")):
        total += sum(1 for r in successful_rows(path) if has_s2_demo(r))
        sizes.append(total)
    return sizes


def write_subset(rows: Sequence[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def epochs_for(samples: int, args: argparse.Namespace) -> int:
    steps_per_epoch = max(1, int(0.9 * samples) // int(args.batch))
    epochs = math.ceil(int(args.target_grad_steps) / steps_per_epoch)
    return int(min(max(epochs, args.min_epochs), args.max_epochs))


def aggregate(runs_jsonl: Path) -> Dict[str, Any]:
    n = success = 0
    qualities: List[float] = []
    solved: List[Any] = []
    for line in runs_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        if row.get("success"):
            success += 1
            solved.append(row.get("scenario_id"))
            q = row.get("quality_score")
            if isinstance(q, (int, float)) and math.isfinite(float(q)):
                qualities.append(float(q))
    return {
        "cases": n,
        "success": success,
        "success_rate": success / n if n else math.nan,
        "mean_quality": sum(qualities) / len(qualities) if qualities else math.nan,
        "solved": sorted(x for x in solved if x is not None),
    }


def spread(values: Sequence[float]) -> float:
    return (max(values) - min(values)) if values else math.nan


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = root / args.out_dir
    suite_dir = root / args.suite_dir
    subsets_dir = out_dir / "subsets"
    model_dir = out_dir / "models"
    bench_dir = out_dir / "bench"
    for d in (out_dir, subsets_dir, model_dir, bench_dir):
        d.mkdir(parents=True, exist_ok=True)

    pool_path = root / args.pool_jsonl
    if not pool_path.is_file():
        raise SystemExit(f"missing unbiased pool {pool_path}")

    biased = biased_rows(suite_dir, args.config)
    pool = [r for r in successful_rows(pool_path) if has_s2_demo(r)]
    random.Random(args.pool_seed).shuffle(pool)

    sources: List[tuple[str, List[Dict[str, Any]]]] = [("biased", biased), ("unbiased", pool)]
    if args.dagger_jsonl:
        dagger_path = root / args.dagger_jsonl
        if not dagger_path.is_file():
            raise SystemExit(f"missing dagger demonstrations {dagger_path}")
        dagger = [r for r in successful_rows(dagger_path) if has_s2_demo(r)]
        random.Random(args.pool_seed).shuffle(dagger)
        sources.append(("dagger", dagger))

    sizes = args.sizes or block_cumulative_sizes(suite_dir, args.config)
    smallest_source = min(len(rows) for _, rows in sources)
    sizes = [s for s in sizes if s <= smallest_source]
    if not sizes:
        raise SystemExit(
            "no usable sizes: "
            + ", ".join(f"{name} has {len(rows)} demos" for name, rows in sources)
        )
    conditions = [name for name, _ in sources]
    for name, rows in sources:
        print(f"[data] {name:9s} demonstrations available = {len(rows)}")
    print(f"[data] sizes under test                 = {sizes}")

    plan: List[Dict[str, Any]] = []
    for condition, source in sources:
        for size in sizes:
            rows = source[:size]
            subset = subsets_dir / f"{condition}_n{size}.jsonl"
            if not args.dry_run:
                write_subset(rows, subset)
            samples = sum(demo_sample_count(r) for r in rows)
            epochs = epochs_for(samples, args) if samples else args.min_epochs
            for repeat in range(1, max(1, int(args.repeats)) + 1):
                tag = f"{condition}_n{size}_r{repeat}"
                plan.append({
                    "condition": condition, "size": size, "repeat": repeat, "tag": tag,
                    "subset": subset, "samples": samples, "epochs": epochs,
                    "model": model_dir / f"s1_{tag}.pth",
                })

    print("\n[plan] condition/size -> samples -> epochs (matched optimizer steps)")
    for item in plan:
        if item["repeat"] == 1:
            print(f"  {item['condition']:9s} n={item['size']:<4d} samples~{item['samples']:<6d} epochs={item['epochs']}")

    if not args.skip_train:
        for item in plan:
            if item["model"].exists() and not args.force:
                print(f"\n[skip] {item['model'].name} already trained")
                continue
            run([args.python, "script/train_s1_nonlinear.py",
                 "--root", str(root),
                 "--dictionary", str(root / args.train_dictionary),
                 "--results_jsonl", str(item["subset"]),
                 "--out_model", str(item["model"]),
                 "--out_dataset", str(model_dir / f"dataset_{item['tag']}.npz"),
                 "--audit_json", str(model_dir / f"audit_{item['tag']}.json"),
                 "--source", "s2",
                 "--max_trajectories", str(item["size"]),
                 "--epochs", str(item["epochs"]),
                 "--batch", str(args.batch),
                 "--lr", str(args.lr),
                 "--device", args.train_device,
                 "--dt_nom", str(args.dt_nom),
                 "--n_steps_nom", str(args.n_steps_nom)],
                root=root, dry_run=args.dry_run)

    results: Dict[str, Dict[str, Any]] = {}
    probe_dict = root / args.probe_dictionary
    if not args.skip_bench:
        for item in plan:
            case_dir = bench_dir / item["tag"]
            runs_jsonl = case_dir / "s1_neural_runs.jsonl"
            if runs_jsonl.exists() and not args.force:
                print(f"\n[skip] {item['tag']} already benchmarked")
            else:
                run([args.python, "run_motion_planning_benchmarks.py",
                     "--root", str(root), "--input_dir", str(probe_dict.parent),
                     "--patterns", probe_dict.name,
                     "--scenario_ids", args.probe_scenario_ids,
                     "--run_type", "s1", "--s1", "neural",
                     "--timeout_sec", str(args.timeout_sec),
                     "--workers", str(args.workers),
                     "--out_dir", str(case_dir),
                     "--out_prefix", "s1_neural"],
                    root=root, dry_run=args.dry_run,
                    extra_env={"SOFAI_NEW_S1_MODEL": str(item["model"]),
                               "SOFAI_S1_FILTER_MODE": args.filter_mode})
            if not args.dry_run and runs_jsonl.exists():
                results[item["tag"]] = {
                    "condition": item["condition"], "size": item["size"],
                    "repeat": item["repeat"], **aggregate(runs_jsonl),
                }

    if args.dry_run:
        return

    summary = out_dir / "collection_bias_summary.json"
    summary.write_text(json.dumps({
        "config": args.config, "sizes": sizes, "conditions": conditions,
        "available": {name: len(rows) for name, rows in sources},
        "filter_mode": args.filter_mode, "results": results,
    }, indent=2))

    print("\n" + "=" * 78)
    print(f"Collection-rule test | {args.family} | probe scenarios {args.probe_scenario_ids}")
    print("=" * 78)
    header = f"{'condition':10s} {'demos':>6s} {'per-repeat':>16s} {'mean rate':>10s} {'spread':>7s} {'quality':>8s}"
    for condition in conditions:
        print(f"\n{header}")
        print("-" * len(header))
        for size in sizes:
            arms = sorted((r for r in results.values()
                           if r["condition"] == condition and r["size"] == size),
                          key=lambda r: r["repeat"])
            if not arms:
                continue
            rates = [a["success_rate"] for a in arms]
            quals = [a["mean_quality"] for a in arms if math.isfinite(a["mean_quality"])]
            print(f"{condition:10s} {size:6d} {' '.join(str(a['success']) for a in arms):>16s} "
                  f"{sum(rates) / len(rates):10.3f} {spread(rates):7.3f} "
                  f"{(sum(quals) / len(quals) if quals else math.nan):8.3f}")

    print("\n[reading]")
    for condition in conditions:
        pts = []
        for size in sizes:
            arms = [r for r in results.values() if r["condition"] == condition and r["size"] == size]
            if arms:
                pts.append((size, sum(a["success_rate"] for a in arms) / len(arms)))
        if len(pts) >= 2:
            print(f"  {condition:9s}: {pts[0][1]:.3f} at n={pts[0][0]} -> {pts[-1][1]:.3f} at n={pts[-1][0]} "
                  f"(change {pts[-1][1] - pts[0][1]:+.3f})")
    within = [spread([r["success_rate"] for r in results.values()
                      if r["condition"] == c and r["size"] == s])
              for c in conditions for s in sizes
              if any(r["condition"] == c and r["size"] == s for r in results.values())]
    within = [w for w in within if math.isfinite(w)]
    if within:
        print(f"  retraining noise floor: mean spread across repeats = {sum(within) / len(within):.3f}")
    print(f"\n[summary] {summary}")


if __name__ == "__main__":
    main()
