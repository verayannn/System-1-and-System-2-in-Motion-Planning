#!/usr/bin/env python3
"""Prepare nonlinear benchmark assets for one or more map families.

This script does three things:
1. Generate the nonlinear benchmark dictionary.
2. Collect successful System 2 trajectories rollouts for training.
3. Train the initial neural System 1 checkpoint from all successful bootstrap trajectories.


cd <repo-root>
PYTHONDONTWRITEBYTECODE=1 \
python script/prepare_environment_assets.py \
  --family bugtrap \
  --s2_solver cbf
  

PYTHONDONTWRITEBYTECODE=1 \
python script/prepare_environment_assets.py \
  --families dense_clutter bugtrap \
  --s2_solver cbf


cd <repo-root>
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
python script/prepare_environment_assets.py \
  --families small_open large_sparse dense_clutter wall_gap serial_walls maze_branching bugtrap \
  --train_n_per_family 500 \
  --eval_n_per_family 10000 \
  --s2_solver cbf


Jul 18th using:
cd <repo-root>
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
python script/prepare_environment_assets.py \
  --families small_open large_sparse wall_gap serial_walls maze_branching \
  --s2_solver cbf


all the families:

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
python script/prepare_environment_assets.py \
  --families small_open large_sparse wall_gap serial_walls maze_branching \
  --s2_solver cbf


new family: long_slalom

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


FAMILY_CHOICES = [
    "small_open",
    "large_sparse",
    "dense_clutter",
    "wall_gap",
    "serial_walls",
    "maze_branching",
    "bugtrap",
    "long_slalom",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--family", choices=FAMILY_CHOICES, default="dense_clutter", help="Single family to process.")
    p.add_argument("--families", nargs="+", default=[], help="Optional list of families to process in one run.")
    p.add_argument("--train_n_per_family", type=int, default=2000, help="Candidate S2 scenarios used to collect bootstrap demonstrations.")
    p.add_argument("--bootstrap_target_successes", type=int, default=500, help="Successful S2 trajectories retained for the base S1 model; 0 keeps all.")
    p.add_argument("--eval_n_per_family", type=int, default=10000, help="Held-out benchmark scenarios per family.")
    p.add_argument("--train_seed", type=int, default=7)
    p.add_argument("--eval_seed", type=int, default=8)
    p.add_argument("--s2_solver", choices=["cbf", "mpc"], default="cbf")
    p.add_argument("--train_source", choices=["s2", "selected", "all_success"], default="all_success")
    p.add_argument("--train_trajectories", type=int, default=0)
    p.add_argument("--train_epochs", type=int, default=40)
    p.add_argument("--train_batch", type=int, default=64)
    p.add_argument("--train_lr", type=float, default=3e-4)
    p.add_argument("--output_dir", default="input/nl")
    p.add_argument("--assets_dir", default="db/by_env/{family}_nl")
    p.add_argument("--results_dir", default="output/bootstrap_{family}_nl")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run(cmd: List[str], *, cwd: Path, dry_run: bool) -> None:
    print("\n[cmd]", " ".join(cmd))
    if dry_run:
        return
    for path in (cwd, cwd / "sofai", cwd / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from solvers._s2_common import resolve_mplconfigdir

    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["MPLCONFIGDIR"] = str(resolve_mplconfigdir(cwd, env.get("MPLCONFIGDIR")))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def resolve_path(root: Path, spec: str, *, family: str) -> Path:
    value = spec.format(family=family)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def successful_trajectory_count(path: Path) -> int:
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if bool(json.loads(line).get("success")):
            count += 1
    return count


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    python = args.python
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    families: Sequence[str] = args.families or [args.family]

    for family in families:
        assets_dir = resolve_path(root, args.assets_dir, family=family)
        results_dir = resolve_path(root, args.results_dir, family=family)
        if not args.dry_run:
            assets_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

        train_dict_path = out_dir / f"benchmark_dualmp_nl_{family}_train_{family}.json"
        eval_dict_path = out_dir / f"benchmark_dualmp_nl_{family}_eval_{family}.json"
        model_path = assets_dir / "s1_policy_nonlinear.pth"
        dataset_path = assets_dir / "s1_nonlinear_dataset.npz"
        target_successes = max(0, int(args.bootstrap_target_successes))

        if not args.skip_generate:
            for split, count, seed in (
                ("train", args.train_n_per_family, args.train_seed),
                ("eval", args.eval_n_per_family, args.eval_seed),
            ):
                prefix = f"benchmark_dualmp_nl_{family}_{split}_{family}"
                if family == "long_slalom":
                    cmd = [
                        python,
                        "input/generate_long_slalom.py",
                        "--root",
                        str(root),
                        "--output_dir",
                        str(out_dir),
                        "--prefix",
                        prefix,
                        "--n_scenarios",
                        str(count),
                        "--seed",
                        str(seed),
                    ]
                else:
                    cmd = [
                        python,
                        "input/generate_nl_dict.py",
                        "--root",
                        str(root),
                        "--output_dir",
                        str(out_dir),
                        "--prefix",
                        f"benchmark_dualmp_nl_{family}_{split}",
                        "--n_per_family",
                        str(count),
                        "--seed",
                        str(seed),
                        "--families",
                        family,
                    ]
                run(cmd, cwd=root, dry_run=args.dry_run)

        if not args.skip_collect:
            collect_cmd = [
                    python,
                    "run_motion_planning_benchmarks.py",
                    "--root",
                    str(root),
                    "--input_dir",
                    str(out_dir),
                    "--patterns",
                    train_dict_path.name,
                    "--scenario_ids",
                    f"0-{args.train_n_per_family - 1}",
                    "--run_type",
                    "s2",
                    "--s2",
                    args.s2_solver,
                    "--timeout_sec",
                    "300",
                    "--workers",
                    "1",
                    "--out_dir",
                    str(results_dir),
                    "--out_prefix",
                    f"{train_dict_path.stem}_{args.s2_solver}_bootstrap",
            ]
            if target_successes:
                collect_cmd.extend(["--stop_after_successes", str(target_successes)])
            run(collect_cmd, cwd=root, dry_run=args.dry_run)

        if not args.skip_train:
            jsonl = results_dir / f"{train_dict_path.stem}_{args.s2_solver}_bootstrap_runs.jsonl"
            if not args.dry_run and target_successes:
                successful = successful_trajectory_count(jsonl)
                if successful < target_successes:
                    print(
                        f"[warn] {family} collected {successful}/{target_successes} successful "
                        f"{args.s2_solver.upper()} bootstrap trajectories; training on all available successes."
                    )
            run(
                [
                    python,
                    "script/train_s1_nonlinear.py",
                    "--root",
                    str(root),
                    "--dictionary",
                    str(train_dict_path),
                    "--results_jsonl",
                    str(jsonl),
                    "--out_model",
                    str(model_path),
                    "--out_dataset",
                    str(dataset_path),
                    "--source",
                    args.train_source,
                    "--max_trajectories",
                    str(target_successes or args.train_trajectories),
                    "--epochs",
                    str(args.train_epochs),
                    "--batch",
                    str(args.train_batch),
                    "--lr",
                    str(args.train_lr),
                ],
                cwd=root,
                dry_run=args.dry_run,
            )

        print("\n[done]")
        print(f"[family] {family}")
        print(f"[train_dictionary] {train_dict_path}")
        print(f"[eval_dictionary] {eval_dict_path}")
        print(f"[model] {model_path}")
        print(f"[dataset] {dataset_path}")
        print(f"[results] {results_dir}")


if __name__ == "__main__":
    main()
