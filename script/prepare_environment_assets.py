#!/usr/bin/env python3
"""Generate per-environment S1 assets and benchmark dictionaries.

Defaults match the planned large run:
  - 500 successful S2 trajectories per environment for the S1 database
  - one neural policy trained from the same trajectories per environment
  - 10k benchmark scenarios per environment

intended large run:

cd /Users/apple/Documents/GitHub/System-1-and-System-2-in-Motion-Planning

PYTHONDONTWRITEBYTECODE=1 \

/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/prepare_environment_assets.py \
  --root /Users/apple/Documents/GitHub/System-1-and-System-2-in-Motion-Planning \
  --families all \
  --training_trajectories 500 \
  --benchmark_instances 10000 \
  --seed 7 \
  --max_attempts 20000 \
  --train_epochs 25 \
  --train_batch 128 \
  --train_lr 5e-4 \
  --assets_dir db/by_env \
  --benchmark_dir input/benchmarks_10k



sample small run:

cd /Users/apple/Documents/GitHub/System-1-and-System-2-in-Motion-Planning

PYTHONDONTWRITEBYTECODE=1 \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 script/prepare_environment_assets.py \
  --root /Users/apple/Documents/GitHub/System-1-and-System-2-in-Motion-Planning \
  --families dense_clutter \
  --training_trajectories 500 \
  --benchmark_instances 10000 \
  --seed 7 \
  --max_attempts 20000 \
  --train_epochs 25 \
  --train_batch 128 \
  --train_lr 5e-4 \
  --assets_dir db/by_env \
  --benchmark_dir input/benchmarks_10k

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


FAMILIES = [
    "small_open",
    "large_sparse",
    "dense_clutter",
    "wall_gap",
    "serial_walls",
    "maze_branching",
    "bugtrap",
]


def run(cmd: List[str], *, cwd: Path, dry_run: bool = False) -> None:
    print("\n[cmd]", " ".join(cmd))
    if dry_run:
        return
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def selected_families(raw: Iterable[str]) -> List[str]:
    names = list(raw)
    if not names or names == ["all"]:
        return list(FAMILIES)
    missing = [name for name in names if name not in FAMILIES]
    if missing:
        raise SystemExit(f"Unknown environment(s): {', '.join(missing)}")
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--families", nargs="+", default=["all"])
    p.add_argument("--training_trajectories", type=int, default=500)
    p.add_argument("--benchmark_instances", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_attempts", type=int, default=20000)
    p.add_argument("--train_epochs", type=int, default=25)
    p.add_argument("--train_batch", type=int, default=128)
    p.add_argument("--train_lr", type=float, default=5e-4)
    p.add_argument("--assets_dir", default="db/by_env")
    p.add_argument("--benchmark_dir", default="input/benchmarks_10k")
    p.add_argument("--skip_data", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_benchmarks", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    families = selected_families(args.families)
    assets_root = root / args.assets_dir
    benchmark_dir = root / args.benchmark_dir

    for family in families:
        env_dir = assets_root / family
        env_dir.mkdir(parents=True, exist_ok=True)

        dataset = env_dir / "nn_dataset_maze.npz"
        traj = env_dir / "s1_sfcbf_success_trajs.npz"
        db = env_dir / "S1_database_maze.json"
        scenarios = env_dir / "benchmark_scenarios_maze_s1_db.json"
        report = env_dir / "diversity_report.json"
        model = env_dir / "s1_policy_control_cnn.pth"

        if not args.skip_data:
            run(
                [
                    args.python,
                    "solvers/base/make_diverse_training_data_maze.py",
                    "--target_trajectories",
                    str(args.training_trajectories),
                    "--map_types",
                    family,
                    "--difficulties",
                    "benchmark",
                    "--out_npz",
                    str(dataset),
                    "--traj_out",
                    str(traj),
                    "--db_out",
                    str(db),
                    "--scenarios_out",
                    str(scenarios),
                    "--report_out",
                    str(report),
                    "--seed",
                    str(args.seed),
                    "--max_attempts",
                    str(args.max_attempts),
                ],
                cwd=root,
                dry_run=args.dry_run,
            )

        if not args.skip_train:
            run(
                [
                    args.python,
                    "solvers/base/train_nn_policy.py",
                    "--dataset",
                    str(dataset),
                    "--model_out",
                    str(model),
                    "--epochs",
                    str(args.train_epochs),
                    "--batch",
                    str(args.train_batch),
                    "--lr",
                    str(args.train_lr),
                    "--lambda_u",
                    "1.0",
                    "--lambda_next",
                    "1.0",
                    "--lambda_dir",
                    "0.5",
                    "--lambda_speed",
                    "1.0",
                    "--lambda_progress",
                    "1.0",
                    "--progress_fraction",
                    "0.9",
                ],
                cwd=root,
                dry_run=args.dry_run,
            )

    if not args.skip_benchmarks:
        run(
            [
                args.python,
                "input/generate_benchmark_dictionaries.py",
                "--root",
                str(root),
                "--output_dir",
                str(benchmark_dir),
                "--prefix",
                "benchmark_dualmp",
                "--n_per_family",
                str(args.benchmark_instances),
                "--seed",
                str(args.seed),
                "--families",
                *families,
                "--write_combined",
            ],
            cwd=root,
            dry_run=args.dry_run,
        )

    print("\n[done] prepared environments:", ", ".join(families))
    print("[assets]", assets_root)
    print("[benchmarks]", benchmark_dir)


if __name__ == "__main__":
    main()
