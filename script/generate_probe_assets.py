#!/usr/bin/env python3
"""Generate held-out nonlinear probe dictionaries for S1 CL evaluation.

Example:
  python script/generate_probe_assets.py \
    --families dense_clutter small_open large_sparse wall_gap serial_walls maze_branching bugtrap \
    --n_per_family 500 \
    --seed 700
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_FAMILIES = [
    "dense_clutter",
    "small_open",
    "large_sparse",
    "wall_gap",
    "serial_walls",
    "maze_branching",
    "bugtrap",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    p.add_argument("--n_per_family", type=int, default=200, help="Fixed held-out S1-only evaluation scenarios per family.")
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--output_dir", default="input/nl")
    p.add_argument("--write_combined", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd))
    if not dry_run:
        subprocess.run(list(cmd), cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    for i, family in enumerate(args.families):
        prefix = f"benchmark_dualmp_nl_{family}_probe"
        cmd = [
            sys.executable,
            "input/generate_nl_dict.py",
            "--root",
            str(root),
            "--output_dir",
            str(output_dir),
            "--prefix",
            prefix,
            "--n_per_family",
            str(args.n_per_family),
            "--seed",
            str(int(args.seed) + i),
            "--families",
            family,
        ]
        if args.write_combined:
            cmd.append("--write_combined")
        run(cmd, cwd=root, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
