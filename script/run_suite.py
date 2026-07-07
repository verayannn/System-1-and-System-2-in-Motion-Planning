#!/usr/bin/env python3
"""Run the nonlinear dense-clutter benchmark suite.

Modes:
  - s1_neural
  - s2_cbf
  - s2_mpc
  - sofai_cbf_cl
  - sofai_mpc_cl

The continual-learning modes advance through 200-scenario blocks, run SOFAI
with all attempts recorded, and retrain the neural System 1 on every
successful trajectory in each block.

python script/run_suite.py --workers 3


PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
python script/run_suite.py \
  --dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --bootstrap_results_dir output/bootstrap_dense_clutter_nl \
  --assets_dir db/by_env/dense_clutter_nl \
  --out_dir output/benchmark_runs/nl_dense_clutter_suite \
  --scenario_ids 0-499 \
  --workers 3 \
  --configs sofai_mpc_cl

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

## MODES = ("s1_neural", "s2_cbf", "s2_mpc")

## MODES = ("s1_neural",)

MODES = ("s1_neural", "s2_mpc", "s2_cbf", "sofai_cbf_cl", "sofai_mpc_cl")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dictionary", default="input/nl/benchmark_dualmp_nl_bugtrap_eval_bugtrap.json")
    p.add_argument("--bootstrap_results_dir", default="output/bootstrap_bugtrap_nl")
    p.add_argument("--scenario_ids", default="0-499")
    p.add_argument("--block_size", type=int, default=100) ## block size for continual learning: continual learning happens after a block finishes
    p.add_argument("--configs", nargs="+", default=list(MODES))
    p.add_argument("--assets_dir", default="db/by_env/bugtrap_nl")
    p.add_argument("--out_dir", default="output/benchmark_runs/nl_bugtrap_suite")
    p.add_argument("--timeout_sec", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--s1_model", default="")
    p.add_argument("--train_epochs", type=int, default=30)
    p.add_argument("--train_batch", type=int, default=64)
    p.add_argument("--train_lr", type=float, default=3e-4)
    p.add_argument("--train_source", choices=["s2", "selected", "all_success"], default="all_success")
    p.add_argument("--mplconfigdir", default="")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run(cmd: List[str], *, cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    print("\n[cmd]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_count(path: Path) -> int:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} does not contain a scenario list")
    return len(data)


def parse_ids(raw: str, count: int) -> List[int]:
    raw = str(raw).strip()
    if raw.lower() in {"", "all"}:
        return list(range(count))
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(v.strip()) for v in part.split("-", 1)]
            lo, hi = sorted((lo, hi))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen = set()
    return [i for i in out if 0 <= i < count and not (i in seen or seen.add(i))]


def chunks(values: Sequence[int], size: int) -> List[List[int]]:
    return [list(values[i : i + size]) for i in range(0, len(values), size)]


def effective_workers(requested: int) -> int:
    return max(2, int(requested))


def base_env(root: Path, model_path: Path, mplconfigdir: str) -> Dict[str, str]:
    from solvers._s2_common import resolve_mplconfigdir

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(resolve_mplconfigdir(root, mplconfigdir))
    env["SOFAI_NEW_S1_MODEL"] = str(model_path)
    env.setdefault("SOFAI_NEW_S1_DEVICE", "cpu")
    return env


def run_benchmark(
    *,
    root: Path,
    python: str,
    dictionary: Path,
    scenario_ids: Sequence[int],
    run_type: str,
    s2: str,
    out_dir: Path,
    out_prefix: str,
    env: Dict[str, str],
    timeout_sec: float,
    workers: int,
    run_all_attempts: bool = False,
    dry_run: bool = False,
) -> Path:
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "run_motion_planning_benchmarks.py",
        "--root",
        str(root),
        "--input_dir",
        str(dictionary.parent),
        "--patterns",
        dictionary.name,
        "--scenario_ids",
        ",".join(str(i) for i in scenario_ids),
        "--run_type",
        run_type,
        "--s1",
        "neural",
        "--s2",
        s2,
        "--timeout_sec",
        str(timeout_sec),
        "--workers",
        str(workers),
        "--out_dir",
        str(out_dir),
        "--out_prefix",
        out_prefix,
    ]
    if run_all_attempts:
        cmd.append("--run_all_attempts")
    run(cmd, cwd=root, env=env, dry_run=dry_run)
    return out_dir / f"{out_prefix}_runs.jsonl"


def train_model(
    *,
    root: Path,
    python: str,
    dictionary: Path,
    results_jsonl: Sequence[Path],
    init_model: Path,
    out_model: Path,
    out_dataset: Path,
    source: str,
    max_trajectories: int,
    train_epochs: int,
    train_batch: int,
    train_lr: float,
    env: Dict[str, str],
    dry_run: bool,
) -> Path:
    if not dry_run:
        out_model.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "script/train_s1_nonlinear.py",
        "--root",
        str(root),
        "--dictionary",
        str(dictionary),
        "--out_model",
        str(out_model),
        "--out_dataset",
        str(out_dataset),
        "--init_model",
        str(init_model),
    ]
    for path in results_jsonl:
        cmd.extend(["--results_jsonl", str(path)])
    cmd.extend([
        "--source",
        source,
        "--max_trajectories",
        str(max_trajectories),
        "--epochs",
        str(train_epochs),
        "--batch",
        str(train_batch),
        "--lr",
        str(train_lr),
    ])
    run(cmd, cwd=root, env=env, dry_run=dry_run)
    return out_model


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    python = args.python
    dictionary = Path(args.dictionary).expanduser()
    if not dictionary.is_absolute():
        dictionary = root / dictionary
    if not dictionary.exists() and not args.dry_run:
        raise FileNotFoundError(dictionary)

    assets_dir = Path(args.assets_dir).expanduser()
    if not assets_dir.is_absolute():
        assets_dir = root / assets_dir
    if not args.dry_run:
        assets_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_results_dir = Path(args.bootstrap_results_dir).expanduser()
    if not bootstrap_results_dir.is_absolute():
        bootstrap_results_dir = root / bootstrap_results_dir

    if dictionary.exists():
        count = read_count(dictionary)
    else:
        count = 1000 if str(args.scenario_ids).strip().lower() == "all" else 10**6
    scenario_ids = parse_ids(args.scenario_ids, count)
    if not scenario_ids:
        raise SystemExit("No scenario ids selected.")
    block_size = max(1, int(args.block_size))
    blocks = chunks(scenario_ids, block_size)
    workers = effective_workers(args.workers)

    if args.s1_model:
        init_model = Path(args.s1_model).expanduser()
    else:
        init_model = assets_dir / "s1_policy_nonlinear.pth"
    if not init_model.is_absolute():
        init_model = root / init_model
    if not init_model.exists() and not args.dry_run:
        raise FileNotFoundError(init_model)

    env = base_env(root, init_model, args.mplconfigdir)
    manifest: Dict[str, object] = {
        "root": str(root),
        "dictionary": str(dictionary),
        "scenario_count": len(scenario_ids),
        "block_size": block_size,
        "configs": {},
    }

    configs = set(args.configs or [])
    unknown = sorted(configs.difference(MODES))
    if unknown:
        raise SystemExit(f"Unknown configs: {', '.join(unknown)}")

    for cfg in MODES:
        if cfg not in configs:
            continue

        cfg_dir = out_dir / cfg
        if not args.dry_run:
            cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_manifest: Dict[str, object] = {"mode": cfg, "runs": []}

        if cfg == "s1_neural":
            run_benchmark(
                root=root,
                python=python,
                dictionary=dictionary,
                scenario_ids=scenario_ids,
                run_type="s1",
                s2="cbf",
                out_dir=cfg_dir,
                out_prefix=cfg,
                env=env,
                timeout_sec=args.timeout_sec,
                workers=workers,
                dry_run=args.dry_run,
            )
            cfg_manifest["runs"].append({"prefix": cfg, "model": str(init_model)})

        elif cfg == "s2_cbf":
            run_benchmark(
                root=root,
                python=python,
                dictionary=dictionary,
                scenario_ids=scenario_ids,
                run_type="s2",
                s2="cbf",
                out_dir=cfg_dir,
                out_prefix=cfg,
                env=env,
                timeout_sec=args.timeout_sec,
                workers=workers,
                dry_run=args.dry_run,
            )
            cfg_manifest["runs"].append({"prefix": cfg})

        elif cfg == "s2_mpc":
            run_benchmark(
                root=root,
                python=python,
                dictionary=dictionary,
                scenario_ids=scenario_ids,
                run_type="s2",
                s2="mpc",
                out_dir=cfg_dir,
                out_prefix=cfg,
                env=env,
                timeout_sec=args.timeout_sec,
                workers=workers,
                dry_run=args.dry_run,
            )
            cfg_manifest["runs"].append({"prefix": cfg})

        else:
            solver = "cbf" if "cbf" in cfg else "mpc" ## continual learning happens here automatically
            current_model = init_model
            bootstrap_stem = dictionary.stem.replace("_eval_", "_train_")
            bootstrap_jsonl = bootstrap_results_dir / f"{bootstrap_stem}_{solver}_bootstrap_runs.jsonl"
            cumulative_jsonls: List[Path] = [bootstrap_jsonl]
            for block_idx, block_ids in enumerate(blocks):
                prefix = f"{cfg}_block{block_idx:02d}"
                block_run_dir = cfg_dir / "runs"
                block_jsonl = run_benchmark(
                    root=root,
                    python=python,
                    dictionary=dictionary,
                    scenario_ids=block_ids,
                    run_type="sofai",
                    s2=solver,
                    out_dir=block_run_dir,
                    out_prefix=prefix,
                    env={**env, "SOFAI_NEW_S1_MODEL": str(current_model)},
                    timeout_sec=args.timeout_sec,
                    workers=workers,
                    dry_run=args.dry_run,
                )
                cumulative_jsonls.append(block_jsonl)
                next_model = cfg_dir / "models" / f"{prefix}_s1_policy_nonlinear.pth"
                next_dataset = cfg_dir / "datasets" / f"{prefix}_s1_nonlinear_dataset.npz"
                if not args.dry_run: ### retraining happens here
                    train_model(
                        root=root,
                        python=python,
                        dictionary=dictionary,
                        results_jsonl=cumulative_jsonls,
                        init_model=current_model,
                        out_model=next_model,
                        out_dataset=next_dataset,
                        source=args.train_source,
                        max_trajectories=0,
                        train_epochs=args.train_epochs,
                        train_batch=args.train_batch,
                        train_lr=args.train_lr,
                        env=env,
                        dry_run=False,
                    )
                current_model = next_model
                cfg_manifest["runs"].append(
                    {
                        "prefix": prefix,
                        "jsonl": str(block_jsonl),
                        "model": str(current_model),
                    }
                )

        manifest["configs"][cfg] = cfg_manifest

    manifest_path = out_dir / "suite_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
