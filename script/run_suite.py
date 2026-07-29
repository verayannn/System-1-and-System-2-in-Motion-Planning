#!/usr/bin/env python3
"""Run nonlinear benchmark suites.

Modes:
  - s1_neural
  - s2_cbf
  - s2_mpc
  - sofai_cbf_cl
  - sofai_mpc_cl
  - sofai_mpc_warm_cl

The continual-learning modes run SOFAI in strict fallback mode:
S1 is attempted first, S2 is only attempted when S1 fails, and retraining uses
successful trajectories accumulated from bootstrap plus completed blocks.
``sofai_mpc_warm_cl`` uses the failed S1 trajectory to warm-start S2 MPC;
``sofai_mpc_cl`` does not.


for the server:


families=(
  large_sparse
  dense_clutter
  serial_walls
  maze_branching
  long_slalom
  bugtrap
)
for family in "${families[@]}"; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python script/run_suite.py \
    --dictionary "input/nl/benchmark_dualmp_nl_${family}_eval_${family}.json" \
    --bootstrap_results_dir "output/bootstrap_${family}_nl" \
    --assets_dir "db/by_env/${family}_nl" \
    --out_dir "output/benchmark_runs/nl_${family}_suite" \
    --scenario_ids 0-499 \
    --block_size 100 \
    --workers 16 \
    --timeout_sec 60 \
    --block_order shuffled \
    --block_seed 42 \
    --cl_bootstrap_solver auto \
    --configs sofai_cbf_cl sofai_mpc_cl sofai_mpc_warm_cl s1_neural s2_cbf s2_mpc \
    --cl_train_mode replay_dagger \
    --replay_fraction 0.60 \
    --dagger_states_per_scenario 4 \
    --train_source s2 \
    --bootstrap_success_weight 1.0 \
    --dagger_success_weight 1.0 \
    --train_epochs 12 \
    --train_batch 64 \
    --train_lr 0.0001 \
    --train_device cpu \
    --probe_dictionary "input/nl/benchmark_dualmp_nl_${family}_probe_${family}.json" \
    --probe_scenario_ids 0-499
done





"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


## MODES = ("sofai_mpc_cl", "sofai_mpc_warm_cl")

MODES = (
    "s1_neural",
    "s2_cbf",
    "s2_mpc",
    "sofai_cbf_cl",
    "sofai_mpc_cl",
    "sofai_mpc_warm_cl",
)

## "s2_mpc_do",

# The legacy cumulative mode only labels MPC arms. Replay-DAgger is teacher
# matched and labels CBF arms with CBF as well.
DAGGER_SOLVERS = frozenset({"mpc", "mpc_warm"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dictionary", default="input/nl/benchmark_dualmp_nl_bugtrap_eval_bugtrap.json")
    p.add_argument("--bootstrap_results_dir", default="output/bootstrap_bugtrap_nl")
    p.add_argument(
        "--cl_bootstrap_solver",
        choices=["auto", "mpc", "mpc_do", "cbf"],
        default="auto",
        help=(
            "System 2 source for the initial S1 training trajectories. 'auto' gives each arm the "
            "teacher it falls back to, so the base checkpoint and every added demonstration come "
            "from one solver. Naming a single solver forces it on every arm, which mixes teachers "
            "with different action magnitudes and leaves the policy regressing contradictory labels."
        ),
    )
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
    p.add_argument(
        "--train_rollout_horizon",
        type=int,
        default=8,
        help="Differentiable rollout horizon passed to continual S1 training.",
    )
    p.add_argument(
        "--train_rollout_every",
        type=int,
        default=8,
        help="Run the rollout loss once every N supervised training batches.",
    )
    p.add_argument(
        "--train_rollout_filter_mode",
        choices=["policy", "none"],
        default="policy",
        help=(
            "Safety filter used inside continual training's differentiable rollout loss. "
            "'policy' matches runtime S1; 'none' is the legacy raw-action objective."
        ),
    )
    p.add_argument(
        "--train_device",
        default="cpu",
        help="PyTorch device passed to continual S1 training, e.g. cuda or cuda:0.",
    )
    p.add_argument(
        "--train_dt_nom",
        type=float,
        default=0.075,
        help=(
            "Integrator step S1 is trained and executed at. Keep it equal to the step the S2 "
            "teachers record, and to the step of the base checkpoint in --assets_dir, otherwise "
            "block 0 is evaluated under a different S1 configuration than later blocks."
        ),
    )
    p.add_argument(
        "--train_n_steps_nom",
        type=int,
        default=900,
        help=(
            "Steps of the nominal rollout that paints the corridor in the situation vector. "
            "Must also match the base checkpoint for the blocks to stay comparable."
        ),
    )
    p.add_argument("--train_source", choices=["s2", "selected", "all_success", "fallback_success"], default="all_success")
    p.add_argument("--fallback_success_weight", type=float, default=5.0)
    p.add_argument("--s1_success_weight", type=float, default=1.0)
    p.add_argument("--bootstrap_success_weight", type=float, default=1.0)
    p.add_argument("--dagger_states_per_scenario", type=int, default=0)
    p.add_argument("--dagger_success_weight", type=float, default=10.0)
    p.add_argument(
        "--cl_train_mode",
        choices=["cumulative", "replay_dagger"],
        default="cumulative",
        help=(
            "cumulative preserves the original retrain-from-base scheme. replay_dagger keeps "
            "only fixed S2 bootstrap replay plus validated S2 labels from S1-visited states, "
            "then fine-tunes the preceding block model."
        ),
    )
    p.add_argument(
        "--replay_fraction",
        type=float,
        default=0.60,
        help="Fixed-teacher replay share in replay_dagger sampler mixing.",
    )
    p.add_argument("--block_order", choices=["shuffled", "sequential"], default="shuffled")
    p.add_argument("--block_seed", type=int, default=42)
    p.add_argument("--probe_dictionary", default="")
    p.add_argument("--probe_scenario_ids", default="")
    p.add_argument("--probe_workers", type=int, default=0)
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


def validate_training_geometry(
    base_model: Path,
    *,
    train_dt_nom: float,
    train_n_steps_nom: int,
) -> None:
    """Keep the base evaluation and every retrained CL block comparable."""
    import torch

    checkpoint = torch.load(str(base_model), map_location="cpu", weights_only=False)
    dataset_meta = dict(checkpoint.get("meta", {}).get("dataset_meta", {}) or {})
    checkpoint_dt = dataset_meta.get("dt_nom")
    checkpoint_steps = dataset_meta.get("n_steps_nom")
    if checkpoint_dt is None or checkpoint_steps is None:
        raise RuntimeError(
            f"Base checkpoint lacks dt_nom/n_steps_nom metadata: {base_model}. "
            "Regenerate the base checkpoint before running continual learning."
        )
    if abs(float(train_dt_nom) - float(checkpoint_dt)) > 1e-9 or int(train_n_steps_nom) != int(checkpoint_steps):
        raise RuntimeError(
            "Training geometry does not match the base checkpoint. "
            f"base dt_nom={float(checkpoint_dt)}, n_steps_nom={int(checkpoint_steps)}; "
            f"requested dt_nom={float(train_dt_nom)}, n_steps_nom={int(train_n_steps_nom)}. "
            "Use the base values for this suite, or regenerate the base checkpoint first."
        )


def validate_bootstrap_scenarios(bootstrap_jsonl: Path, train_dictionary: Path) -> None:
    """Reject stale bootstrap results before a suite spends time on block zero."""
    train_count = read_count(train_dictionary)
    indices: List[int] = []
    with bootstrap_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            indices.append(int(row.get("scenario_index", row.get("scenario_id", -1))))
    invalid = [index for index in indices if index < 0 or index >= train_count]
    if invalid:
        raise RuntimeError(
            "Bootstrap/train dictionary mismatch: "
            f"{bootstrap_jsonl.name} contains scenario IDs through {max(indices)}, but "
            f"{train_dictionary.name} has only {train_count} scenarios. "
            "Regenerate the train dictionary and its bootstrap results together."
        )


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


# The System 2 whose demonstrations should teach each continual-learning arm.
# mpc_warm falls back to MPC with a warm start, so plain MPC demonstrations are
# the matching teacher.
ARM_BOOTSTRAP_SOLVER = {
    "sofai_cbf_cl": "cbf",
    "sofai_mpc_cl": "mpc",
    "sofai_mpc_warm_cl": "mpc",
}


def arm_bootstrap_solver(cfg: str, requested: str) -> str:
    if str(requested) != "auto":
        return str(requested)
    return ARM_BOOTSTRAP_SOLVER.get(cfg, "mpc")


def arm_base_model(assets_dir: Path, root: Path, solver: str, override: str) -> Path:
    """Prefer the per-solver base checkpoint written by prepare_environment_assets."""
    if str(override).strip():
        model = Path(override).expanduser()
    else:
        model = assets_dir / f"s1_policy_nonlinear_{solver}.pth"
        if not model.exists():
            model = assets_dir / "s1_policy_nonlinear.pth"
    return model if model.is_absolute() else root / model


def base_env(root: Path, model_path: Path, mplconfigdir: str) -> Dict[str, str]:
    for path in (root, root / "sofai", root / "solvers"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
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
    same_process: bool = False,
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
    if same_process:
        cmd.append("--same_process")
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
    train_rollout_horizon: int,
    train_rollout_every: int,
    train_rollout_filter_mode: str,
    train_device: str,
    train_dt_nom: float,
    train_n_steps_nom: int,
    fallback_success_weight: float,
    s1_success_weight: float,
    bootstrap_success_weight: float,
    dagger_success_weight: float,
    target_replay_fraction: float,
    preserve_init_norm: bool,
    audit_json: Path,
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
    # argparse uses nargs="+" here, so this must be one option followed by
    # every JSONL. Repeating the option silently retained only the last block.
    cmd.extend(["--results_jsonl", *(str(path.resolve()) for path in results_jsonl)])
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
        "--rollout_horizon",
        str(train_rollout_horizon),
        "--rollout_every",
        str(train_rollout_every),
        "--rollout_filter_mode",
        str(train_rollout_filter_mode),
        "--device",
        str(train_device),
        "--dt_nom",
        str(train_dt_nom),
        "--n_steps_nom",
        str(train_n_steps_nom),
        "--fallback_success_weight",
        str(fallback_success_weight),
        "--s1_success_weight",
        str(s1_success_weight),
        "--bootstrap_success_weight",
        str(bootstrap_success_weight),
        "--dagger_success_weight",
        str(dagger_success_weight),
        "--target_replay_fraction",
        str(target_replay_fraction),
        "--audit_json",
        str(audit_json),
    ])
    if preserve_init_norm:
        cmd.append("--preserve_init_norm")
    run(cmd, cwd=root, env=env, dry_run=dry_run)
    return out_model


def collect_s2_dagger(
    *,
    root: Path,
    python: str,
    dictionary: Path,
    model: Path,
    scenario_ids: Sequence[int],
    states_per_scenario: int,
    s2_solver: str,
    out_jsonl: Path,
    env: Dict[str, str],
    dry_run: bool,
) -> Path:
    cmd = [
        python,
        "script/collect_mpc_dagger.py",
        "--root",
        str(root),
        "--dictionary",
        str(dictionary),
        "--s1_model",
        str(model),
        "--scenario_ids",
        ",".join(str(index) for index in scenario_ids),
        "--states_per_scenario",
        str(states_per_scenario),
        "--s2_solver",
        s2_solver,
        "--out_jsonl",
        str(out_jsonl),
    ]
    run(cmd, cwd=root, env=env, dry_run=dry_run)
    return out_jsonl


def verify_cumulative_training_audit(audit_json: Path, expected_jsonls: Sequence[Path], previous_count: int) -> int:
    if not audit_json.is_file():
        raise RuntimeError(f"Training audit was not written: {audit_json}")
    audit = json.loads(audit_json.read_text())
    expected = [str(path.resolve()) for path in expected_jsonls]
    actual = [str(path) for path in audit.get("input_jsonls", [])]
    if sorted(actual) != sorted(expected):
        raise RuntimeError(
            "Cumulative training audit mismatch. "
            f"expected JSONLs={expected}; recorded JSONLs={actual}"
        )
    if int(audit.get("selected_success_count", -1)) != int(audit.get("trajectory_count", -2)):
        raise RuntimeError(f"Training audit dropped successful trajectories: {audit_json}")
    count = int(audit.get("trajectory_count", 0))
    if count < previous_count:
        raise RuntimeError(
            f"Cumulative training trajectory count decreased from {previous_count} to {count}: {audit_json}"
        )
    return count


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
    block_ids = list(scenario_ids)
    if args.block_order == "shuffled":
        random.Random(int(args.block_seed)).shuffle(block_ids)
    blocks = chunks(block_ids, block_size)
    probe_dictionary = Path(args.probe_dictionary).expanduser() if str(args.probe_dictionary).strip() else dictionary
    if not probe_dictionary.is_absolute():
        probe_dictionary = root / probe_dictionary
    if not probe_dictionary.exists() and not args.dry_run:
        raise FileNotFoundError(probe_dictionary)
    probe_count = read_count(probe_dictionary) if probe_dictionary.exists() else count
    probe_ids = parse_ids(args.probe_scenario_ids, probe_count) if str(args.probe_scenario_ids).strip() else []
    workers = effective_workers(args.workers)
    probe_workers = effective_workers(args.probe_workers or args.workers)

    # Standalone S1/S2 arms keep the legacy unsuffixed checkpoint; each CL arm
    # resolves its own teacher-matched checkpoint below.
    if args.s1_model:
        default_init_model = Path(args.s1_model).expanduser()
    else:
        default_init_model = assets_dir / "s1_policy_nonlinear.pth"
    if not default_init_model.is_absolute():
        default_init_model = root / default_init_model
    if not default_init_model.exists() and not args.dry_run:
        raise FileNotFoundError(default_init_model)
    if not args.dry_run:
        validate_training_geometry(
            default_init_model,
            train_dt_nom=args.train_dt_nom,
            train_n_steps_nom=args.train_n_steps_nom,
        )

    init_model = default_init_model
    env = base_env(root, default_init_model, args.mplconfigdir)
    manifest: Dict[str, object] = {
        "root": str(root),
        "dictionary": str(dictionary),
        "scenario_count": len(scenario_ids),
        "block_size": block_size,
        "block_order": args.block_order,
        "block_seed": int(args.block_seed),
        "train_device": str(args.train_device),
        "train_rollout_horizon": int(args.train_rollout_horizon),
        "train_rollout_every": int(args.train_rollout_every),
        "train_rollout_filter_mode": str(args.train_rollout_filter_mode),
        "train_dt_nom": float(args.train_dt_nom),
        "train_n_steps_nom": int(args.train_n_steps_nom),
        "blocks": blocks,
        "probe_dictionary": str(probe_dictionary) if probe_ids else "",
        "probe_scenario_ids": probe_ids,
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

        elif cfg in {"s2_mpc", "s2_mpc_do"}:
            run_benchmark(
                root=root,
                python=python,
                dictionary=dictionary,
                scenario_ids=scenario_ids,
                run_type="s2",
                s2="mpc_do" if cfg == "s2_mpc_do" else "mpc",
                out_dir=cfg_dir,
                out_prefix=cfg,
                env=env,
                timeout_sec=args.timeout_sec,
                workers=workers,
                dry_run=args.dry_run,
            )
            cfg_manifest["runs"].append({"prefix": cfg})

        else:
            ##solver = "mpc_warm" if cfg == "sofai_mpc_warm_cl" else "mpc_do" if "mpc_do" in cfg else "mpc"

            if cfg == "sofai_cbf_cl":
                solver = "cbf"
            elif cfg == "sofai_mpc_warm_cl":
                solver = "mpc_warm"
            elif "mpc_do" in cfg:
                solver = "mpc_do"
            else:
                solver = "mpc"

            bootstrap_solver = arm_bootstrap_solver(cfg, args.cl_bootstrap_solver)
            init_model = arm_base_model(assets_dir, root, bootstrap_solver, args.s1_model)
            if not init_model.exists() and not args.dry_run:
                raise FileNotFoundError(
                    f"Missing base checkpoint for {cfg} taught by {bootstrap_solver.upper()}: {init_model}. "
                    f"Run prepare_environment_assets.py --s2_solvers {bootstrap_solver}."
                )
            if not args.dry_run:
                validate_training_geometry(
                    init_model,
                    train_dt_nom=args.train_dt_nom,
                    train_n_steps_nom=args.train_n_steps_nom,
                )
            print(f"[{cfg}] teacher={bootstrap_solver} base_checkpoint={init_model}")

            current_model = init_model
            bootstrap_stem = dictionary.stem.replace("_eval_", "_train_")
            bootstrap_jsonl = bootstrap_results_dir / f"{bootstrap_stem}_{bootstrap_solver}_bootstrap_runs.jsonl"
            if not bootstrap_jsonl.is_file() and not args.dry_run:
                raise FileNotFoundError(
                    f"Missing {bootstrap_solver.upper()} base successful-trajectory JSONL: {bootstrap_jsonl}"
                )
            train_dictionary = dictionary.with_name(bootstrap_stem + ".json")
            if not args.dry_run and not train_dictionary.is_file():
                raise FileNotFoundError(f"Missing base training dictionary: {train_dictionary}")
            if not args.dry_run:
                validate_bootstrap_scenarios(bootstrap_jsonl, train_dictionary)
            cumulative_jsonls: List[Path] = [bootstrap_jsonl]
            dagger_jsonls: List[Path] = []
            if args.cl_train_mode == "replay_dagger" and int(args.dagger_states_per_scenario) <= 0:
                raise RuntimeError(
                    "replay_dagger requires --dagger_states_per_scenario > 0; otherwise there "
                    "is no online teacher supervision to mix with replay."
                )
            previous_trajectory_count = 0
            if probe_ids:
                # Establish a fixed S1-only baseline before any CL block is used.
                probe_dir = cfg_dir / "probe"
                base_probe_prefix = f"{cfg}_base_probe_s1"
                base_probe_jsonl = run_benchmark(
                    root=root,
                    python=python,
                    dictionary=probe_dictionary,
                    scenario_ids=probe_ids,
                    run_type="s1",
                    s2=solver,
                    out_dir=probe_dir,
                    out_prefix=base_probe_prefix,
                    env={**env, "SOFAI_NEW_S1_MODEL": str(init_model)},
                    timeout_sec=args.timeout_sec,
                    workers=probe_workers,
                    dry_run=args.dry_run,
                )
                cfg_manifest["base_probe"] = {
                    "prefix": base_probe_prefix,
                    "jsonl": str(base_probe_jsonl),
                    "model": str(init_model),
                }
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
                if args.cl_train_mode == "cumulative":
                    cumulative_jsonls.append(block_jsonl)
                dagger_jsonl = None
                # In replay-DAgger every arm queries its own teacher. The legacy
                # cumulative path retains its MPC-only behavior for compatibility.
                collect_dagger = (
                    args.cl_train_mode == "replay_dagger" or solver in DAGGER_SOLVERS
                ) and int(args.dagger_states_per_scenario) > 0
                if collect_dagger:
                    dagger_solver = "cbf" if solver == "cbf" else "mpc"
                    dagger_jsonl = cfg_dir / "dagger" / f"{prefix}_{dagger_solver}_recoveries.jsonl"
                    collect_s2_dagger(
                        root=root,
                        python=python,
                        dictionary=dictionary,
                        model=current_model,
                        scenario_ids=block_ids,
                        states_per_scenario=int(args.dagger_states_per_scenario),
                        s2_solver=dagger_solver,
                        out_jsonl=dagger_jsonl,
                        env=env,
                        dry_run=args.dry_run,
                    )
                    if args.cl_train_mode == "replay_dagger":
                        dagger_jsonls.append(dagger_jsonl)
                    else:
                        cumulative_jsonls.append(dagger_jsonl)
                next_model = cfg_dir / "models" / f"{prefix}_s1_policy_nonlinear.pth"
                next_dataset = cfg_dir / "datasets" / f"{prefix}_s1_nonlinear_dataset.npz"
                training_audit = cfg_dir / "audits" / f"{prefix}_training_audit.json"
                if args.cl_train_mode == "replay_dagger":
                    training_jsonls = [bootstrap_jsonl, *dagger_jsonls]
                    training_init_model = current_model
                    training_source = "s2"
                    target_replay_fraction = args.replay_fraction
                else:
                    training_jsonls = cumulative_jsonls
                    training_init_model = init_model
                    training_source = args.train_source
                    target_replay_fraction = -1.0
                # Legacy cumulative mode restarts from the frozen base. The
                # replay-DAgger mode carries the prior model forward while
                # retaining fixed teacher replay at every update.
                if not args.dry_run: ### retraining happens here
                    train_model(
                        root=root,
                        python=python,
                        dictionary=dictionary,
                        results_jsonl=training_jsonls,
                        init_model=training_init_model,
                        out_model=next_model,
                        out_dataset=next_dataset,
                        source=training_source,
                        max_trajectories=0,
                        train_epochs=args.train_epochs,
                        train_batch=args.train_batch,
                        train_lr=args.train_lr,
                        train_rollout_horizon=args.train_rollout_horizon,
                        train_rollout_every=args.train_rollout_every,
                        train_rollout_filter_mode=args.train_rollout_filter_mode,
                        train_device=args.train_device,
                        train_dt_nom=args.train_dt_nom,
                        train_n_steps_nom=args.train_n_steps_nom,
                        fallback_success_weight=args.fallback_success_weight,
                        s1_success_weight=args.s1_success_weight,
                        bootstrap_success_weight=args.bootstrap_success_weight,
                        dagger_success_weight=args.dagger_success_weight,
                        target_replay_fraction=target_replay_fraction,
                        preserve_init_norm=(args.cl_train_mode == "replay_dagger"),
                        audit_json=training_audit,
                        env=env,
                        dry_run=False,
                    )
                    previous_trajectory_count = verify_cumulative_training_audit(
                        training_audit,
                        training_jsonls,
                        previous_trajectory_count,
                    )
                current_model = next_model
                run_entry = {
                    "prefix": prefix,
                    "jsonl": str(block_jsonl),
                    "model": str(current_model),
                    "training_initialization": (
                        "prior_block" if args.cl_train_mode == "replay_dagger" else "base_checkpoint"
                    ),
                    "training_init_model": str(training_init_model),
                    "cl_train_mode": args.cl_train_mode,
                    "replay_fraction": target_replay_fraction,
                    "train_rollout_filter_mode": args.train_rollout_filter_mode,
                    "bootstrap_solver": bootstrap_solver,
                    "bootstrap_jsonl": str(bootstrap_jsonl),
                    "block_ids": block_ids,
                    "dagger_jsonl": "" if dagger_jsonl is None else str(dagger_jsonl),
                    "training_jsonls": [str(path) for path in training_jsonls],
                    "training_audit": str(training_audit),
                    "training_trajectory_count": previous_trajectory_count,
                }
                if probe_ids:
                    probe_dir = cfg_dir / "probe"
                    probe_prefix = f"{prefix}_probe_s1"
                    probe_jsonl = run_benchmark(
                        root=root,
                        python=python,
                        dictionary=probe_dictionary,
                        scenario_ids=probe_ids,
                        run_type="s1",
                        s2=solver,
                        out_dir=probe_dir,
                        out_prefix=probe_prefix,
                        env={**env, "SOFAI_NEW_S1_MODEL": str(current_model)},
                        timeout_sec=args.timeout_sec,
                        workers=probe_workers,
                        dry_run=args.dry_run,
                    )
                    run_entry["probe_jsonl"] = str(probe_jsonl)
                    run_entry["probe_prefix"] = probe_prefix
                cfg_manifest["runs"].append(run_entry)

        manifest["configs"][cfg] = cfg_manifest

    manifest_path = out_dir / "suite_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
