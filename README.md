# Dual Process Motion Planning

This repository contains the code used for the AAAI 2027 submission *Dual Process Motion Planning*. It implements a SOFAI-based dual-process motion-planning stack for 2D obstacle-avoidance tasks.

The core idea is to combine:

- **System 1**: a fast experience-driven planner
  - neural policy (`neural`)
- **System 2**: a slower but more reliable online solver
  - model predictive control (`mpc`)
  - control barrier functions (`cbf`)
- **Metacognitive arbitration**: the SOFAI controller decides whether to accept the System 1 proposal or fall back to System 2
- **Continual-learning variants**: successful System 2 trajectories will be used for retraining System 1 to improve later runs


## Installation

The recommended setup uses Python 3.10 or 3.11, `uv`, and the repo-local
`safe_control/acados` tree. MPC additionally requires a C/C++ compiler, CMake,
Make, and Git. System 1 and the CBF System 2 solver do not require acados.

The Python environment alone is insufficient for MPC: acados native shared
libraries must be built for the same architecture as the Python interpreter.

### 1. Clone the repository

```bash
git clone <repository-url>
cd System-1-and-System-2-in-Motion-Planning
```

`safe_control/acados` is vendored in this repository, including its required
BLASFEO and HPIPM sources. Do not run `git -C safe_control/acados submodule
update ...`: this vendored directory is not an independent Git worktree.

### 2. Install prerequisites

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake python3-dev
```

On macOS:

```bash
xcode-select --install
brew install cmake uv
```

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) by its
official instructions if it is not available through your package manager.

### 3. Create the Python environment

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra acados-template
```

The root `pyproject.toml` installs the repo itself in editable mode and exposes the local source packages directly:

- `sofai/`
- `safe_control/`

Do not install `./sofai` and `./safe_control` as separate editable packages for the normal experiment environment; that reintroduces nested build-system resolution.

The lock file pins a tested dependency set; use `uv sync` rather than manually
upgrading individual scientific packages.

### 4. Build and register acados

```bash
python script/setup_acados.py --jobs 4
source .env.acados
```

The setup script:

- builds `safe_control/acados` with CMake
- checks that `libacados`, `libblasfeo`, and `libhpipm` exist
- installs `acados_template` into the active Python environment
- writes `.env.acados` with the correct library path variables for macOS or Linux

It builds the portable BLASFEO `GENERIC` target. This is deliberate: the
vendored upstream snapshot does not track BLASFEO's ignored ISA-probing assembly
files, and `GENERIC` avoids that fragile CPU-specific probe on both macOS and
Linux.

Source `.env.acados` in every new shell before running an MPC command. It sets
the acados source and dynamic-library paths for the local build.

If you want to use a separate acados checkout instead of the repo-local one:

```bash
python script/setup_acados.py --acados_root /path/to/acados --skip_build
source .env.acados
```

### 5. Verify the installation

```bash
python - <<'PY'
import acados_template
import sofai_tool
import safe_control
from solvers._s2_common import detect_acados_root
from solvers.S2_mpc import solve_MPC_with_info

root = detect_acados_root()
print("acados_template import: ok")
print("SOFAI import: ok")
print("safe_control import: ok")
print("MPC solver import: ok")
print("acados root:", root)
assert root is not None, "acados shared libraries were not found"
PY
```

If this command segfaults (exit code 139) after a prior or interrupted acados
build, rebuild the generated native artifacts for the active machine:

```bash
rm -rf safe_control/acados/build safe_control/acados/lib
python script/setup_acados.py --clean --jobs 4
source .env.acados
```

### Pip or Conda fallback

If `uv` is not available, use Python 3.10 or 3.11:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python script/setup_acados.py --jobs 4
source .env.acados
```

The repository scripts automatically choose a writable `MPLCONFIGDIR`, so you do not normally need to set it manually. If you want to override it, use a writable path such as `/tmp/mpl` on both macOS and Linux.

## Directory Structure

```text
System-1-and-System-2-in-Motion-Planning/
├── README.md                           # Project documentation and usage
├── motion_planning_solver.py           # Single-scenario planner entry point
├── run_motion_planning_benchmarks.py   # Run motion-planning benchmarks and save JSONL/CSV metrics
├── visualize_mp.py                     # Single-scenario visualisation
├── analyze_suite_results.py            # Unified benchmark analysis pipeline
├── requirements.txt / pyproject.toml   # Python environment and package metadata
├── input/                              # Benchmark dictionaries and generation utilities
├── db/by_env/                          # Per-environment neural S1 datasets and checkpoints
├── solvers/                            # S1 policies, MPC/CBF S2 solvers, quality metrics
├── script/                             # Experiment orchestration utilities
│   ├── prepare_environment_assets.py   # Create per-environment S1 assets
│   ├── run_suite.py                    # Run benchmark suites and continual learning
│   └── train_s1_nonlinear.py           # Train or fine-tune the neural S1 policy
├── sofai/                              # Vendored SOFAI framework
└── safe_control/                       # Vendored CBF and acados dependencies
```


## Usage: Run One Scenario

`motion_planning_solver.py` is the main single-scenario driver. It supports:

- `--s1`: `neural`
- `--s2`: `mpc` or `cbf`
- `--run_type`: `s1`, `s2`, or `sofai`

Example: run the dual-process planner with System 1 and MPC System 2:

```bash
python motion_planning_solver.py \
  --problem_dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --scenario_id 1 \
  --s1 neural \
  --s2 mpc \
  --run_type sofai
```

Example: neural System 1 only:

```bash
python motion_planning_solver.py \
  --problem_dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --scenario_id 1 \
  --s1 neural \
  --s2 mpc \
  --run_type s1
```

Example: CBF System 2 only:

```bash
python motion_planning_solver.py \
  --problem_dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --scenario_id 1 \
  --s1 neural \
  --s2 cbf \
  --run_type s2
```



## Usage: Running the Benchmarks

### 1. Generate S1 assets and benchmark dictionaries

Use `script/prepare_environment_assets.py` to create:

- successful S2 trajectory libraries
- neural-policy training datasets
- trained neural S1 checkpoints
- solver-ready benchmark dictionaries
- probe set for continual learning evaluation

Supported environment families:


- `large_sparse`
- `dense_clutter`
- `serial_walls`
- `maze_branching`
- `long_slalom`
- `bugtrap`

For a single family with S2 CBF solver:

```bash
python script/prepare_environment_assets.py \
  --family dense_clutter \
  --train_n_per_family 100 \
  --eval_n_per_family 500 \
  --s2_solver cbf
```

For all the families:

```bash
export SOFAI_S1_FILTER_MODE=policy
families=(
  large_sparse
  dense_clutter
  serial_walls
  maze_branching
  long_slalom
  bugtrap
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
.venv/bin/python script/prepare_environment_assets.py \
  --families "${families[@]}" \
  --s2_solvers cbf mpc \
  --train_n_per_family 100 \
  --eval_n_per_family 500 \
  --probe_n_per_family 500 \
  --train_seed 7 \
  --eval_seed 8 \
  --probe_seed 700 \
  --train_epochs 35 \
  --train_batch 64 \
  --train_lr 0.0003 \
  --workers 16
```



### 2. Run the full benchmark set

`script/run_suite.py` supports:

- `s1_neural`
- `s2_cbf`
- `s2_mpc`
- `sofai_cbf_cl`
- `sofai_mpc_cl`
- `sofai_mpc_warm_cl`


SOFAI always runs S1 first and invokes S2 only after a failed S1 rollout. Each CL checkpoint is retrained from the frozen base model using the successful base and completed benchmark trajectories only. Probe JSONLs are never training inputs. 

For all the families:

```bash
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
    --dagger_workers 8 \
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
```

`--dagger_workers` controls scenario-level parallelism while collecting S2
DAgger recoveries. It defaults to `0`, which reuses `--workers`; set it to `1`
to collect sequentially.



### 3. Plot results

To run all archive analysis, continual-learning figures, and System 1 / System 2
split plots in one command:

```bash
python analyze_suite_results.py \
  --archive_dir output/benchmark_runs
```

All generated CSVs, tables, and figures are written below
`output/benchmark_runs/analysis/`. In particular, cross-family continual-learning
figures are placed in `analysis/figures/`, and per-family System 1/System 2
plots in `analysis/s1_s2_ratio/`.


## Upstream SOFAI Reference

This repository vendors and extends the SOFAI framework locally under `sofai/`. The upstream project is:

[ai4society/sofai_tool](https://github.com/ai4society/sofai_tool/)

This repo adds the motion-planning experiments, benchmark generators, solver wrappers, neural System 1 path, and continual-learning benchmark workflows used in the NeurIPS submission.
