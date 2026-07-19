# Dual Process Motion Planning

This repository contains the code used for the NeurIPS 2026 submission *Dual Process Motion Planning*. It implements a SOFAI-based dual-process motion-planning stack for 2D obstacle-avoidance tasks.

The core idea is to combine:

- **System 1**: a fast experience-driven planner
  - motion-primitives retrieval (`primitives`)
  - memory + neural policy (`neural`)
- **System 2**: a slower but more reliable online solver
  - model predictive control (`mpc`)
  - control barrier functions (`cbf`)
- **Metacognitive arbitration**: the SOFAI controller decides whether to accept the System 1 proposal or fall back to System 2
- **Continual-learning variants**: successful System 2 trajectories can be written back into memory and used to improve later runs


## Installation

The recommended setup uses `uv` plus the repo-local `safe_control/acados` tree. This works on macOS and Linux as long as the machine has a C/C++ compiler, `cmake`, `make`, and `git`.

System 2 MPC is acados-only in this repo. The Python dependencies can be installed by `uv` or `pip`, but the native acados shared libraries must still be built with CMake. System 1 and System 2 CBF do not require acados.

### 1. Clone the repository

```bash
git clone <repository-url>
cd System-1-and-System-2-in-Motion-Planning
git submodule update --recursive --init
```

### 2. Install Python dependencies with `uv`

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra acados-template
```

The root `pyproject.toml` installs the repo itself in editable mode and exposes the local source packages directly:

- `sofai/`
- `safe_control/`

Do not install `./sofai` and `./safe_control` as separate editable packages for the normal experiment environment; that reintroduces nested build-system resolution.

It also keeps the numerical stack on `numpy>=1.26.4,<2.0`, which avoids the previous mismatch between vendored SOFAI and `safe_control`.

### 3. Build and register acados

```bash
python script/setup_acados.py --jobs 4
source .env.acados
```

The setup script:

- builds `safe_control/acados` with CMake
- checks that `libacados`, `libblasfeo`, and `libhpipm` exist
- installs `acados_template` into the active Python environment
- writes `.env.acados` with the correct library path variables for macOS or Linux

On Linux servers, install native build tools first if they are missing:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake python3-dev
```

On macOS, install build tools if they are missing:

```bash
xcode-select --install
brew install cmake
```

If you want to use a separate acados checkout instead of the repo-local one:

```bash
python script/setup_acados.py --acados_root /path/to/acados --skip_build
source .env.acados
```

### 4. Verify the install

```bash
python - <<'PY'
import sofai_tool
import safe_control
from solvers._s2_common import detect_acados_root

root = detect_acados_root()
print("SOFAI import: ok")
print("safe_control import: ok")
print("acados root:", root)
assert root is not None, "acados shared libraries were not found"
PY
```

### Pip or Conda fallback

If `uv` is not available, use a Python 3.10 environment and install the compatibility requirements:

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
sofai/
├── README.md                           # Project documentation
├── requirements.txt                    # Top-level experiment environment
├── motion_planning_solver.py           # Main single-scenario runner
├── visualize_mp.py                     # Single-case runner + plot output
├── run_motion_planning_benchmarks.py   # Batch benchmark runner
├── plot_suite_results.py               # Suite success-rate / runtime plots
├── plot_suite_s1_s2_ratio.py           # S1 vs S2 split plot for CL runs
│
├── sofai/                              # Vendored upstream SOFAI package
├── safe_control/                       # Vendored safe_control + acados
│
├── solvers/                            # Motion-planning solver implementations
│   ├── S1_motion_primitives.py         # Primitive-based System 1
│   ├── S1_memory_neural.py             # Memory + neural System 1
│   ├── S2_mpc.py                       # MPC System 2 wrapper
│   ├── S2_cbf.py                       # CBF System 2 wrapper
│   ├── base/                           # Core planning / training logic folder
│   └── combinations/                   # Combined SOFAI runner variants
│       ├── mpc_solver.py
│       ├── mpc_solver_new_S1.py
│       ├── cbf_solver.py
│       └── cbf_solver_new_S1.py
│
├── input/                              # Benchmark dictionaries and metadata
│   ├── input_handler.py
│   ├── generate_benchmark_dictionaries.py
│   ├── generate_nl_dict.py
│   ├── meta/
│   │   ├── context.txt
│   │   └── thresholds.txt
│   └── nl/                             # Nonlinear benchmark dictionaries
│
├── db/                                 # Active/default S1 assets
│   └── by_env/                         # Per-environment assets
│
├── script/                             # Experiment orchestration scripts
│   ├── prepare_environment_assets.py
│   ├── run_suite.py
│   └── train_s1_nonlinear.py
│
└── output/                             # Generated results, plots, summaries
    ├── single_scenario_runs/
    └── benchmark_runs

```


## Usage: Run One Scenario

`motion_planning_solver.py` is the main single-scenario driver. It supports:

- `--s1`: `primitives` or `neural`
- `--s2`: `mpc` or `cbf`
- `--run_type`: `s1`, `s2`, or `sofai`

Example: run the dual-process planner with primitive System 1 and MPC System 2:

```bash
python motion_planning_solver.py \
  --problem_dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --scenario_id 1 \
  --s1 primitives \
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
  --s1 primitives \
  --s2 cbf \
  --run_type s2
```

Notes:

- `--problem_dictionary` is resolved inside `input/`
- `--scenario_id` is zero-based
- `--new_run True` resets the SOFAI experience log when starting a fresh run


## Usage: Running the Benchmarks

### 1. Generate S1 assets and benchmark dictionaries

Use `script/prepare_environment_assets.py` to create:

- motion-primitive databases
- successful S2 trajectory libraries
- neural-policy training datasets
- trained neural S1 checkpoints
- solver-ready benchmark dictionaries

Supported environment families:

- `small_open`
- `large_sparse`
- `dense_clutter`
- `wall_gap`
- `serial_walls`
- `maze_branching`
- `bugtrap`

For a single family:

```bash
python script/prepare_environment_assets.py \
  --family dense_clutter \
  --train_n_per_family 100 \
  --eval_n_per_family 500 \
  --s2_solver cbf
```

For all the families:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl \
python script/prepare_environment_assets.py \
  --families small_open large_sparse dense_clutter wall_gap serial_walls maze_branching bugtrap \
  --train_n_per_family 500 \
  --eval_n_per_family 2500 \
  --s2_solver cbf
```

The script writes benchmark dictionaries to `input/nl/` and assets to `db/by_env/<family>_nl/`.

### 2. Run the full benchmark set

`script/run_suite.py` supports:

- `s1_neural`
- `s2_cbf`
- `s2_mpc`
- `sofai_cbf_cl`
- `sofai_mpc_cl`

The SOFAI continual-learning modes run in blocks, retrain the neural System 1 after each block, and carry the updated checkpoint into the next block.

For a single family:

```bash
python script/run_suite.py \
  --dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --bootstrap_results_dir output/bootstrap_dense_clutter_nl \
  --assets_dir db/by_env/dense_clutter_nl \
  --out_dir output/benchmark_runs/nl_dense_clutter_suite \
  --scenario_ids 0-2499 \
  --block_size 500 \
  --workers 6 \
  --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl
```

For all the families:

```bash
for family in dense_clutter small_open large_sparse wall_gap serial_walls maze_branching bugtrap; do
 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl" \
 python script/run_suite.py \
   --dictionary "input/nl/benchmark_dualmp_nl_${family}_eval_${family}.json" \
   --bootstrap_results_dir "output/bootstrap_${family}_nl" \
   --assets_dir "db/by_env/${family}_nl" \
   --out_dir "output/benchmark_runs/nl_${family}_suite" \
   --scenario_ids 0-2499 \
   --block_size 500 \
   --workers 6 \
   --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl
done
```



### 3. Plot results

Example:

```bash
python plot_suite_results.py \
  --suite_dir output/benchmark_runs/nl_dense_clutter_suite \
  --out output/benchmark_runs/nl_dense_clutter_suite/summary.png
```

To plot the System 1 / System 2 split by block:

```bash
python plot_suite_s1_s2_ratio.py \
  --suite_dir output/benchmark_runs/nl_dense_clutter_suite \
  --config sofai_cbf_cl \
  --out output/benchmark_runs/nl_dense_clutter_suite/s1_s2_ratio.png
```


## Upstream SOFAI Reference

This repository vendors and extends the SOFAI framework locally under `sofai/`. The upstream project is:

[ai4society/sofai_tool](https://github.com/ai4society/sofai_tool/)

This repo adds the motion-planning experiments, benchmark generators, solver wrappers, neural System 1 path, and continual-learning benchmark workflows used in the NeurIPS submission.
