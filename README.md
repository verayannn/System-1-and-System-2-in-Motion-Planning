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

Follow the steps below to install the project locally on either macOS or Linux.

### 0. Install `acados`

System 2 MPC uses a native `acados` backend. Build or install `acados` first, then point the environment to the installed prefix.

If you are using a separate `acados` install:

```bash
export ACADOS_SOURCE_DIR=/path/to/built/acados
python -m pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
```

If you want to use the vendored `acados` tree shipped in this repo, build it first and then point `ACADOS_SOURCE_DIR` at that built prefix.

### 1. Clone the repository

```bash
git clone <repository-url>
cd System-1-and-System-2-in-Motion-Planning
```

### 2. Create a Python 3.10 environment

Pick one of the following:

```bash
# uv
uv venv --python 3.10
source .venv/bin/activate

# conda
conda create --name dualmp_env python=3.10 -y
conda activate dualmp_env

# standard venv
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer `uv`, use:

```bash
uv pip install -r requirements.txt
```

### 4. Verify the install

```bash
python -c "import sofai_tool, safe_control; print('SOFAI installation verified.')"
```

The root `requirements.txt` installs both local packages in editable mode:

- `-e ./sofai`
- `-e ./safe_control`

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

Example:

```bash
python script/prepare_environment_assets.py \
  --families small_open large_sparse dense_clutter wall_gap serial_walls maze_branching bugtrap \
  --train_n_per_family 100 \
  --eval_n_per_family 500 \
  --s2_solver cbf
```

For a single family:

```bash
python script/prepare_environment_assets.py \
  --family dense_clutter \
  --train_n_per_family 100 \
  --eval_n_per_family 500 \
  --s2_solver cbf
```

Supported environment families:

- `small_open`
- `large_sparse`
- `dense_clutter`
- `wall_gap`
- `serial_walls`
- `maze_branching`
- `bugtrap`

The script writes benchmark dictionaries to `input/nl/` and assets to `db/by_env/<family>_nl/`.

Example:

```bash
python script/run_suite.py \
  --dictionary input/nl/benchmark_dualmp_nl_dense_clutter_eval_dense_clutter.json \
  --bootstrap_results_dir output/bootstrap_dense_clutter_nl \
  --assets_dir db/by_env/dense_clutter_nl \
  --out_dir output/benchmark_runs/nl_dense_clutter_suite \
  --scenario_ids 0-499 \
  --block_size 100 \
  --workers 2 \
  --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl
```

`script/run_suite.py` supports:

- `s1_neural`
- `s2_cbf`
- `s2_mpc`
- `sofai_cbf_cl`
- `sofai_mpc_cl`

The SOFAI continual-learning modes run in blocks, retrain the neural System 1 after each block, and carry the updated checkpoint into the next block.

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
