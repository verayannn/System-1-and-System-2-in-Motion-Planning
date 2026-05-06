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

This repo is organized around reproducible experiment scripts rather than a polished Python package. The recommended workflow is: install the local `sofai` package in editable mode, then run the repo-level experiment drivers from the repository root.



## Repository Layout

```text
motion_planning_solver.py               Main single-scenario entry point
run_and_plot_single_benchmark.py        Run one case and save a trajectory plot
run_motion_planning_benchmarks.py       Batch runner for one or more benchmark JSON files
run_dense_clutter_benchmark_suite.py    Comparison runner for the dense-clutter environment
script/prepare_environment_assets.py    Build S1 assets and benchmark dictionaries from scratch
script/run_12_solver_suite.py           Full seven-environment / twelve-configuration suite
script/plot_12_solver_suite.py          Plot suite-level summary CSV outputs
input/                                  Benchmark dictionaries and generator
db/                                     Active/default S1 assets used by direct runners
db/by_env/                              Per-environment regenerated assets
solvers/                                S1/S2 wrappers and base implementations
sofai/                                  Local copy of the upstream SOFAI package
output/                                 Example outputs and benchmark artifacts
```

## Environment Requirements

- Python **3.10**
- A Unix-like shell (`bash` or `zsh`)
- A working `cvxpy` install for the CBF solvers
- A working `casadi` + `do-mpc` install for the MPC solvers
- A working `torch` install for neural System 1 and continual-learning runs

The commands below assume you are running from:

```bash
cd /Users/apple/Desktop/sofai
```

## Installation

### Recommended: `uv`

Create a virtual environment, activate it, and install the full repo requirements:

```bash
cd /Users/apple/Desktop/sofai

uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt
```

The root `requirements.txt` installs:

- the vendored upstream `sofai` package in editable mode
- the MPC dependencies
- the CBF dependencies
- the neural-policy dependencies

### Alternative: standard `venv`

```bash
cd /Users/apple/Desktop/sofai

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Quick Start

### 1. Run one scenario

`motion_planning_solver.py` is the main single-scenario driver. It supports:

- `--s1`: `primitives` or `neural`
- `--s2`: `mpc` or `cbf`
- `--run_type`: `s1`, `s2`, or `sofai`

Example: run the dual-process planner with primitive System 1 and MPC System 2:

```bash
python motion_planning_solver.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_id 1 \
  --s1 primitives \
  --s2 mpc \
  --run_type sofai
```

Example: neural System 1 only:

```bash
python motion_planning_solver.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_id 1 \
  --s1 neural \
  --s2 mpc \
  --run_type s1
```

Example: CBF System 2 only:

```bash
python motion_planning_solver.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_id 1 \
  --s1 primitives \
  --s2 cbf \
  --run_type s2
```

Notes:

- `--problem_dictionary` is resolved inside `input/`
- `--scenario_id` is zero-based
- `--new_run True` resets the SOFAI experience log when starting a fresh run

### 2. Run one scenario and save a trajectory plot

Use `run_and_plot_single_benchmark.py` when you want a saved JSON result plus a PNG trajectory figure for one scenario:

```bash
python run_and_plot_single_benchmark.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_ids 6 \
  --s1 primitives \
  --s2 mpc \
  --run_type sofai \
  --out_dir output/single_scenario_runs/dense_clutter_demo \
  --out_prefix dense_clutter_sc6_sofai
```

This writes:

- `output/single_scenario_runs/.../*_result.json`
- `output/single_scenario_runs/.../*_trajectory.png`

For `run_type=sofai`, the script can also preserve both S1 and S2 attempts:

```bash
python run_and_plot_single_benchmark.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_ids 6 \
  --s1 primitives \
  --s2 mpc \
  --run_type sofai \
  --run_all_attempts \
  --plot_all_attempts \
  --out_dir output/single_scenario_runs/dense_clutter_demo \
  --out_prefix dense_clutter_sc6_sofai_all
```



## Usage

### 1. Regenerate S1 assets and benchmark dictionaries from scratch


Use `script/prepare_environment_assets.py` to create:

- motion-primitive databases
- successful S2 trajectory libraries
- neural-policy training datasets
- trained neural S1 checkpoints
- solver-ready benchmark dictionaries

Example:

```bash
python script/prepare_environment_assets.py \
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
```

This writes per-environment assets under:

```text
db/by_env/<family>/
```

and benchmark dictionaries under:

```text
input/benchmarks_10k/
```

Supported environment families:

- `small_open`
- `large_sparse`
- `dense_clutter`
- `wall_gap`
- `serial_walls`
- `maze_branching`
- `bugtrap`

### 2. Run the full seven-environment / twelve-configuration suite

`script/run_12_solver_suite.py` is the main paper-scale runner. It assumes environment assets already exist under `db/by_env/<family>/`.

Example:

```bash
python script/run_12_solver_suite.py \
  --families all \
  --configs all \
  --assets_dir db/by_env \
  --benchmark_dir input/benchmarks_10k \
  --out_dir output/benchmark_runs/twelve_solver_suite \
  --scenario_ids all \
  --workers 1 \
  --case_workers 1 \
  --timeout_sec 300 \
  --retrain_every 100 \
  --train_epochs_cl 25 \
  --mplconfigdir /private/tmp/mpl
```

The twelve configuration labels are:

- `s1_primitives`
- `s1_neural`
- `s2_mpc`
- `s2_cbf`
- `sofai_mpc_primitives`
- `sofai_mpc_neural`
- `sofai_cbf_primitives`
- `sofai_cbf_neural`
- `sofai_mpc_primitives_cl`
- `sofai_mpc_neural_cl`
- `sofai_cbf_primitives_cl`
- `sofai_cbf_neural_cl`

The `_cl` variants enable online memory updates and continual-learning behavior.

### 3. Outputs

Common output locations:

- `output/single_scenario_runs/`: one-case JSON results and PNG trajectories
- `output/benchmark_runs/<run_name>/`: JSONL records and CSV summaries
- `output/benchmark_runs/twelve_solver_suite/`: cross-environment suite tables

For continual-learning runs, additional snapshots may appear under run-local `cl_assets/` directories.



## Upstream SOFAI Reference

This repository vendors and extends the SOFAI framework locally under `sofai/`. The upstream project is:

[ai4society/sofai_tool](https://github.com/ai4society/sofai_tool/)

This repo adds the motion-planning experiments, benchmark generators, solver wrappers, neural System 1 path, and continual-learning benchmark workflows used in the NeurIPS submission.
