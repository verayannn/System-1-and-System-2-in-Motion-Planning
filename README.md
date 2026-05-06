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



## Installation

Follow the steps below to install the project locally.

### Option A: `uv` (recommended)

1. Clone the repository or download the source archive, then move into the project root:

```bash
git clone <repository-url>
cd <repo-root>
```

2. Create and activate a Python 3.10 virtual environment:

```bash
uv venv --python 3.10
source .venv/bin/activate
```

3. Install the full experiment stack:

```bash
uv pip install -r requirements.txt
```

4. Verify that the vendored SOFAI package is importable:

```bash
python -c "import sofai_tool; print('SOFAI installation verified.')"
```

The root `requirements.txt` installs:

- the vendored upstream `sofai` package in editable mode
- the MPC dependencies
- the CBF dependencies
- the neural-policy dependencies

### Option B: Conda

1. Clone the repository or download the source archive, then move into the project root:

```bash
git clone <repository-url>
cd <repo-root>
```

2. Create and activate a Conda environment:

```bash
conda create --name dualmp_env python=3.10 -y
conda activate dualmp_env
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Verify the installation:

```bash
python -c "import sofai_tool; print('SOFAI installation verified.')"
```

### Option C: standard `venv`

1. Clone the repository or download the source archive, then move into the project root:

```bash
git clone <repository-url>
cd <repo-root>
```

2. Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Verify the installation:

```bash
python -c "import sofai_tool; print('SOFAI installation verified.')"
```

### Development-mode install

The default repo-level install already places the vendored `sofai/` package in editable mode through `requirements.txt`, which is the right default for experiment work in this repository.

If you only want to install the vendored SOFAI package itself in editable mode, use:

```bash
pip install -e ./sofai
```

## Directory Structure

```text
sofai/
├── README.md                           # Project documentation
├── requirements.txt                    # Top-level experiment environment
├── motion_planning_solver.py           # Main single-scenario runner
├── run_motion_planning_benchmarks.py   # Batch benchmark runner
│
├── sofai/                              # Vendored upstream SOFAI package
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
│   ├── meta/
│   │   ├── context.txt
│   │   └── thresholds.txt
│   └── benchmarks_10k/                 # Large generated benchmark sets
│
├── db/                                 # Active/default S1 assets
│   └── by_env/                         # Per-environment assets
│
├── script/                             # Experiment orchestration scripts
│   ├── prepare_environment_assets.py
│   ├── run_12_solver_suite.py
│   └── plot_12_solver_suite.py
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


## Usage: Running the Benchmarks

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


## Upstream SOFAI Reference

This repository vendors and extends the SOFAI framework locally under `sofai/`. The upstream project is:

[ai4society/sofai_tool](https://github.com/ai4society/sofai_tool/)

This repo adds the motion-planning experiments, benchmark generators, solver wrappers, neural System 1 path, and continual-learning benchmark workflows used in the NeurIPS submission.
