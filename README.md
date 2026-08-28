# Dual Process Motion Planning

This repository implements a SOFAI-based dual-process motion-planning stack for 2D obstacle-avoidance tasks under nonlinear dynamics.

The core idea is to combine:

- **System 1**: a fast experience-driven planner
  - neural policy (`neural`)
- **System 2**: a slower but more reliable online solver
  - model predictive control (`mpc`)
  - control barrier functions (`cbf`)
- **Metacognitive arbitration**: the SOFAI controller decides whether to accept the System 1 proposal or fall back to System 2
- **Continual-learning variants**: successful System 2 trajectories are aggregated for retraining System 1 to improve later runs


## Installation

Follow the steps below to install the dual-process motion-planning stack:


### 1. Install prerequisites:

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

### 2. Clone the Repository:

```bash
git clone https://github.com/verayannn/System-1-and-System-2-in-Motion-Planning.git
cd System-1-and-System-2-in-Motion-Planning
```

### 3. Set up the environment:

```bash
./setup.sh
```

### 4. Use the environment

```bash
source .venv/bin/activate
```

### 5. Verify the installation

`setup.sh` runs this check automatically and prints `[ok] imports: ...`. To repeat it later:

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


Without `uv`, use Python 3.10 or 3.11:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python script/setup_acados.py --jobs 4
```

## Directory Structure

```text
System-1-and-System-2-in-Motion-Planning/
├── README.md                           # Project documentation and usage
├── setup.sh                            # One-command environment and acados setup
├── motion_planning_solver.py           # Single-scenario planner entry point
├── run_motion_planning_benchmarks.py   # Run motion-planning benchmarks and save JSONL/CSV metrics
├── visualize_mp.py                     # Single-scenario visualisation
├── analyze_suite_results.py            # Unified benchmark analysis pipeline
├── requirements.txt / pyproject.toml   # Python environment and package metadata
├── input/                              # Benchmark dictionaries and generation utilities
├── db/by_env/                          # Per-environment neural S1 datasets and checkpoints
├── solvers/                            # S1 policies, MPC/CBF S2 solvers, quality metrics
├── script/                             # Experiment orchestration utilities
│   ├── setup_acados.py                 # Build and register the acados backend
│   ├── prepare_environment_assets.py   # Create per-environment S1 assets
│   ├── run_suite.py                    # Run benchmark suites and continual learning
│   └── train_s1_nonlinear.py           # Train or fine-tune the neural S1 policy
├── sofai/                              # Vendored SOFAI framework
└── safe_control/                       # Vendored CBF and acados dependencies
```


## Usage: Running the Benchmarks

Supported environment families:

- `large_sparse`
- `dense_clutter`
- `serial_walls`
- `maze_branching`
- `long_slalom`
- `bugtrap`

Supported benchmark configurations:

- `s1_neural`
- `s2_cbf`
- `s2_mpc`
- `sofai_cbf_cl`
- `sofai_mpc_cl`
- `sofai_mpc_warm_cl`

In standard SOFAI mode, System 1 is attempted first and System 2 is invoked only when the System 1 rollout fails. Continual-learning checkpoints are updated after each benchmark block. Probe JSONLs are never used as the training inputs.

### Smoke Test Run


#### 1. Generate S1 assets and benchmark dictionaries (estimated runtime: ~1 minute)

Use `script/prepare_environment_assets.py` to create:

- successful S2 trajectory libraries
- neural-policy training datasets
- trained neural S1 checkpoints
- solver-ready benchmark dictionaries
- probe set for continual learning evaluation

```bash
export SOFAI_S1_FILTER_MODE=policy
families=(
  dense_clutter
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
.venv/bin/python script/prepare_environment_assets.py \
  --families "${families[@]}" \
  --s2_solvers cbf mpc \
  --train_n_per_family 20 \
  --eval_n_per_family 10 \
  --probe_n_per_family 10 \
  --train_seed 7 \
  --eval_seed 8 \
  --probe_seed 700 \
  --train_epochs 10 \
  --train_batch 32 \
  --train_lr 0.0003 \
  --workers 4
```


#### 2. Run the smoke test benchmark sets (estimated runtime: ~3 minute)

Run all benchmark configurations on 10 evaluation instances and 10 probe instances from the Dense
Clutter family. This command uses the generated instances in `input/` and the pretrained System 1 checkpoints in `db/` from above.


```bash
families=(
  dense_clutter
)
for family in "${families[@]}"; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python script/run_suite.py \
    --dictionary "input/nl/benchmark_dualmp_nl_${family}_eval_${family}.json" \
    --bootstrap_results_dir "output/bootstrap_${family}_nl" \
    --assets_dir "db/by_env/${family}_nl" \
    --out_dir "output/benchmark_smoke_test_runs/nl_${family}_suite" \
    --scenario_ids 0-9 \
    --block_size 10 \
    --workers 4 \
    --timeout_sec 60 \
    --block_order shuffled \
    --block_seed 42 \
    --cl_bootstrap_solver auto \
    --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl sofai_mpc_warm_cl \
    --cl_train_mode replay_dagger \
    --replay_fraction 0.60 \
    --dagger_states_per_scenario 4 \
    --dagger_workers 4 \
    --train_source s2 \
    --bootstrap_success_weight 1.0 \
    --dagger_success_weight 1.0 \
    --train_epochs 6 \
    --train_batch 32 \
    --train_lr 0.0001 \
    --train_device cpu \
    --probe_dictionary "input/nl/benchmark_dualmp_nl_${family}_probe_${family}.json" \
    --probe_scenario_ids 0-9
done
```


> [!NOTE]
> Delete the `input/nl/`, `db/`, and `output/` folders after the smoke test completes.


### Full-Scale Benchmark Run

#### 1. Generate S1 assets and benchmark dictionaries


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


#### 2. Run the full benchmark set

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
    --timeout_sec 60 \
    --block_order shuffled \
    --block_seed 42 \
    --cl_bootstrap_solver auto \
    --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl sofai_mpc_warm_cl \
    --cl_train_mode replay_dagger \
    --replay_fraction 0.60 \
    --dagger_states_per_scenario 4 \
    --dagger_workers 16 \
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

#### 3. Plot results

Plot results in one command:

```bash
python analyze_suite_results.py --archive_dir output/benchmark_runs \
      --families dense_clutter large_sparse maze_branching serial_walls long_slalom bugtrap \
      --configs s1_neural s2_cbf s2_mpc sofai_cbf_cl sofai_mpc_cl sofai_mpc_warm_cl
```

All generated CSVs, tables, and figures are written below `output/benchmark_runs/analysis/`.


## Upstream SOFAI Reference

This repository vendors and extends the SOFAI framework locally under `sofai/`. The upstream project is:

[ai4society/sofai_tool](https://github.com/ai4society/sofai_tool/)

This repo adds the motion-planning experiments, benchmark generators, solver wrappers, neural System 1 path, and continual-learning benchmark workflows used in the AAAI submission.
