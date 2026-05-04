# System 1 and System 2 in Motion Planning with SOFAI

This repository integrates motion-planning solvers into the [SOFAI Tool](https://github.com/ai4society/sofai_tool/) architecture. The main working instance is:

```text
sofai_tool/sofai_instances/mpc-sofai/
```

The project studies a fast System 1 planner and a slower deliberative System 2 planner for 2D maze navigation. SOFAI's metacognition module decides whether to accept the System 1 solution or fall back to System 2.

The current maze instance supports:

- System 1 motion primitives
- System 1 memory + neural policy
- System 2 MPC
- System 2 CBF/SFCBF
- SOFAI arbitration between S1 and S2
- Continual-learning experiments for the memory-neural S1
- Plotting utilities for solver outputs

## Reference

This repository builds on the SOFAI Tool framework:

- SOFAI Tool: <https://github.com/ai4society/sofai_tool/>

SOFAI provides the generic `System1Solver`, `System2Solver`, and metacognition interfaces. This repository adds a motion-planning instance under `sofai_tool/sofai_instances/mpc-sofai`.

## Repository Layout

```text
System-1-and-System-2-in-Motion-Planning-sofai-integration/
├── README.md
├── circle/                         # Older circle-obstacle experiments
├── maze/                           # Older standalone maze experiments
├── generate_benchmark_dictionaries.py
└── sofai_tool/
    ├── sofai_tool/                 # Core SOFAI package
    │   ├── metacognition/
    │   ├── solvers/
    │   └── utils/
    ├── requirements.txt
    ├── setup.py
    ├── pyproject.toml
    └── sofai_instances/
        └── mpc-sofai/              # Main motion-planning SOFAI instance
            ├── input/
            │   ├── benchmark_scenarios_maze_1199_block200.json
            │   ├── input_handler.py
            │   └── meta/
            │       ├── context.txt
            │       └── thresholds.txt
            ├── Solvers/
            │   ├── S1_motion_primitives.py
            │   ├── S1_memory_neural.py
            │   ├── S2_mpc.py
            │   ├── S2_cbf.py
            │   ├── S1_database_maze.json
            │   ├── s1_sfcbf_success_trajs.npz
            │   └── Base/
            │       ├── S1_usage_maze.py
            │       ├── S1_NN_usage_maze.py
            │       ├── S1_S2_continual_maze.py
            │       ├── S2_mpc_maze.py
            │       ├── S2_cbf_maze.py
            │       ├── train_nn_policy.py
            │       └── make_diverse_training_data_maze.py
            ├── mpc_solver.py
            ├── mpc_solver_new_S1.py
            ├── cbf_solver.py
            ├── cbf_solver_new_S1.py
            ├── run_s1_memory_neural_continual_experiment.py
            ├── plot_mpc_result.py
            ├── plot_s1_memory_neural_continual_results.py
            ├── db/
            └── output/
```

## Main Files

### SOFAI entry points

| File | System 1 | System 2 | Purpose |
|---|---|---|---|
| `mpc_solver.py` | Motion primitives | MPC | Original SOFAI maze solver |
| `mpc_solver_new_S1.py` | Memory + neural S1 | MPC | New S1 with MPC fallback |
| `cbf_solver.py` | Motion primitives | CBF/SFCBF | Original S1 with CBF fallback |
| `cbf_solver_new_S1.py` | Memory + neural S1 | CBF/SFCBF | New S1 with CBF fallback |

### Solver wrappers

| File | Main function | Description |
|---|---|---|
| `Solvers/S1_motion_primitives.py` | `solveMotionPrimitives(scenario)` | Retrieves a stored motion primitive from the S1 database |
| `Solvers/S1_memory_neural.py` | `solveMemoryNeural(scenario, return_info=True)` | Uses episodic memory first, then neural rollout |
| `Solvers/S2_mpc.py` | `solve_MPC(scenario)` | Wraps the MPC maze solver |
| `Solvers/S2_cbf.py` | `solve_CBF(scenario)` | Wraps the SFCBF maze solver |
| `input/input_handler.py` | `load_scenarios(path)` | Loads scenario JSON into `MazeProblem` objects |

### Base implementation files

| File | Description |
|---|---|
| `Solvers/Base/S2_mpc_maze.py` | Standalone MPC implementation and CLI |
| `Solvers/Base/S2_cbf_maze.py` | Standalone CBF/SFCBF implementation and CLI |
| `Solvers/Base/S1_usage_maze.py` | Motion-primitive S1 retrieval utilities |
| `Solvers/Base/S1_NN_usage_maze.py` | Neural S1 rollout utilities |
| `Solvers/Base/S1_S2_continual_maze.py` | Memory-neural S1, continual learning, retraining helpers |
| `Solvers/Base/train_nn_policy.py` | Neural policy training script |
| `Solvers/Base/make_diverse_training_data_maze.py` | Dataset construction utilities |

## Installation From Scratch

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd System-1-and-System-2-in-Motion-Planning-sofai-integration
```

### 2. Create a Python environment

Python 3.10 is recommended.

```bash
conda create -n s12_env python=3.10 -y
conda activate s12_env
```

### 3. Install the local SOFAI package

```bash
python -m pip install -U pip
python -m pip install -e ./sofai_tool
```

### 4. Install motion-planning dependencies

The core SOFAI package has its own requirements, but the motion-planning instance also uses MPC, CasADi, CBF, and neural-network dependencies.

```bash
python -m pip install -r sofai_tool/requirements.txt
python -m pip install numpy matplotlib casadi do-mpc cvxpy torch
```

If you do not use the memory-neural System 1, `torch` is not needed.

### 5. Verify the installation

```bash
python - <<'PY'
import sofai_tool
import numpy
import matplotlib
import casadi
import do_mpc
import cvxpy
print("installation ok")
PY
```

## Working Directory

Most commands should be run from the SOFAI maze instance directory:

```bash
cd sofai_tool/sofai_instances/mpc-sofai
```

If you installed `sofai_tool` with `pip install -e ./sofai_tool`, the solver scripts can import the SOFAI package directly.

If you did not install the package, run commands with this `PYTHONPATH` prefix from inside `mpc-sofai`:

```bash
PYTHONPATH="$(pwd):$(cd ../.. && pwd)"
```

For example:

```bash
PYTHONPATH="$(pwd):$(cd ../.. && pwd)" python mpc_solver.py --problem_dictionary benchmark_scenarios_maze_1199_block200.json --scenario_id 0
```

## Scenario Format

The benchmark scenarios live in:

```text
sofai_tool/sofai_instances/mpc-sofai/input/benchmark_scenarios_maze_1199_block200.json
```

Each scenario is a JSON object with fields like:

```json
{
  "scenario_id": 0,
  "A_query": [[0.0, 1.0], [-1.0, -0.2]],
  "B_query": [[0.0], [1.0]],
  "rectangles": [[-2.0, -1.0, 2.0, 1.0]],
  "bounds": [-10.0, -10.0, 10.0, 10.0],
  "start": [-8.0, -8.0],
  "goal": [8.0, 8.0],
  "u_max": 3.0,
  "goal_tol": 0.5
}
```

`scenario_id` in the command line is the zero-based index into the JSON list.

## Quick Start

Go to the maze SOFAI instance:

```bash
cd sofai_tool/sofai_instances/mpc-sofai
mkdir -p output
```

### Run original SOFAI with motion-primitives S1 and MPC S2

```bash
MPLCONFIGDIR=/tmp/mpl python mpc_solver.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 0
```

### Run new memory-neural S1 with MPC S2

```bash
MPLCONFIGDIR=/tmp/mpl python mpc_solver_new_S1.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 0
```

### Run original SOFAI with motion-primitives S1 and CBF S2

```bash
MPLCONFIGDIR=/tmp/mpl python cbf_solver.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 0
```

### Run new memory-neural S1 with CBF S2

```bash
MPLCONFIGDIR=/tmp/mpl python cbf_solver_new_S1.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 0
```

## Plot a SOFAI Result

After running a solver, plot the saved SOFAI trajectory:

```bash
MPLCONFIGDIR=/tmp/mpl python plot_mpc_result.py \
  --experience db/plan_experience.json \
  --input_dir input \
  --problem_name benchmark_scenarios_maze_1199_block200_sc_0 \
  --out output/scenario_0_result.png
```

The output image will be saved to:

```text
output/scenario_0_result.png
```

If you want to plot the latest recorded case, omit `--problem_name`:

```bash
MPLCONFIGDIR=/tmp/mpl python plot_mpc_result.py
```

## Run Pure System 2 Solvers

These commands bypass SOFAI metacognition and run System 2 directly on benchmark scenarios.

### Run MPC on the full benchmark file

```bash
MPLCONFIGDIR=/tmp/mpl python Solvers/Base/S2_mpc_maze.py \
  --scenarios input/benchmark_scenarios_maze_1199_block200.json \
  --out output/s2_mpc_results_1199.json \
  --dt 0.05 \
  --n_steps 800 \
  --n_horizon 20 \
  --wall_margin 0.2 \
  --smooth_kappa 20.0 \
  --goal_tol 0.5
```

### Run CBF/SFCBF on the full benchmark file

```bash
MPLCONFIGDIR=/tmp/mpl python Solvers/Base/S2_cbf_maze.py \
  --scenarios input/benchmark_scenarios_maze_1199_block200.json \
  --out output/s2_cbf_results_1199.json \
  --dt 0.05 \
  --n_steps 800 \
  --u_max 3.0 \
  --margin 0.35 \
  --gamma 2.0 \
  --goal_tol 0.6 \
  --collision_margin 0.05
```

### Run `Solvers/S2_mpc.py` on one scenario

`Solvers/S2_mpc.py` is a wrapper module, not a standalone CLI script. Use a Python one-liner:

```bash
MPLCONFIGDIR=/tmp/mpl python -c 'import numpy as np; from input.input_handler import load_scenarios; from Solvers.S2_mpc import solve_MPC; from Solvers.Base.S2_mpc_maze import collision_free_rectangles, goal_reached; i=0; s=load_scenarios("input/benchmark_scenarios_maze_1199_block200.json")[i]; states=solve_MPC(s); print("failed" if states is None else f"scenario_id={s.scenario_id}, states_shape={np.asarray(states).shape}, collision_free={collision_free_rectangles(np.asarray(states), s.rects)}, goal_reached={goal_reached(np.asarray(states), s.goal, s.goal_tol)}, final_state={np.asarray(states)[-1].tolist()}")'
```

Change `i=0` to run another scenario.

## Memory-Neural System 1 Assets

`Solvers/S1_memory_neural.py` can use assets stored directly in `Solvers/`, or paths provided with environment variables.

Expected assets include:

```text
s1_policy_full_retrain_latest.pth
s1_policy_control_cnn_diverse_5k.pth
nn_dataset_maze_diverse_5k.npz
s1_sfcbf_success_trajs_diverse_5k.npz
benchmark_scenarios_maze_diverse_5k.json
```

If the files are not committed because they are large, set explicit paths:

```bash
export SOFAI_NEW_S1_MODEL="/path/to/s1_policy_control_cnn_diverse_5k.pth"
export SOFAI_NEW_S1_BASE_DATASET="/path/to/nn_dataset_maze_diverse_5k.npz"
export SOFAI_NEW_S1_BASE_MEMORY_TRAJ="/path/to/s1_sfcbf_success_trajs_diverse_5k.npz"
export SOFAI_NEW_S1_BASE_MEMORY_SCENARIOS="/path/to/benchmark_scenarios_maze_diverse_5k.json"
```

Optional runtime controls:

```bash
export SOFAI_NEW_S1_DEVICE="cpu"
export SOFAI_NEW_S1_ENABLE_MEMORY="true"
export SOFAI_NEW_S1_MEMORY_BEFORE_NN="true"
export SOFAI_NEW_S1_STEPS="120"
export SOFAI_NEW_S1_CONFIDENCE_THRESHOLD="0.35"
export SOFAI_NEW_S1_MEMORY_SCORE_THRESHOLD="0.65"
export SOFAI_NEW_S1_MEMORY_MAP_THRESHOLD="0.45"
export SOFAI_NEW_S1_MEMORY_DYN_SIGMA="0.45"
```

## Motion-Primitive System 1 Assets

`Solvers/S1_motion_primitives.py` expects these files:

```text
Solvers/S1_database_maze.json
Solvers/s1_sfcbf_success_trajs.npz
```

If you see this error:

```text
[System1 ERROR] [Errno 2] No such file or directory: 'Solvers/S1_database_maze.json'
```

then those files are missing from `mpc-sofai/Solvers/`. Copy or regenerate the motion-primitive database before running `mpc_solver.py` or `cbf_solver.py`.

## Continual-Learning Experiment

The continual-learning experiment compares:

- static memory-neural S1: no new S2 successes are stored and no retraining happens
- continual memory-neural S1: successful S2 fallback trajectories are added to memory, and the neural model is periodically retrained

Run both modes on the 1199-scenario benchmark:

```bash
MPLCONFIGDIR=/tmp/mpl python run_s1_memory_neural_continual_experiment.py \
  --scenarios input/benchmark_scenarios_maze_1199_block200.json \
  --workdir output/s1_memory_neural_continual_1199 \
  --mode both \
  --block_size 200 \
  --update_every_blocks 1 \
  --min_s2_full_records_for_update 1
```

Useful smaller dry run:

```bash
MPLCONFIGDIR=/tmp/mpl python run_s1_memory_neural_continual_experiment.py \
  --scenarios input/benchmark_scenarios_maze_1199_block200.json \
  --workdir output/s1_memory_neural_debug \
  --mode both \
  --max_scenarios 20 \
  --block_size 10 \
  --dry_run
```

Plot the experiment results:

```bash
MPLCONFIGDIR=/tmp/mpl python plot_s1_memory_neural_continual_results.py \
  --workdir output/s1_memory_neural_continual_1199 \
  --outdir output/s1_memory_neural_continual_1199/plots
```

Important output files:

```text
output/s1_memory_neural_continual_1199/static/results_all.json
output/s1_memory_neural_continual_1199/static/learning_curve.csv
output/s1_memory_neural_continual_1199/continual/results_all.json
output/s1_memory_neural_continual_1199/continual/learning_curve.csv
output/s1_memory_neural_continual_1199/continual/episodic_s2_memory.json
```

## CBF Solver Environment Variables

`Solvers/S2_cbf.py` exposes CBF parameters through environment variables:

```bash
export SOFAI_CBF_DT="0.05"
export SOFAI_CBF_STEPS="800"
export SOFAI_CBF_MARGIN="0.35"
export SOFAI_CBF_GAMMA="2.0"
export SOFAI_CBF_GOAL_TOL="0.6"
export SOFAI_CBF_COLLISION_MARGIN="0.05"
```

Example:

```bash
SOFAI_CBF_GOAL_TOL=0.6 SOFAI_CBF_STEPS=1000 python cbf_solver_new_S1.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 0
```

## Python API Examples

### Load scenarios

```python
from input.input_handler import load_scenarios

scenarios = load_scenarios("input/benchmark_scenarios_maze_1199_block200.json")
scenario = scenarios[0]
```

### Run motion-primitives S1

```python
from Solvers.S1_motion_primitives import solveMotionPrimitives

states, confidence = solveMotionPrimitives(scenario)
```

### Run memory-neural S1

```python
from Solvers.S1_memory_neural import solveMemoryNeural

states, confidence, info = solveMemoryNeural(scenario, return_info=True)
print(confidence)
print(info["source"])
```

`info["source"]` is usually one of:

```text
S1_memory
S1_neural
none
```

### Run MPC System 2

```python
from Solvers.S2_mpc import solve_MPC

states = solve_MPC(scenario)
```

### Run CBF System 2

```python
from Solvers.S2_cbf import solve_CBF

states = solve_CBF(scenario)
```

### Check correctness

```python
import numpy as np
from Solvers.Base.S2_mpc_maze import collision_free_rectangles, goal_reached

states = np.asarray(states)
collision_free = collision_free_rectangles(states, scenario.rects)
reached = goal_reached(states, scenario.goal, scenario.goal_tol)
success = collision_free and reached
```

## How SOFAI Is Used Here

Each SOFAI entry script defines two solver classes:

```python
class CustomSystem1Solver(sofai1.System1Solver):
    def solve(self, problem_id):
        ...

    def calculate_correctness(self, problem_id):
        ...
```

```python
class CustomSystem2Solver(sofai2.System2Solver):
    def solve(self, problem_id, time_limit):
        ...

    def estimate_difficulty(self, problem_id):
        ...

    def calculate_correctness(self, problem_id):
        ...
```

Then the script calls:

```python
meta.metacognition(
    problem_name,
    system1_solver,
    system2_solver,
    context_file,
    thresholds_file,
    experience_file,
    new_run=False,
    run_type="sofai",
)
```

The `problem_name` convention is:

```text
<scenario_json_stem>_sc_<scenario_index>
```

Example:

```text
benchmark_scenarios_maze_1199_block200_sc_0
```

## Outputs

SOFAI runs write solver experience to the SOFAI database file, usually:

```text
db/plan_experience.json
```

Plot files are written to:

```text
output/
```

Standalone batch solvers write JSON files wherever `--out` points.

## Troubleshooting

### `ModuleNotFoundError: No module named 'sofai_tool'`

Install the local package:

```bash
cd System-1-and-System-2-in-Motion-Planning-sofai-integration
python -m pip install -e ./sofai_tool
```

Or run from `mpc-sofai` with:

```bash
PYTHONPATH="$(pwd):$(cd ../.. && pwd)" python mpc_solver.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 0
```

### `No such file or directory: 'Solvers/S1_database_maze.json'`

The motion-primitive S1 database is missing. Make sure these files exist:

```text
Solvers/S1_database_maze.json
Solvers/s1_sfcbf_success_trajs.npz
```

### Matplotlib cache permission errors

Set a writable Matplotlib cache directory:

```bash
MPLCONFIGDIR=/tmp/mpl python plot_mpc_result.py
```

### Neural S1 cannot find `.pth` or `.npz` files

Set the explicit asset paths:

```bash
export SOFAI_NEW_S1_MODEL="/path/to/model.pth"
export SOFAI_NEW_S1_BASE_DATASET="/path/to/nn_dataset_maze_diverse_5k.npz"
export SOFAI_NEW_S1_BASE_MEMORY_TRAJ="/path/to/s1_sfcbf_success_trajs_diverse_5k.npz"
export SOFAI_NEW_S1_BASE_MEMORY_SCENARIOS="/path/to/benchmark_scenarios_maze_diverse_5k.json"
```

### Force memory-neural S1 to run on CPU

```bash
export SOFAI_NEW_S1_DEVICE=cpu
```

### MPC is slow

MPC is the deliberative System 2 solver and can be expensive. For quick debugging, run a single scenario or reduce the number of scenarios. For batch experiments, start with:

```bash
--max_scenarios 20
```

in `run_s1_memory_neural_continual_experiment.py`.

## Development Notes

- Keep high-level SOFAI entry points small.
- Put reusable solver logic in `Solvers/`.
- Put standalone algorithm implementations and training utilities in `Solvers/Base/`.
- Keep `S1_memory_neural.py` independent of any specific S2 solver.
- Use `mpc_solver_new_S1.py` or `cbf_solver_new_S1.py` to pair the memory-neural S1 with a chosen S2.
- Use `run_s1_memory_neural_continual_experiment.py` for offline continual-learning evaluation.

## License

Check the repository license and the upstream SOFAI Tool license before redistribution. The upstream SOFAI Tool repository is available at:

```text
https://github.com/ai4society/sofai_tool/
```
