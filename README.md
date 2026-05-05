# System-1-and-System-2-in-Motion-Planning

This repository runs motion-planning experiments with a configurable System 1 / System 2 architecture.

System 1 provides fast approximate planning. System 2 provides slower deliberative planning. The solver can run either system alone, or run in `sofai` mode where the metacognition layer decides whether to accept System 1 or fall back to System 2.

## Solvers

System 1 options:

- `primitives`: motion-primitive retrieval
- `neural`: memory + neural policy

System 2 options:

- `mpc`: model predictive control
- `cbf`: control barrier function solver

Run modes:

- `s1`: run only System 1
- `s2`: run only System 2
- `sofai`: run the combined System 1 / System 2 architecture

## Install

Create and activate your virtual environment first. Then install the local SOFAI package included in this repo:

```bash
uv pip install sofai/.
```

If solver dependencies are missing:

```bash
uv pip install numpy matplotlib casadi do-mpc cvxpy torch
```

## Launch

Run commands from the repo root:

```bash
cd /Users/apple/Desktop/sofai
```

Run neural System 1 only:

```bash
python3 motion_planning_solver.py --s2 mpc --s1 neural --run_type s1
```

Run SOFAI with primitive System 1 and CBF System 2:

```bash
python3 motion_planning_solver.py --s2 cbf --s1 primitives --run_type sofai
```

Run MPC System 2 only:

```bash
python3 motion_planning_solver.py --s2 mpc --s1 primitives --run_type s2
```

Run a specific scenario:

```bash
python3 motion_planning_solver.py \
  --problem_dictionary benchmark_scenarios_maze.json \
  --scenario_id 1 \
  --s1 primitives \
  --s2 mpc \
  --run_type sofai
```

## Main Entry Point

```text
motion_planning_solver.py
```

Important arguments:

```text
--s1                  neural | primitives
--s2                  mpc | cbf
--run_type            s1 | s2 | sofai
--problem_dictionary  benchmark JSON file inside input/
--scenario_id         scenario index in the benchmark JSON
--new_run             start a new SOFAI experience run
```

## Repository Structure

```text
input/                         Benchmark scenario files
input/meta/                    SOFAI context and threshold files
db/                            Experience database and S1 primitive data
solvers/                       Main S1 and S2 solver wrappers
solvers/base/                  Base MPC, CBF, neural, and training code
solvers/combinations/          Older combined solver scripts
sofai/                         Local SOFAI package dependency
output/                        Generated plots and benchmark outputs
```

## Plot One Scenario

```bash
python3 run_and_plot_single_benchmark.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_ids 6 \
  --s1 primitives \
  --s2 cbf \
  --run_type s2 \
  --out_dir output/single_scenario_runs/dense_clutter_demo \
  --out_prefix dense_clutter_sc6_s2
```

## Run Benchmarks

```bash
python3 run_motion_planning_benchmarks.py \
  --patterns benchmark_dualmp_dense_clutter.json \
  --scenario_ids 0-9 \
  --s1 primitives \
  --s2 mpc \
  --run_type sofai \
  --timeout_sec 300 \
  --out_dir output/benchmark_runs/dense_clutter_mpc_primitives \
  --out_prefix dense_clutter_mpc_primitives
```

## Outputs

Solver runs write results to:

```text
db/plan_experience.json
output/
```

Batch benchmark scripts write CSV/JSONL summaries and plots under the selected `--out_dir`.
