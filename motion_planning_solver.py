import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import time
import logging
import traceback
import sys
from pathlib import Path
import numpy as np
import math
from typing import Any, Dict, List, Optional

# Make bundled packages importable when running from the repo root.
ROOT = Path(__file__).resolve().parent
SOFAI_PKG = ROOT / "sofai"
if str(SOFAI_PKG) not in sys.path:
    sys.path.insert(0, str(SOFAI_PKG))

# SOFAI
from sofai_tool.solvers import system1 as sofai1
from sofai_tool.solvers import system2 as sofai2
from sofai_tool.metacognition import metacognition_module as meta

from input.input_handler import load_scenarios
from solvers.S2_cbf import solve_CBF
from solvers._s2_common import collision_free_rectangles, goal_reached

PATH_TO_INPUT = "input/"

S1_MODE = "primitives"
S2_MODE = "mpc"
ACTIVE_DICTIONARY_PATH: Optional[Path] = None


def resolve_problem_dictionary(problem_id: str) -> Path:
    stem = problem_id.split("_sc_")[0]
    candidates = []
    if ACTIVE_DICTIONARY_PATH is not None:
        candidates.append(ACTIVE_DICTIONARY_PATH)
    root = Path(__file__).resolve().parent
    candidates.extend(
        [
            root / "input" / f"{stem}.json",
            root / "input" / "nl" / f"{stem}.json",
            root / f"{stem}.json",
            Path(PATH_TO_INPUT) / f"{stem}.json",
        ]
    )
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve dictionary for {problem_id}")


def load_problem_scenario(problem_id: str):
    problem_dictionary = resolve_problem_dictionary(problem_id)
    scenario_id = int(problem_id.split("_sc_")[1])
    scenarios = load_scenarios(str(problem_dictionary))
    return problem_dictionary, scenario_id, scenarios[scenario_id]


'''

python motion_planning_solver.py \
  --problem_dictionary benchmark_dualmp_dense_clutter.json \
  --scenario_id 1 \
  --s1 neural \
  --s2 mpc \
  --run_type s2



cd /Users/apple/Desktop/sofai
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl \
python motion_planning_solver.py \
  --problem_dictionary benchmark_dualmp_nl_dense_clutter.json \
  --scenario_id 3 \
  --s1 neural \
  --s2 cbf \
  --run_type s2


PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl \
/Users/apple/miniconda3/envs/s12_env/bin/python3.10 motion_planning_solver.py \
  --problem_dictionary benchmark_dualmp_nl_dense_clutter.json \
  --scenario_id 3 \
  --s1 neural \
  --s2 mpc \
  --run_type s2


'''


# ============================================================
# SYSTEM 1
# ============================================================

class CustomSystem1Solver(sofai1.System1Solver):

    def solve(self, problem_id):
        timer = time.time()

        self.solution_raw = None
        self.solution = "noSolution"
        self.confidence = 0.0

        try:
            _, scenario_id, scenario = load_problem_scenario(problem_id)

            if S1_MODE == "neural":
                from solvers.S1_memory_neural import solveMemoryNeural

                states, confidence = solveMemoryNeural(scenario, return_info=False)
            else:
                from solvers.S1_motion_primitives import solveMotionPrimitives

                states, confidence = solveMotionPrimitives(scenario)

            self.solution_raw = states
            self.solution = states.tolist() if states is not None else "noSolution"
            self.confidence = float(confidence)

        except Exception:
            logging.error(traceback.format_exc())
            self.solution_raw = None
            self.solution = "noSolution"
            self.confidence = 0.0

        self.running_time = time.time() - timer

    def calculate_correctness(self, problem_id):
        if self.solution_raw is None:
            self.correctness = 0.0
            return

        try:
            _, scenario_id, case = load_problem_scenario(problem_id)

            rects = case.rects
            goal = case.goal
            goal_tol = getattr(case, "goal_tol", 0.5)

            states = np.array(self.solution_raw)

            collision_free = collision_free_rectangles(states, rects)
            reached = goal_reached(states, goal, goal_tol)

            if reached and collision_free:
                self.correctness = 1.0
                return

            diffs = states[1:] - states[:-1]
            path_length = np.abs(diffs).sum()

            dx = states[-1, 0] - goal[0]
            dy = states[-1, 1] - goal[1]
            goal_error = np.sqrt(dx * dx + dy * dy)

            collision_count = 0 if collision_free else int(len(states) * 0.1)
            collision_count = min(collision_count, 50)

            score = path_length + goal_error + 5.0 * collision_count
            self.correctness = 1.0 / (1.0 + score)

        except Exception:
            logging.error(traceback.format_exc())
            self.correctness = 0.0


# ============================================================
# SYSTEM 2
# ============================================================

class CustomSystem2Solver(sofai2.System2Solver):

    def solve(self, problem_id, time_limit):
        timer = time.time()

        self.solution_raw = None
        self.solution = "noSolution"
        self.confidence = 0.0

        try:
            _, scenario_id, scenario = load_problem_scenario(problem_id)

            if S2_MODE == "cbf":
                solve_fn = solve_CBF
            else:
                from solvers.S2_mpc import solve_MPC

                solve_fn = solve_MPC

            with ThreadPoolExecutor(max_workers=1) as executor: 
                future = executor.submit(solve_fn, scenario) ## each executor get a different bck

                try:
                    result = future.result(timeout=time_limit)
                    self.solution_raw = result
                    self.solution = "noSolution" if result is None else result.tolist()
                except TimeoutError:
                    logging.warning("timeout")
                    self.solution = "noSolution"
                    self.solution_raw = None

            self.confidence = 1.0

        except Exception:
            logging.error(traceback.format_exc())
            self.solution = "noSolution"
            self.solution_raw = None
            self.confidence = 0.0

        self.running_time = time.time() - timer


    def estimate_difficulty(self, problem_id):
        eps = 1e-6

        try:
            _, scenario_id, case = load_problem_scenario(problem_id)

            rects = case.rects
            xmin, ymin, xmax, ymax = case.bounds
            start = case.start
            goal = case.goal

            # Defaults (since not in JSON)
            goal_tol = getattr(case, "goal_tol", 0.5)
            collision_margin = getattr(case, "collision_margin", 0.5)
            horizon = getattr(case, "horizon", 10)

            # --- Workspace area ---
            workspace_area = (xmax - xmin) * (ymax - ymin) + eps

            # --- Total obstacle area ---
            total_obs_area = 0.0
            for (x1, y1, x2, y2) in rects:
                total_obs_area += abs((x2 - x1) * (y2 - y1))

            occupancy = total_obs_area / workspace_area

            # --- Distance start-goal ---
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            dist = math.hypot(dx, dy) + eps

            # --- Midpoint ---
            midx = (start[0] + goal[0]) / 2.0
            midy = (start[1] + goal[1]) / 2.0

            # --- Cheap clearance (to rectangle edges, not centers → better) ---
            def point_to_rect_dist(px, py, x1, y1, x2, y2):
                dx = max(x1 - px, 0, px - x2)
                dy = max(y1 - py, 0, py - y2)
                return math.hypot(dx, dy)

            min_clearance = float("inf")
            for (x1, y1, x2, y2) in rects:
                d = point_to_rect_dist(midx, midy, x1, y1, x2, y2)
                if d < min_clearance:
                    min_clearance = d

            if min_clearance == float("inf"):
                min_clearance = dist

            # --- Components ---
            geom = occupancy * (1.0 / (min_clearance + eps))
            control = 1.0 + math.log(1.0 + horizon)

            # Clamp to avoid instability
            precision = 1.0 + min(10.0, 1.0 / (goal_tol + eps))
            safety = 1.0 + min(10.0, 1.0 / (collision_margin + eps))

            difficulty = geom * control * precision * safety

            return difficulty

        except Exception:
            logging.error(traceback.format_exc())
            return 0.0

    def calculate_correctness(self, problem_id):
        if self.solution_raw is None:
            self.correctness = 0.0
            return

        try:
            _, scenario_id, case = load_problem_scenario(problem_id)

            rects = case.rects
            goal = case.goal
            goal_tol = getattr(case, "goal_tol", 0.6)

            states = self.solution_raw

            collision_free = collision_free_rectangles(states, rects)
            reached = goal_reached(states, goal, goal_tol)

            self.correctness = 1.0 if (collision_free and reached) else 0.0

        except Exception:
            logging.error(traceback.format_exc())
            self.correctness = 0.0


# ============================================================
# BENCHMARK CASE HELPERS
# ============================================================

def _scenario_payload(scenario, scenario_id: int) -> Dict[str, Any]:
    payload = {
        "scenario_id": int(getattr(scenario, "scenario_id", scenario_id)),
        "A_query": getattr(scenario, "A", None),
        "B_query": getattr(scenario, "B", None),
        "rectangles": [list(r) for r in getattr(scenario, "rects", [])],
        "start": list(getattr(scenario, "start", (0.0, 0.0))),
        "goal": list(getattr(scenario, "goal", (0.0, 0.0))),
        "bounds": list(getattr(scenario, "bounds", (-10.0, -10.0, 10.0, 10.0))),
        "u_max": float(getattr(scenario, "u_max", 3.0)),
        "goal_tol": float(getattr(scenario, "goal_tol", 0.5)),
    }
    dynamics_type = getattr(scenario, "dynamics_type", None)
    dynamics_model = getattr(scenario, "dynamics_model", None)
    nonlinear_dynamics = getattr(scenario, "nonlinear_dynamics", None)
    if dynamics_type is not None:
        payload["dynamics_type"] = dynamics_type
    if dynamics_model:
        payload["dynamics_model"] = dynamics_model
    if nonlinear_dynamics is not None:
        payload["nonlinear_dynamics"] = nonlinear_dynamics
    return payload


def _augment_attempt(attempt: Dict[str, Any], scenario) -> Dict[str, Any]:
    states_raw = attempt.get("states")
    if states_raw is None:
        attempt.update(
            success=False,
            collision_free=False,
            goal_reached=False,
            final_goal_error=None,
            path_length=None,
            num_states=0,
            correctness=0.0,
        )
        return attempt

    states = np.asarray(states_raw, dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        attempt["states"] = None
        return _augment_attempt(attempt, scenario)

    xy = states[:, :2]
    collision_free = bool(collision_free_rectangles(xy, scenario.rects))
    reached = bool(goal_reached(xy, scenario.goal, getattr(scenario, "goal_tol", 0.5)))
    path_length = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum()) if len(xy) > 1 else 0.0
    final_error = float(np.linalg.norm(xy[-1] - np.asarray(scenario.goal, dtype=float)))
    attempt.update(
        success=bool(collision_free and reached),
        collision_free=collision_free,
        goal_reached=reached,
        final_goal_error=final_error,
        path_length=path_length,
        num_states=int(states.shape[0]),
        correctness=float(attempt.get("correctness", 1.0 if collision_free and reached else 0.0)),
    )
    return attempt


def _attempt_record(name: str, system: str, mode: str, solver, scenario) -> Dict[str, Any]:
    return _augment_attempt(
        {
            "name": name,
            "system": system,
            "mode": mode,
            "states": None if solver.solution_raw is None else np.asarray(solver.solution_raw).tolist(),
            "confidence": float(getattr(solver, "confidence", 0.0)),
            "runtime_sec": float(getattr(solver, "running_time", 0.0)),
            "correctness": float(getattr(solver, "correctness", 0.0)),
        },
        scenario,
    )


def solve_benchmark_case(
    problem_dictionary: str | Path,
    scenario_id: int,
    *,
    s1: str = "neural",
    s2: str = "mpc",
    run_type: str = "sofai",
    run_all_attempts: bool = False,
) -> Dict[str, Any]:
    global S1_MODE, S2_MODE, ACTIVE_DICTIONARY_PATH

    problem_dictionary = Path(problem_dictionary).expanduser()
    if not problem_dictionary.is_file():
        root = Path(__file__).resolve().parent
        candidates = [
            root / problem_dictionary,
            root / "input" / problem_dictionary.name,
            root / "input" / "nl" / problem_dictionary.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                problem_dictionary = candidate
                break
    problem_dictionary = problem_dictionary.resolve()
    ACTIVE_DICTIONARY_PATH = problem_dictionary
    S1_MODE = s1
    S2_MODE = s2

    scenarios = load_scenarios(str(problem_dictionary))
    if scenario_id < 0 or scenario_id >= len(scenarios):
        raise IndexError(f"scenario_id {scenario_id} outside 0..{len(scenarios) - 1}")

    scenario = scenarios[scenario_id]
    problem_name = f"{problem_dictionary.stem}_sc_{scenario_id}"
    timer = time.time()
    attempts: List[Dict[str, Any]] = []

    s1_solver = CustomSystem1Solver()
    s2_solver = CustomSystem2Solver()

    if run_type in {"s1", "sofai"}:
        s1_solver.solve(problem_name)
        s1_solver.calculate_correctness(problem_name)
        attempts.append(_attempt_record(f"s1_{s1}", "s1", s1, s1_solver, scenario))

    s1_attempted = run_type in {"s1", "sofai"}
    s2_attempted = False
    s2_skipped = False

    if run_type == "s2":
        s2_solver.solve(problem_name, 10**9)
        s2_solver.calculate_correctness(problem_name)
        attempts.append(_attempt_record(f"s2_{s2}", "s2", s2, s2_solver, scenario))
        s2_attempted = True
    elif run_type == "sofai":
        need_s2 = run_all_attempts or not attempts or not attempts[0].get("success", False)
        if need_s2:
            s2_solver.solve(problem_name, 10**9)
            s2_solver.calculate_correctness(problem_name)
            attempts.append(_attempt_record(f"s2_{s2}", "s2", s2, s2_solver, scenario))
            s2_attempted = True
        else:
            s2_skipped = True

    selected = attempts[0] if run_type in {"s1", "s2"} else next((a for a in attempts if a.get("success")), attempts[-1] if attempts else None)
    running_time = time.time() - timer
    selected_states = None if selected is None else selected.get("states")

    return {
        "status": "ok",
        "problem_name": problem_name,
        "dictionary": problem_dictionary.name,
        "dictionary_path": str(problem_dictionary),
        "scenario_id": scenario_id,
        "run_type": run_type,
        "s1": s1,
        "s2": s2,
        "scenario": _scenario_payload(scenario, scenario_id),
        "attempts": attempts,
        "s1_attempted": s1_attempted,
        "s2_attempted": s2_attempted,
        "s2_skipped": s2_skipped,
        "selected_attempt": None if selected is None else selected["name"],
        "success": bool(selected and selected.get("success")),
        "collision_free": bool(selected and selected.get("collision_free")),
        "goal_reached": bool(selected and selected.get("goal_reached")),
        "final_goal_error": None if selected is None else selected.get("final_goal_error"),
        "path_length": None if selected is None else selected.get("path_length"),
        "num_states": 0 if selected is None else int(selected.get("num_states", 0)),
        "selected_runtime_sec": None if selected is None else selected.get("runtime_sec"),
        "runtime_sec": running_time,
        "running_time": running_time,
        "timed_out": False,
        "error_message": "",
        "traceback": "",
        "solution_raw": selected_states,
        "solution": "noSolution" if selected_states is None else selected_states,
        "confidence": 0.0 if selected is None else float(selected.get("confidence", 0.0)),
        "correctness": 0.0 if selected is None else float(selected.get("correctness", 0.0)),
        "meta": {
            "problem_name": problem_name,
            "scenario": _scenario_payload(scenario, scenario_id),
            "solution_raw": selected_states,
            "solution": "noSolution" if selected_states is None else selected_states,
            "confidence": 0.0 if selected is None else float(selected.get("confidence", 0.0)),
            "correctness": 0.0 if selected is None else float(selected.get("correctness", 0.0)),
            "running_time": running_time,
        },
    }


# ============================================================
# ENTRY
# ============================================================

def run(problem_name, run_type="sofai", new_run=False):
    meta.metacognition(
        problem_name,
        CustomSystem1Solver(),
        CustomSystem2Solver(),
        "input/meta/context.txt",
        "input/meta/thresholds.txt",
        "plan_experience.json",
        new_run=new_run,
        run_type=run_type,
    )


def main():
    global S1_MODE, S2_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", choices=["neural", "primitives"], default="primitives")
    parser.add_argument("--s2", choices=["cbf", "mpc"], default="mpc")
    parser.add_argument("--problem_dictionary", default="benchmark_scenarios_maze.json")
    parser.add_argument("--scenario_id", type=int, default=1)
    parser.add_argument("--run_type", choices=["sofai", "s1", "s2"], default="sofai")
    parser.add_argument("--new_run", type=bool, default=False) ## true: deletes the experiences // deletes the experiences when starting a new bck -- clean the experiences


    args = parser.parse_args()

    S1_MODE = args.s1
    S2_MODE = args.s2

    name = Path(args.problem_dictionary).stem
    problem_name = f"{name}_sc_{args.scenario_id}"

    run(problem_name, run_type=args.run_type, new_run=args.new_run)


if __name__ == "__main__":
    main()
