import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import time
import logging
import traceback
import math
import os
import sys
from pathlib import Path

import numpy as np

'''
MPLCONFIGDIR=/tmp/mpl /Users/apple/miniconda3/envs/s12_env/bin/python3.10 cbf_solver.py \
  --problem_dictionary benchmark_scenarios_maze_1199_block200.json \
  --scenario_id 2
'''


INSTANCE_DIR = Path(__file__).resolve().parent
SOFAI_PACKAGE_ROOT = INSTANCE_DIR.parents[1]
for path in [INSTANCE_DIR, SOFAI_PACKAGE_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# SOFAI
from sofai_tool.solvers import system1 as sofai1
from sofai_tool.solvers import system2 as sofai2
from sofai_tool.metacognition import metacognition_module as meta

# Your modules
from input.input_handler import load_scenarios
from ..S2_cbf import solve_CBF
from ..S1_motion_primitives import solveMotionPrimitives
from ..base.S2_cbf_maze import collision_free_rectangles, goal_reached


PATH_TO_INPUT = "input/"


def _cbf_goal_tol_for(problem_dictionary: str, scenario_id: int) -> float:
    default_tol = float(os.environ.get("SOFAI_CBF_GOAL_TOL", "0.6"))
    try:
        scenarios = json.loads(Path(problem_dictionary).read_text())
        return float(scenarios[scenario_id].get("goal_tol", default_tol))
    except Exception:
        return default_tol


# ============================================================
# SYSTEM 1 (motion primitives)
# ============================================================

class CustomSystem1Solver(sofai1.System1Solver):

    def solve(self, problem_id):
        timer = time.time()

        # --- Init ---
        self.solution_raw = None
        self.solution = "noSolution"
        self.confidence = 0.0

        try:
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            scenarios = json.loads(Path(problem_dictionary).read_text())
            scenario = scenarios[scenario_id]

            states, confidence = solveMotionPrimitives(scenario)

            self.solution_raw = states
            self.solution = states.tolist() if states is not None else "noSolution"
            self.confidence = float(confidence)

        except Exception as e:
            print(f"[System1 ERROR] {e}")
            logging.error(traceback.format_exc())
            self.solution_raw = None
            self.solution = "noSolution"
            self.confidence = 0.0

        self.running_time = time.time() - timer

    def calculate_correctness(self, problem_id):
        # --- No solution case ---
        if self.solution_raw is None:
            self.correctness = 0.0
            return

        try:
            # --- Load case ---
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])
            problems = load_scenarios(problem_dictionary)
            case = problems[scenario_id]

            rects = case.rects
            goal = case.goal
            goal_tol = _cbf_goal_tol_for(problem_dictionary, scenario_id)

            states = np.array(self.solution_raw)

            # --- Use SAME definitions as CBF ---
            collision_free = collision_free_rectangles(states, rects)
            reached = goal_reached(states, goal, goal_tol)

            # --- Hard success ---
            if reached and collision_free:
                self.correctness = 1.0
                return

            # --- Path length ---
            diffs = states[1:] - states[:-1]
            path_length = np.abs(diffs).sum()

            # --- Goal error ---
            dx = states[-1, 0] - goal[0]
            dy = states[-1, 1] - goal[1]
            goal_error = np.sqrt(dx * dx + dy * dy)

            # --- Cheap collision penalty ---
            collision_count = 0 if collision_free else int(len(states) * 0.1)
            collision_count = min(collision_count, 50)

            # --- Soft score ---
            score = path_length + goal_error + 5.0 * collision_count
            self.correctness = 1.0 / (1.0 + score)

        except Exception as e:
            print(f"[System1 ERROR] {e}")
            logging.error(traceback.format_exc())
            self.correctness = 0.0


# ============================================================
# SYSTEM 2 (CBF)
# ============================================================

class CustomSystem2Solver(sofai2.System2Solver):

    def solve(self, problem_id, time_limit):
        timer = time.time()

        # --- Init ---
        self.solution_raw = None
        self.solution = "noSolution"
        self.confidence = 0.0

        try:
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            scenario_to_solve = problems[scenario_id]

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_CBF, scenario_to_solve)

                try:
                    result = future.result(timeout=time_limit)

                    self.solution_raw = result

                    if result is None:
                        self.solution = "noSolution"
                    else:
                        self.solution = result.tolist()
                except TimeoutError:
                    logging.warning("Solver timed out")
                    self.solution_raw = None
                    self.solution = "noSolution"

            self.confidence = 1.0

        except Exception as e:
            print(f"[System2 ERROR] {e}")
            logging.error(traceback.format_exc())
            self.solution_raw = None
            self.solution = "noSolution"
            self.confidence = 0.0

        self.running_time = time.time() - timer

    def estimate_difficulty(self, problem_id):
        eps = 1e-6

        try:
            # --- Load scenario ---
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            case = problems[scenario_id]

            rects = case.rects
            xmin, ymin, xmax, ymax = case.bounds
            start = case.start
            goal = case.goal

            goal_tol = _cbf_goal_tol_for(problem_dictionary, scenario_id)
            collision_margin = float(os.environ.get("SOFAI_CBF_COLLISION_MARGIN", "0.05"))

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

            # --- Midpoint clearance ---
            midx = (start[0] + goal[0]) / 2.0
            midy = (start[1] + goal[1]) / 2.0

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

            geom = occupancy * (1.0 / (min_clearance + eps))
            control = 1.0 + math.log(1.0 + float(os.environ.get("SOFAI_CBF_STEPS", "800")))
            precision = 1.0 + min(10.0, 1.0 / (goal_tol + eps))
            safety = 1.0 + min(10.0, 1.0 / (collision_margin + eps))

            return geom * control * precision * safety

        except Exception:
            logging.error(traceback.format_exc())
            return 0.0

    def calculate_correctness(self, problem_id):
        # --- No solution ---
        if self.solution_raw is None:
            self.correctness = 0.0
            return

        try:
            # --- Load case ---
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])
            problems = load_scenarios(problem_dictionary)
            case = problems[scenario_id]

            rects = case.rects
            goal = case.goal
            goal_tol = _cbf_goal_tol_for(problem_dictionary, scenario_id)

            # --- Use SAME logic as CBF ---
            states = self.solution_raw

            collision_free = collision_free_rectangles(states, rects)
            reached = goal_reached(states, goal, goal_tol)

            self.correctness = 1.0 if (collision_free and reached) else 0.0

        except Exception:
            logging.error(traceback.format_exc())
            self.correctness = 0.0


# ============================================================
# SOFAI ENTRY POINT
# ============================================================

def cbf_solve(problem_name):

    system1_solver = CustomSystem1Solver()
    system2_solver = CustomSystem2Solver()

    context_file = "input/meta/context.txt"
    thresholds_file = "input/meta/thresholds.txt"
    experience_file = "plan_experience.json"

    meta.metacognition(
        problem_name,
        system1_solver,
        system2_solver,
        context_file,
        thresholds_file,
        experience_file,
        new_run=False,
        run_type="sofai"
    )


def main():
    parser = argparse.ArgumentParser(description="Run CBF solver on a scenario.")
    parser.add_argument(
        "--problem_dictionary",
        type=str,
        default="benchmark_scenarios_maze.json",
        help="Path to the problem dictionary JSON file",
    )
    parser.add_argument(
        "--scenario_id",
        type=int,
        default=1,
        help="Scenario ID to run",
    )

    args = parser.parse_args()

    database_name = Path(args.problem_dictionary).stem
    problem_name = f"{database_name}_sc_{args.scenario_id}"

    cbf_solve(problem_name)


if __name__ == "__main__":
    main()
