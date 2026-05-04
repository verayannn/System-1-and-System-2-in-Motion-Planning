import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import time
import logging
import traceback
from pathlib import Path
import numpy as np
import math

# SOFAI
from sofai_tool.solvers import system1 as sofai1
from sofai_tool.solvers import system2 as sofai2
from sofai_tool.metacognition import metacognition_module as meta

from input.input_handler import load_scenarios
from solvers.S2_cbf import solve_CBF
from solvers.S2_mpc import solve_MPC
from solvers.S1_memory_neural import solveMemoryNeural
from solvers.S1_motion_primitives import solveMotionPrimitives
from solvers.base.S2_mpc_maze import (
    collision_free_rectangles,
    goal_reached,
)

PATH_TO_INPUT = "input/"

S1_MODE = "neural"
S2_MODE = "mpc"


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
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            scenarios = json.loads(Path(problem_dictionary).read_text())
            scenario = scenarios[scenario_id]

            if S1_MODE == "neural":
                scenario.setdefault("scenario_id", scenario_id)
                states, confidence = solveMemoryNeural(scenario, return_info=False)
            else:
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
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            case = problems[scenario_id]

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
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            scenario = problems[scenario_id]

            solve_fn = solve_CBF if S2_MODE == "cbf" else solve_MPC

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_fn, scenario)

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
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            sceneario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            case = problems[sceneario_id]

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
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            scenario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            case = problems[scenario_id]

            rects = case.rects
            goal = case.goal
            goal_tol = getattr(case, "goal_tol", 0.5)

            states = self.solution_raw

            collision_free = collision_free_rectangles(states, rects)
            reached = goal_reached(states, goal, goal_tol)

            self.correctness = 1.0 if (collision_free and reached) else 0.0

        except Exception:
            logging.error(traceback.format_exc())
            self.correctness = 0.0


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
    parser.add_argument("--new_run", type=bool, default=False)


    args = parser.parse_args()

    S1_MODE = args.s1
    S2_MODE = args.s2

    name = Path(args.problem_dictionary).stem
    problem_name = f"{name}_sc_{args.scenario_id}"

    run(problem_name, run_type=args.run_type, new_run=args.new_run)


if __name__ == "__main__":
    main()