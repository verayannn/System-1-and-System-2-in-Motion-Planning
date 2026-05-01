import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import logging
import traceback
import math

# SOFAI
from pathlib import Path
import numpy as np
from sofai_tool.solvers import system1 as sofai1
from sofai_tool.solvers import system2 as sofai2
from sofai_tool.metacognition import metacognition_module as meta

# Your modules
from input.input_handler import load_scenarios
from Solvers.S2_mpc import solve_MPC
from Solvers.S1_motion_primitives import solveMotionPrimitives
from Solvers.Base.S2_mpc_maze import collision_free_rectangles, goal_reached


PATH_TO_INPUT = "input/"

# ============================================================
# SYSTEM 1 (simple fallback)
# ============================================================

class CustomSystem1Solver(sofai1.System1Solver):

    def solve(self, problem_id):
        timer = time.time()

        # --- Init ---
        self.solution_raw = None
        self.solution = "noSolution"

        try:
            problem_dictionary = PATH_TO_INPUT + problem_id.split("_sc_")[0] + ".json"
            sceneario_id = int(problem_id.split("_sc_")[1])

            scenarios = json.loads(Path(problem_dictionary).read_text())
            scenario = scenarios[sceneario_id]

            states, confidence = solveMotionPrimitives(scenario)

            self.solution_raw = states
            self.solution = states.tolist() if states is not None else "noSolution"
            self.confidence = confidence
            #print(f"[System1] Confidence: {self.confidence:.4f}")

        except Exception as e:
            print(f"[System1 ERROR] {e}")
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
            goal_tol = getattr(case, "goal_tol", 0.5)

            states = np.array(self.solution_raw)

            # --- Use SAME definitions as MPC ---
            collision_free = collision_free_rectangles(states, rects)
            reached = goal_reached(states, goal, goal_tol)

            # --- Hard success ---
            if reached and collision_free:
                self.correctness = 1.0
                return

            # --- Path length (keep your fast approx) ---
            diffs = states[1:] - states[:-1]
            path_length = np.abs(diffs).sum()

            # --- Goal error (make it consistent: L2 like MPC) ---
            dx = states[-1,0] - goal[0]
            dy = states[-1,1] - goal[1]
            goal_error = np.sqrt(dx*dx + dy*dy)

            # --- Collision count (reuse logic properly) ---
            collision_mask = ~collision_free_rectangles(states, rects, margin=0.0)
            collision_count = 0 if collision_free else int(len(states) * 0.1)  # cheap approx
            collision_count = min(collision_count, 50)

            # --- Soft score ---
            score = path_length + goal_error + 5.0 * collision_count
            self.correctness = 1.0 / (1.0 + score)

        except Exception as e:
            print(f"[System1 ERROR] {e}")
            logging.error(traceback.format_exc())
            self.correctness = 0.0


# ============================================================
# SYSTEM 2 (MPC)
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
            sceneario_id = int(problem_id.split("_sc_")[1])

            problems = load_scenarios(problem_dictionary)
            scenario_to_solve = problems[sceneario_id]

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_MPC, scenario_to_solve)

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
            goal_tol = getattr(case, "goal_tol", 0.5)


            # --- Use SAME logic as MPC ---
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

def mpc_solve(problem_name):

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
    parser = argparse.ArgumentParser(description="Run MPC solver on a scenario.")
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

    mpc_solve(problem_name)


if __name__ == "__main__":
    main()