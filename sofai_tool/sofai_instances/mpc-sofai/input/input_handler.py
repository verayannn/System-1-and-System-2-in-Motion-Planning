from dataclasses import dataclass
from typing import List, Tuple, Any, Dict
import json
from pathlib import Path


@dataclass
class MazeProblem:
    scenario_id: int
    A: Any
    B: Any
    rects: List[Tuple[float, float, float, float]]
    start: Tuple[float, float]
    goal: Tuple[float, float]
    bounds: Tuple[float, float, float, float]
    u_max: float
    goal_tol: float = 0.5

    @staticmethod
    def from_dict(sc: Dict[str, Any], default_u_max: float = 3.0, idx: int = 0):
        sid = int(sc.get("scenario_id", idx))

        A = sc.get("A_query", sc.get("A"))
        B = sc.get("B_query", sc.get("B"))
        if A is None or B is None:
            raise KeyError(
                f"Scenario {sid} missing A_query/B_query (or A/B). Keys={list(sc.keys())}"
            )

        rects = [tuple(map(float, r)) for r in sc["rectangles"]]

        start = tuple(map(float, sc.get("start", (5.0, 5.0))))
        goal  = tuple(map(float, sc.get("goal", (0.0, 0.0))))
        bounds = tuple(map(float, sc.get("bounds", (-10, -10, 10, 10))))

        u_max = float(sc.get("u_max", default_u_max))

        return MazeProblem(
            scenario_id=sid,
            A=A,
            B=B,
            rects=rects,
            start=start,
            goal=goal,
            bounds=bounds,
            u_max=u_max,
            goal_tol=float(sc.get("goal_tol", 0.5))
        )


def load_scenarios(path: str, default_u_max: float = 3.0):
    scenarios = json.loads(Path(path).read_text())

    if not isinstance(scenarios, list):
        raise TypeError("Expected scenarios JSON to be a list of scenario dicts.")

    return [
        MazeProblem.from_dict(sc, default_u_max, k)
        for k, sc in enumerate(scenarios)
    ]