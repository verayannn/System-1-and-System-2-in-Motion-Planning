"""
generate_benchmark_scenarios.py

Generate benchmark query scenarios by perturbing existing
System-1 dynamics (A, M) and sampling new obstacles.
Compatible with S1_usage.py.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

def perturb_A(A_base: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    noise = noise_level * np.random.randn(*A_base.shape)
    return A_base + noise

def perturb_M(M_base: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    noise = noise_level * np.random.randn(*M_base.shape)
    return M_base + noise

def generate_scenarios(
    db_json: str,
    n_scenarios: int = 100,
    noise_level: float = 0.05,
    obstacle_range: Tuple[float, float] = (-4.0, 4.0), # Keeping obstacle strictly between start/goal
    seed: int = 42,
) -> List[Dict[str, Any]]:

    rng = np.random.default_rng(seed)
    payload = json.loads(Path(db_json).read_text())
    db = payload["db"]
    dyn_ids = sorted(db["dyn_nodes"].keys(), key=lambda x: int(x))
    scenarios = []

    for k in range(n_scenarios):
        base_dyn_id = int(rng.choice(dyn_ids))
        dyn_node = db["dyn_nodes"][str(base_dyn_id)]

        A_base = np.array(dyn_node["A"], dtype=float)
        M_base = np.array(dyn_node["M"], dtype=float)

        A_query = perturb_A(A_base, noise_level=noise_level)
        M_query = perturb_M(M_base, noise_level=noise_level)

        # Generate random obstacle center
        obstacle_center = [
            float(rng.uniform(*obstacle_range)),
            float(rng.uniform(*obstacle_range)),
        ]

        scenarios.append(
            {
                "scenario_id": k,
                "base_dyn_id": base_dyn_id,
                "A_query": A_query.tolist(),
                "M_query": M_query.tolist(),
                "obstacle_center": obstacle_center, # Top-level key expected by S1_usage.py
                "task_params": {
                    "start_x": 5.0,
                    "start_y": 5.0,
                    "goal_x": 0.0,
                    "goal_y": 0.0,
                    "base_radius": 1.5 
                }
            }
        )

    return scenarios

if __name__ == "__main__":
    scenarios = generate_scenarios(
        db_json="S1_database_single_obstacle.json",
        n_scenarios=100
    )
    out_path = Path("benchmark_scenarios.json")
    out_path.write_text(json.dumps(scenarios, indent=2))
    print(f"✅ Generated {len(scenarios)} scenarios -> {out_path}")