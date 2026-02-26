"""
generate_benchmark_scenarios_maze.py  (SCATTER BLOCKS, FIXED START/GOAL)

Writes:
  1) --out       : full benchmark scenarios (for S2 + reports)
  2) --out_query : S1-friendly query scenarios (minimal fields)

Fixed:
- bounds: [-10,10] x [-10,10]
- start: (5,5)
- goal:  (0,0)

Run:
  python maze/generate_benchmark_scenarios_maze.py \
    --out maze/benchmark_scenarios_maze.json \
    --out_query maze/benchmark_queries_maze.json \
    --n 500

(50 in the old bck report)

S2 MPC / CBF benchmarking uses benchmark_scenarios_maze.json
S1 retrieval can use benchmark_queries_maze.json (minimal, clean)

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]


# ============================
# Dynamics sampling
# ============================

def sample_stable_A(seed: int) -> np.ndarray:
    rng = random.Random(seed)
    a11 = -rng.uniform(0.2, 1.2)
    a22 = -rng.uniform(0.2, 1.2)
    a12 = rng.uniform(-0.4, 0.4)
    a21 = rng.uniform(-0.4, 0.4)
    return np.array([[a11, a12], [a21, a22]], dtype=float)


# ============================
# Geometry helpers
# ============================

def point_in_rect(p: Tuple[float, float], r: Rect, margin: float = 0.0) -> bool:
    x, y = p
    xmin, ymin, xmax, ymax = r
    m = float(margin)
    return (xmin - m <= x <= xmax + m) and (ymin - m <= y <= ymax + m)


def rects_overlap(a: Rect, b: Rect, gap: float = 0.0) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    g = float(gap)
    return not (ax2 + g <= bx1 or bx2 + g <= ax1 or ay2 + g <= by1 or by2 + g <= ay1)


# ============================
# Connectivity (optional)
# ============================

def rasterize(rects: List[Rect], bounds: Rect, cell: float) -> Tuple[np.ndarray, Tuple[float, float]]:
    xmin, ymin, xmax, ymax = bounds
    W = int(np.ceil((xmax - xmin) / cell))
    H = int(np.ceil((ymax - ymin) / cell))
    occ = np.zeros((H, W), dtype=np.uint8)

    def world_to_cell_xy(x: float, y: float) -> Tuple[int, int]:
        j = int((x - xmin) // cell)
        i = int((y - ymin) // cell)
        return i, j

    for (rx1, ry1, rx2, ry2) in rects:
        i1, j1 = world_to_cell_xy(rx1, ry1)
        i2, j2 = world_to_cell_xy(rx2, ry2)
        i1 = max(i1, 0); j1 = max(j1, 0)
        i2 = min(i2, H - 1); j2 = min(j2, W - 1)
        occ[i1:i2+1, j1:j2+1] = 1

    return occ, (xmin, ymin)


def world_to_cell(p: Tuple[float, float], origin: Tuple[float, float], cell: float) -> Tuple[int, int]:
    x, y = p
    ox, oy = origin
    j = int((x - ox) // cell)
    i = int((y - oy) // cell)
    return i, j


def bfs_reachable(occ: np.ndarray, s: Tuple[int, int], g: Tuple[int, int]) -> bool:
    H, W = occ.shape
    si, sj = s
    gi, gj = g
    if not (0 <= si < H and 0 <= sj < W and 0 <= gi < H and 0 <= gj < W):
        return False
    if occ[si, sj] == 1 or occ[gi, gj] == 1:
        return False

    q = [(si, sj)]
    vis = np.zeros_like(occ, dtype=np.uint8)
    vis[si, sj] = 1
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)]

    head = 0
    while head < len(q):
        i, j = q[head]
        head += 1
        if (i, j) == (gi, gj):
            return True
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and vis[ni, nj] == 0 and occ[ni, nj] == 0:
                vis[ni, nj] = 1
                q.append((ni, nj))
    return False


# ============================
# Scatter blocks
# ============================

def sample_rect(
    rng: random.Random,
    bounds: Rect,
    w_range: Tuple[float, float],
    h_range: Tuple[float, float],
) -> Rect:
    xmin, ymin, xmax, ymax = bounds
    w = rng.uniform(*w_range)
    h = rng.uniform(*h_range)
    x1 = rng.uniform(xmin, xmax - w)
    y1 = rng.uniform(ymin, ymax - h)
    return (float(x1), float(y1), float(x1 + w), float(y1 + h))


def generate_scatter_rectangles(
    seed: int,
    *,
    bounds: Rect,
    n_rect: int,
    w_range: Tuple[float, float],
    h_range: Tuple[float, float],
    min_gap: float,
    fixed_points: List[Tuple[float, float]],
    avoid_margin: float,
    max_tries: int = 50000,
) -> List[Rect]:
    rng = random.Random(seed)
    rects: List[Rect] = []
    tries = 0

    while len(rects) < n_rect and tries < max_tries:
        tries += 1
        r = sample_rect(rng, bounds, w_range, h_range)

        # avoid fixed points (start/goal) with margin
        if any(point_in_rect(p, r, margin=avoid_margin) for p in fixed_points):
            continue

        # keep separation between blocks
        if any(rects_overlap(r, rr, gap=min_gap) for rr in rects):
            continue

        rects.append(r)

    return rects


def generate_one_scenario(
    scenario_id: int,
    *,
    seed: int,
    bounds: Rect,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    cell_size: float,
    n_rect: int,
    w_range: Tuple[float, float],
    h_range: Tuple[float, float],
    min_gap: float,
    avoid_margin: float,
    ensure_connectivity: bool,
    connectivity_cell: float,
) -> Dict[str, object]:

    rects = generate_scatter_rectangles(
        seed=seed + 17,
        bounds=bounds,
        n_rect=n_rect,
        w_range=w_range,
        h_range=h_range,
        min_gap=min_gap,
        fixed_points=[start, goal],
        avoid_margin=avoid_margin,
    )

    if ensure_connectivity:
        for t in range(10):
            occ, origin = rasterize(rects, bounds, cell=connectivity_cell)
            s_ij = world_to_cell(start, origin, connectivity_cell)
            g_ij = world_to_cell(goal, origin, connectivity_cell)
            if bfs_reachable(occ, s_ij, g_ij):
                break

            rects = generate_scatter_rectangles(
                seed=seed + 1000 + t,
                bounds=bounds,
                n_rect=max(6, int(0.85 * n_rect)),
                w_range=w_range,
                h_range=h_range,
                min_gap=min_gap,
                fixed_points=[start, goal],
                avoid_margin=avoid_margin,
            )

    A = sample_stable_A(seed + 23)
    B = np.eye(2, dtype=float)

    return {
        "scenario_id": int(scenario_id),
        "A_query": A.tolist(),
        "B_query": B.tolist(),
        "rectangles": [list(map(float, r)) for r in rects],
        "bounds": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
        "cell_size": float(cell_size),
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="maze/benchmark_scenarios_maze.json")
    ap.add_argument("--out_query", type=str, default="maze/benchmark_queries_maze.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)

    # fixed world / points
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--ymin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--ymax", type=float, default=10.0)
    ap.add_argument("--start_x", type=float, default=5.0)
    ap.add_argument("--start_y", type=float, default=5.0)
    ap.add_argument("--goal_x", type=float, default=0.0)
    ap.add_argument("--goal_y", type=float, default=0.0)

    # obstacle density
    ap.add_argument("--n_rect", type=int, default=26)
    ap.add_argument("--w_min", type=float, default=0.7)
    ap.add_argument("--w_max", type=float, default=2.0)
    ap.add_argument("--h_min", type=float, default=0.7)
    ap.add_argument("--h_max", type=float, default=2.0)
    ap.add_argument("--min_gap", type=float, default=0.15)

    ap.add_argument("--avoid_margin", type=float, default=1.0)
    ap.add_argument("--cell_size", type=float, default=0.5)

    ap.add_argument("--ensure_connectivity", action="store_true")
    ap.add_argument("--conn_cell", type=float, default=0.5)

    # optional: write a *minimal* query schema
    ap.add_argument("--query_minimal", action="store_true",
                    help="If set, out_query contains only {scenario_id,A_query,B_query,rectangles}.")

    args = ap.parse_args()

    bounds = (args.xmin, args.ymin, args.xmax, args.ymax)
    start = (args.start_x, args.start_y)
    goal = (args.goal_x, args.goal_y)

    full: List[Dict[str, object]] = []
    for k in range(int(args.n)):
        seed = int(args.seed) + 97 * k
        sc = generate_one_scenario(
            scenario_id=k,
            seed=seed,
            bounds=bounds,
            start=start,
            goal=goal,
            cell_size=float(args.cell_size),
            n_rect=int(args.n_rect),
            w_range=(float(args.w_min), float(args.w_max)),
            h_range=(float(args.h_min), float(args.h_max)),
            min_gap=float(args.min_gap),
            avoid_margin=float(args.avoid_margin),
            ensure_connectivity=bool(args.ensure_connectivity),
            connectivity_cell=float(args.conn_cell),
        )
        full.append(sc)

    # build query list
    if args.query_minimal:
        query = [
            {
                "scenario_id": int(sc["scenario_id"]),
                "A_query": sc["A_query"],
                "B_query": sc["B_query"],
                "rectangles": sc["rectangles"],
            }
            for sc in full
        ]
    else:
        # query includes bounds/start/goal too (usually safer)
        query = [
            {
                "scenario_id": int(sc["scenario_id"]),
                "A_query": sc["A_query"],
                "B_query": sc["B_query"],
                "rectangles": sc["rectangles"],
                "bounds": sc["bounds"],
                "start": sc["start"],
                "goal": sc["goal"],
            }
            for sc in full
        ]

    Path(args.out).write_text(json.dumps(full, indent=2))
    Path(args.out_query).write_text(json.dumps(query, indent=2))

    print(f"✅ wrote full benchmark: {args.out} ({len(full)} scenarios)")
    print(f"✅ wrote S1 queries:     {args.out_query} ({len(query)} scenarios)")
    print(f"   bounds={bounds}")
    print(f"   start={start}, goal={goal}")
    print(f"   example rectangles={len(full[0]['rectangles'])}")


if __name__ == "__main__":
    main()
