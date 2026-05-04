"""
S1_layers_maze.py — Build System-1 DB for SCATTER-BLOCK "maze" using your RAG-like situation vector.

Key idea (your proposal):
- Discretize plane into an N x N grid (labels bottom->top, left->right).
- Simulate a NOMINAL trajectory with NO obstacles (under same dynamics) from start->goal.
- Mark all grid cells the nominal trajectory visits.
- Dilate (buffer) that visited mask by `buffer_cells` (Manhattan disk).
- Rasterize obstacles into grid cells.
- The "area-that-matters" mask = visited_mask OR (obstacle_cells AND buffered_corridor_mask)
- Flatten to a binary vector v in {0,1}^{N^2}.
- Similarity between situations: cosine(v_query, v_db).

DB hierarchy:
  ConsensusDyn(cluster_id)
    -> Dyn(dyn_id)
      -> EnvType("maze")  (scatter rectangles)
        -> MazeInstance(map_idx) with:
             rectangles + precomputed situation vector for THIS dyn (nominal traj depends on A,B)
             + (optional) other features

This script:
- Samples stable dynamics with "consensus clusters" using your alpha-alignment routine (kept).
- Generates 64 shared scatter-rectangle maps (same maps reused for every dyn).
- For EACH dyn and map, computes the situation vector using nominal (no-obstacle) rollout.

Fixed world:
- bounds = (-10, -10, 10, 10)
- start  = (5, 5)
- goal   = (0, 0)

Run:
  python maze/S1_layers_maze.py --out_json maze/S1_database_maze.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

Rect = Tuple[float, float, float, float]


# =============================================================================
# 0) Alignment feasibility (same as your original)
# =============================================================================

def hermitian_part(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.conj().T)

def skew_hermitian_part(X: np.ndarray) -> np.ndarray:
    return (X - X.conj().T) / (2j)

def is_alignable(
    M_list: List[np.ndarray],
    alpha: float,
    solver: str = "SCS",
    verbose: bool = False,
) -> Tuple[bool, Optional[np.ndarray]]:
    if len(M_list) == 0:
        return False, None

    m, n = M_list[0].shape
    K = cp.Variable((n, n), complex=True)
    tan_alpha = np.tan(alpha)
    constraints = []

    for M in M_list:
        if M.shape != (m, n):
            raise ValueError("All M in M_list must have the same shape.")
        MK = M @ K
        Re = hermitian_part(MK)
        Im = skew_hermitian_part(MK)

        constraints.append(Re - (M @ M.conj().T) >> 0)
        constraints.append(Im - tan_alpha * Re << 0)
        constraints.append(-Im - tan_alpha * Re << 0)

    prob = cp.Problem(cp.Minimize(cp.norm(K, "fro")), constraints)

    try:
        prob.solve(solver=solver, verbose=verbose)
    except cp.SolverError:
        return False, None

    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return True, K.value
    return False, None

def system_to_M(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega0: float = 1.0) -> np.ndarray:
    n = A.shape[0]
    jwI_minus_A = 1j * omega0 * np.eye(n) - A
    M_complex = C @ np.linalg.inv(jwI_minus_A) @ B
    return np.real(M_complex)


# =============================================================================
# 1) Stable dynamics sampling (same spirit)
# =============================================================================

def _normalize(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(M, 2)
    return M / max(n, eps)

def _is_hurwitz(A: np.ndarray) -> bool:
    return np.all(np.real(np.linalg.eigvals(A)) < 0)

def _make_stable_A(
    m: int = 2,
    rng: Optional[np.random.Generator] = None,
    theta: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng() if rng is None else rng

    R0 = np.array([[0.0, -1.0],
                   [1.0,  0.0]])

    if theta is None:
        theta = float(rng.uniform(-np.pi, np.pi))

    R = np.cos(theta) * R0

    S0 = rng.standard_normal((m, m))
    S = 0.5 * (S0 + S0.T)

    K0 = rng.standard_normal((m, m))
    K = 0.5 * (K0 - K0.T)

    S = _normalize(S)
    K = _normalize(K)

    alpha = float(rng.uniform(0.3, 1.0))
    beta = float(rng.uniform(0.05, 0.25))
    eps_s = float(rng.uniform(0.01, 0.08))
    eps_k = float(rng.uniform(0.01, 0.08))

    A = -alpha * np.eye(m) + beta * R + eps_s * S + eps_k * K

    if not _is_hurwitz(A):
        lam = np.linalg.eigvals(A)
        shift = max(0.0, float(np.max(np.real(lam)) + 1e-3))
        A = A - shift * np.eye(m)

    return A, theta

def _perturb_A(A0: np.ndarray, rng: np.random.Generator, scale: float = 0.12) -> np.ndarray:
    m = A0.shape[0]
    dA = rng.standard_normal((m, m))
    dA = _normalize(dA) * float(rng.uniform(0.0, scale))
    A = A0 + dA

    if not _is_hurwitz(A):
        lam = np.linalg.eigvals(A)
        shift = max(0.0, float(np.max(np.real(lam)) + 1e-3))
        A = A - shift * np.eye(m)

    return A

def generate_consensus_then_clustered_dynamics(
    *,
    K_consensus: int = 5,
    n_per_consensus: int = 10,
    m: int = 2,
    seed: int = 1,
    alpha_within: float = np.deg2rad(5),
    omega0: float = 1.0,
    solver: str = "SCS",
    verbose: bool = False,
    max_tries_per_member: int = 300,
    perturb_scale: float = 0.12,
    phase_centers_deg: Tuple[float, ...] = (0, 25, -25, 50, -50, 75),
    B: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[np.ndarray], Dict[str, Any]]:
    if K_consensus <= 0 or n_per_consensus <= 0:
        raise ValueError("K_consensus and n_per_consensus must be positive integers.")

    rng = np.random.default_rng(seed)
    phase_centers = np.deg2rad(np.array(phase_centers_deg, dtype=float))

    if B is None:
        B = np.eye(2)
    if C is None:
        C = np.eye(2)

    consensus_As: List[np.ndarray] = []
    consensus_thetas: List[float] = []
    for _ in range(K_consensus):
        theta = float(rng.choice(phase_centers))
        A0, th = _make_stable_A(m=m, rng=rng, theta=theta)
        consensus_As.append(A0)
        consensus_thetas.append(float(th))

    clusters_A: List[List[np.ndarray]] = []
    clusters_M: List[List[np.ndarray]] = []
    tries_used: List[List[int]] = []

    for c_idx, A0 in enumerate(consensus_As):
        M0 = system_to_M(A0, B, C, omega0=omega0)

        cluster_A = [A0]
        cluster_M = [M0]
        cluster_tries = [0]

        while len(cluster_A) < n_per_consensus:
            accepted = False
            for t in range(1, max_tries_per_member + 1):
                A_cand = _perturb_A(A0, rng=rng, scale=perturb_scale)
                M_cand = system_to_M(A_cand, B, C, omega0=omega0)

                feasible, _ = is_alignable(cluster_M + [M_cand], alpha_within, solver=solver, verbose=verbose)
                if feasible:
                    cluster_A.append(A_cand)
                    cluster_M.append(M_cand)
                    cluster_tries.append(t)
                    accepted = True
                    break

            if not accepted:
                raise RuntimeError(
                    f"[Cluster {c_idx}] Failed to sample an alpha-alignable member.\n"
                    f"Try: increase max_tries_per_member, increase alpha_within, or reduce perturb_scale."
                )

        clusters_A.append(cluster_A)
        clusters_M.append(cluster_M)
        tries_used.append(cluster_tries)

    meta = {
        "K_consensus": K_consensus,
        "n_per_consensus": n_per_consensus,
        "m": m,
        "seed": seed,
        "alpha_within_rad": float(alpha_within),
        "alpha_within_deg": float(np.rad2deg(alpha_within)),
        "omega0": float(omega0),
        "solver": solver,
        "perturb_scale": float(perturb_scale),
        "max_tries_per_member": int(max_tries_per_member),
        "consensus_thetas": consensus_thetas,
        "phase_centers_deg": list(phase_centers_deg),
        "B": B.tolist(),
        "C": C.tolist(),
        "tries_used": tries_used,
    }
    return clusters_A, clusters_M, consensus_As, meta


# =============================================================================
# 2) SCATTER-BLOCK maps (shared)
# =============================================================================

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

def sample_rect(rng: random.Random, bounds: Rect, w_range: Tuple[float, float], h_range: Tuple[float, float]) -> Rect:
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
    border: bool = False,
    border_thick: float = 0.4,
    max_tries: int = 200000,
) -> List[Rect]:
    rng = random.Random(seed)
    rects: List[Rect] = []

    xmin, ymin, xmax, ymax = bounds
    if border:
        rects.extend([
            (xmin, ymin, xmax, ymin + border_thick),
            (xmin, ymax - border_thick, xmax, ymax),
            (xmin, ymin, xmin + border_thick, ymax),
            (xmax - border_thick, ymin, xmax, ymax),
        ])

    tries = 0
    while len(rects) < (n_rect + (4 if border else 0)) and tries < max_tries:
        tries += 1
        r = sample_rect(rng, bounds, w_range, h_range)

        if any(point_in_rect(p, r, margin=avoid_margin) for p in fixed_points):
            continue
        if any(rects_overlap(r, rr, gap=min_gap) for rr in rects):
            continue

        rects.append(r)

    return rects


# =============================================================================
# 3) Situation vector computation (your “area that matters”)
# =============================================================================

def _clip_to_bounds(x: np.ndarray, bounds: Rect) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    x = x.copy()
    x[0] = float(np.clip(x[0], xmin, xmax))
    x[1] = float(np.clip(x[1], ymin, ymax))
    return x

def u_nominal_no_obs(x: np.ndarray, goal: np.ndarray, A: np.ndarray, B: np.ndarray, u_max: float) -> np.ndarray:
    """
    Simple nominal: least-squares to move toward goal in state derivative.
    (No scipy dependency here, stable enough for your use as a 'reference' path.)
    """
    x = x.reshape(2)
    g = goal.reshape(2)
    # desired xdot direction
    v = -(x - g)
    rhs = v - A @ x
    u, *_ = np.linalg.lstsq(B, rhs, rcond=None)
    u = np.clip(u, -u_max, u_max)
    return u.reshape(-1)

def simulate_nominal_trajectory(
    A: np.ndarray,
    B: np.ndarray,
    *,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    dt: float,
    n_steps: int,
    u_max: float,
    bounds: Rect,
    stop_tol: float = 0.6,
) -> np.ndarray:
    A = np.asarray(A, dtype=float).reshape(2, 2)
    B = np.asarray(B, dtype=float)
    x = np.array(start, dtype=float).reshape(2)
    g = np.array(goal, dtype=float).reshape(2)

    X = [x.copy()]
    for _ in range(int(n_steps)):
        if float(np.sum((x - g) ** 2)) <= float(stop_tol) ** 2:
            break
        u = u_nominal_no_obs(x, g, A, B, u_max=u_max)
        xdot = A @ x + B @ u
        x = x + dt * xdot
        x = _clip_to_bounds(x, bounds)
        X.append(x.copy())
    return np.array(X, dtype=float)

def world_to_grid_ij(
    p: Tuple[float, float],
    *,
    bounds: Rect,
    grid_n: int,
) -> Optional[Tuple[int, int]]:
    """
    Grid indexing convention to match your labeling:
      - columns j: left -> right
      - rows i: bottom -> top
    So flatten index = i*grid_n + j corresponds to:
      1..grid_n^2 from bottom-left to top-right (if +1)
    """
    x, y = float(p[0]), float(p[1])
    xmin, ymin, xmax, ymax = bounds
    if x < xmin or x > xmax or y < ymin or y > ymax:
        return None

    # map to [0, grid_n)
    j = int(np.floor((x - xmin) / (xmax - xmin) * grid_n))
    i = int(np.floor((y - ymin) / (ymax - ymin) * grid_n))
    # clamp edge case when x==xmax or y==ymax
    j = min(max(j, 0), grid_n - 1)
    i = min(max(i, 0), grid_n - 1)
    return i, j

def rasterize_rectangles_to_grid(
    rects: List[Rect],
    *,
    bounds: Rect,
    grid_n: int,
) -> np.ndarray:
    occ = np.zeros((grid_n, grid_n), dtype=np.uint8)

    xmin, ymin, xmax, ymax = bounds
    sx = (xmax - xmin) / grid_n
    sy = (ymax - ymin) / grid_n

    for (rx1, ry1, rx2, ry2) in rects:
        ax1, ax2 = (min(rx1, rx2), max(rx1, rx2))
        ay1, ay2 = (min(ry1, ry2), max(ry1, ry2))

        # convert to cell ranges (inclusive)
        j1 = int(np.floor((ax1 - xmin) / sx))
        j2 = int(np.floor((ax2 - xmin) / sx))
        i1 = int(np.floor((ay1 - ymin) / sy))
        i2 = int(np.floor((ay2 - ymin) / sy))

        j1 = max(j1, 0); j2 = min(j2, grid_n - 1)
        i1 = max(i1, 0); i2 = min(i2, grid_n - 1)

        occ[i1:i2+1, j1:j2+1] = 1

    return occ

def dilate_manhattan(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    H, W = mask.shape
    out = mask.copy()
    ones = np.argwhere(mask > 0)
    for (i, j) in ones:
        for di in range(-radius, radius + 1):
            rem = radius - abs(di)
            i2 = i + di
            if i2 < 0 or i2 >= H:
                continue
            j_lo = j - rem
            j_hi = j + rem
            j_lo = max(j_lo, 0)
            j_hi = min(j_hi, W - 1)
            out[i2, j_lo:j_hi+1] = 1
    return out

def situation_vector(
    A: np.ndarray,
    B: np.ndarray,
    rects: List[Rect],
    *,
    bounds: Rect,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    grid_n: int,
    dt_nom: float,
    n_steps_nom: int,
    u_max_nom: float,
    buffer_cells: int,
    stop_tol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns:
      v: (grid_n^2,) uint8 in {0,1}
      dbg: small debug info (visited count etc.)
    """
    X = simulate_nominal_trajectory(
        A=A, B=B,
        start=start, goal=goal,
        dt=dt_nom, n_steps=n_steps_nom,
        u_max=u_max_nom, bounds=bounds, stop_tol=stop_tol,
    )

    visited = np.zeros((grid_n, grid_n), dtype=np.uint8)
    for k in range(X.shape[0]):
        ij = world_to_grid_ij((X[k, 0], X[k, 1]), bounds=bounds, grid_n=grid_n)
        if ij is None:
            continue
        i, j = ij
        visited[i, j] = 1

    corridor = dilate_manhattan(visited, radius=buffer_cells)
    obs = rasterize_rectangles_to_grid(rects, bounds=bounds, grid_n=grid_n)

    area = visited.copy()
    # only include obstacles that lie in the buffered corridor region
    area = np.maximum(area, (obs & corridor).astype(np.uint8))

    v = area.reshape(-1).astype(np.uint8)
    dbg = {
        "nominal_len": int(X.shape[0]),
        "visited_cells": int(np.sum(visited)),
        "corridor_cells": int(np.sum(corridor)),
        "obs_cells_total": int(np.sum(obs)),
        "area_cells": int(np.sum(area)),
    }
    return v, dbg


# =============================================================================
# 4) DB structures + serialization
# =============================================================================

@dataclass
class MazeInstance:
    map_idx: int
    rectangles: List[Rect]
    bounds: Rect
    start: Tuple[float, float]
    goal: Tuple[float, float]
    # the situation vector is stored per (dyn,map) in DynNode.env_types["maze"][map_idx]["situation_vec"]
    # so here MazeInstance only holds geometry.

@dataclass
class DynNode:
    dyn_id: int
    cluster_id: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    M: np.ndarray
    env_types: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsensusNode:
    cluster_id: int
    cd_dyn_id: int
    M_cd: np.ndarray
    dyn_children: Dict[int, int] = field(default_factory=dict)

@dataclass
class HierarchicalDB:
    consensus_nodes: Dict[int, ConsensusNode] = field(default_factory=dict)
    dyn_nodes: Dict[int, DynNode] = field(default_factory=dict)
    dyn_to_cluster: Dict[int, int] = field(default_factory=dict)


def serialize_db(
    db: HierarchicalDB,
    *,
    shared_maps: List[MazeInstance],
    per_dyn_situation_vecs: Dict[int, List[np.ndarray]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    We store shared_maps once, and store per-dyn situation vectors aligned by map_idx.

    JSON payload layout:
      payload = {
        "meta": {...grid params...},
        "shared_maps": [ {map_idx, rectangles, bounds, start, goal}, ... ],
        "db": {
          "consensus_nodes": {...},
          "dyn_nodes": {
            "0": {
               "A":..., "B":..., "env_types": {
                   "maze": {
                      "map_count": 64,
                      "situation_vecs": [ [0,1,0,...], ... ]   # list of lists length grid_n^2
                   }
               }
            }, ...
          }
        }
      }
    """
    out: Dict[str, Any] = {
        "meta": meta,
        "shared_maps": [],
        "db": {
            "consensus_nodes": {},
            "dyn_nodes": {},
            "dyn_to_cluster": {str(k): int(v) for k, v in db.dyn_to_cluster.items()},
        }
    }

    for mi in shared_maps:
        out["shared_maps"].append({
            "map_idx": int(mi.map_idx),
            "rectangles": [list(map(float, r)) for r in mi.rectangles],
            "bounds": list(map(float, mi.bounds)),
            "start": [float(mi.start[0]), float(mi.start[1])],
            "goal": [float(mi.goal[0]), float(mi.goal[1])],
        })

    for cid, cn in db.consensus_nodes.items():
        out["db"]["consensus_nodes"][str(cid)] = {
            "cluster_id": int(cn.cluster_id),
            "cd_dyn_id": int(cn.cd_dyn_id),
            "M_cd": cn.M_cd.tolist(),
            "dyn_children": [int(k) for k in cn.dyn_children.keys()],
        }

    for did, dn in db.dyn_nodes.items():
        vecs = per_dyn_situation_vecs[int(did)]
        out["db"]["dyn_nodes"][str(did)] = {
            "dyn_id": int(dn.dyn_id),
            "cluster_id": int(dn.cluster_id),
            "A": dn.A.tolist(),
            "B": dn.B.tolist(),
            "C": dn.C.tolist(),
            "M": dn.M.tolist(),
            "env_types": {
                "maze": {
                    "map_count": int(len(shared_maps)),
                    "situation_vecs": [v.astype(int).tolist() for v in vecs],
                }
            }
        }

    return out


# =============================================================================
# 5) Build MAZE DB (UPDATED SECTION 5: scatter-block maps + situation vectors)
# =============================================================================

def build_hierarchical_db_maze_scatter(
    *,
    clusters_A: List[List[np.ndarray]],
    clusters_M: List[List[np.ndarray]],
    B: np.ndarray,
    C: np.ndarray,
    # fixed world
    bounds: Rect = (-10.0, -10.0, 10.0, 10.0),
    start: Tuple[float, float] = (5.0, 5.0),
    goal: Tuple[float, float] = (0.0, 0.0),
    # maps
    n_maps: int = 64,
    seed_maps: int = 123,
    n_rect: int = 26,
    w_range: Tuple[float, float] = (0.7, 2.0),
    h_range: Tuple[float, float] = (0.7, 2.0),
    min_gap: float = 0.15,
    avoid_margin: float = 1.0,
    border: bool = False,
    # situation vector params
    grid_n: int = 25,
    dt_nom: float = 0.05,
    n_steps_nom: int = 400,
    u_max_nom: float = 3.0,
    buffer_cells: int = 2,
    stop_tol: float = 0.6,
) -> Tuple[HierarchicalDB, List[MazeInstance], Dict[int, List[np.ndarray]], Dict[str, Any]]:
    """
    Returns:
      db
      shared_maps: List[MazeInstance] length n_maps
      per_dyn_situation_vecs: did -> list of vectors aligned with map_idx
      info/meta
    """
    t0 = time.time()
    rng = np.random.default_rng(seed_maps)
    random.seed(seed_maps)

    # 1) shared map bank
    shared_maps: List[MazeInstance] = []
    for k in range(int(n_maps)):
        rects = generate_scatter_rectangles(
            seed=int(seed_maps) + 999 * k,
            bounds=bounds,
            n_rect=int(n_rect),
            w_range=w_range,
            h_range=h_range,
            min_gap=float(min_gap),
            fixed_points=[start, goal],
            avoid_margin=float(avoid_margin),
            border=bool(border),
        )
        shared_maps.append(MazeInstance(
            map_idx=int(k),
            rectangles=[tuple(map(float, r)) for r in rects],
            bounds=bounds,
            start=start,
            goal=goal,
        ))

    # 2) hierarchical DB nodes
    db = HierarchicalDB()
    dyn_id_global = 0

    # 3) compute per-dyn situation vecs (nominal depends on A,B)
    per_dyn_situation_vecs: Dict[int, List[np.ndarray]] = {}

    for cluster_id, (cluster_As, cluster_Ms) in enumerate(zip(clusters_A, clusters_M)):
        if len(cluster_As) == 0:
            raise ValueError(f"Cluster {cluster_id} empty.")

        cd_dyn_id = dyn_id_global
        db.consensus_nodes[int(cluster_id)] = ConsensusNode(
            cluster_id=int(cluster_id),
            cd_dyn_id=int(cd_dyn_id),
            M_cd=cluster_Ms[0],
            dyn_children={},
        )

        for local_idx, (A, M) in enumerate(zip(cluster_As, cluster_Ms)):
            did = int(dyn_id_global)

            node = DynNode(
                dyn_id=did,
                cluster_id=int(cluster_id),
                A=np.array(A, dtype=float),
                B=np.array(B, dtype=float),
                C=np.array(C, dtype=float),
                M=np.array(M, dtype=float),
                env_types={"maze": {"map_count": int(n_maps)}},
            )
            db.dyn_nodes[did] = node
            db.dyn_to_cluster[did] = int(cluster_id)
            db.consensus_nodes[int(cluster_id)].dyn_children[did] = did

            # per-map vectors for this dyn
            vecs: List[np.ndarray] = []
            for mi in shared_maps:
                v, _dbg = situation_vector(
                    A=node.A,
                    B=node.B,
                    rects=mi.rectangles,
                    bounds=bounds,
                    start=start,
                    goal=goal,
                    grid_n=grid_n,
                    dt_nom=dt_nom,
                    n_steps_nom=n_steps_nom,
                    u_max_nom=u_max_nom,
                    buffer_cells=buffer_cells,
                    stop_tol=stop_tol,
                )
                vecs.append(v)
            per_dyn_situation_vecs[did] = vecs

            dyn_id_global += 1

    t1 = time.time()
    meta = {
        "build_wall_time_sec": float(t1 - t0),
        "bounds": list(map(float, bounds)),
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "maps": {
            "n_maps": int(n_maps),
            "n_rect": int(n_rect),
            "w_range": [float(w_range[0]), float(w_range[1])],
            "h_range": [float(h_range[0]), float(h_range[1])],
            "min_gap": float(min_gap),
            "avoid_margin": float(avoid_margin),
            "border": bool(border),
        },
        "situation_vector": {
            "grid_n": int(grid_n),
            "vector_dim": int(grid_n * grid_n),
            "dt_nom": float(dt_nom),
            "n_steps_nom": int(n_steps_nom),
            "u_max_nom": float(u_max_nom),
            "buffer_cells": int(buffer_cells),
            "stop_tol": float(stop_tol),
            "labeling": "bottom->top, left->right; index = i*grid_n + j",
        },
        "counts": {
            "n_clusters": int(len(clusters_A)),
            "n_dyn_total": int(sum(len(c) for c in clusters_A)),
            "n_maps_shared": int(n_maps),
        },
    }
    return db, shared_maps, per_dyn_situation_vecs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", type=str, default="maze/S1_database_maze.json")

    ap.add_argument("--K_consensus", type=int, default=5)
    ap.add_argument("--n_per_consensus", type=int, default=10)
    ap.add_argument("--seed_dyn", type=int, default=1)
    ap.add_argument("--alpha_deg", type=float, default=5.0)
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--solver", type=str, default="SCS")

    # shared maps
    ap.add_argument("--n_maps", type=int, default=64)
    ap.add_argument("--seed_maps", type=int, default=123)
    ap.add_argument("--n_rect", type=int, default=26)
    ap.add_argument("--w_min", type=float, default=0.7)
    ap.add_argument("--w_max", type=float, default=2.0)
    ap.add_argument("--h_min", type=float, default=0.7)
    ap.add_argument("--h_max", type=float, default=2.0)
    ap.add_argument("--min_gap", type=float, default=0.15)
    ap.add_argument("--avoid_margin", type=float, default=1.0)
    ap.add_argument("--border", action="store_true")

    # situation vector
    ap.add_argument("--grid_n", type=int, default=25)
    ap.add_argument("--dt_nom", type=float, default=0.05)
    ap.add_argument("--n_steps_nom", type=int, default=400)
    ap.add_argument("--u_max_nom", type=float, default=3.0)
    ap.add_argument("--buffer_cells", type=int, default=2)
    ap.add_argument("--stop_tol", type=float, default=0.6)

    args = ap.parse_args()

    B = np.eye(2)
    C = np.eye(2)

    clusters_A, clusters_M, _consensus_As, meta_sampling = generate_consensus_then_clustered_dynamics(
        K_consensus=args.K_consensus,
        n_per_consensus=args.n_per_consensus,
        seed=args.seed_dyn,
        alpha_within=np.deg2rad(args.alpha_deg),
        omega0=args.omega0,
        solver=args.solver,
        B=B,
        C=C,
        verbose=False,
    )

    db, shared_maps, per_dyn_vecs, meta = build_hierarchical_db_maze_scatter(
        clusters_A=clusters_A,
        clusters_M=clusters_M,
        B=B,
        C=C,
        bounds=(-10.0, -10.0, 10.0, 10.0),
        start=(5.0, 5.0),
        goal=(0.0, 0.0),
        n_maps=args.n_maps,
        seed_maps=args.seed_maps,
        n_rect=args.n_rect,
        w_range=(args.w_min, args.w_max),
        h_range=(args.h_min, args.h_max),
        min_gap=args.min_gap,
        avoid_margin=args.avoid_margin,
        border=bool(args.border),
        grid_n=args.grid_n,
        dt_nom=args.dt_nom,
        n_steps_nom=args.n_steps_nom,
        u_max_nom=args.u_max_nom,
        buffer_cells=args.buffer_cells,
        stop_tol=args.stop_tol,
    )

    payload = serialize_db(
        db=db,
        shared_maps=shared_maps,
        per_dyn_situation_vecs=per_dyn_vecs,
        meta={"meta_sampling": meta_sampling, "meta_db": meta},
    )

    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"[ok] wrote DB JSON: {args.out_json}")
    print("[info] dyn_total =", meta["counts"]["n_dyn_total"], "| maps =", meta["counts"]["n_maps_shared"])


if __name__ == "__main__":
    main()
