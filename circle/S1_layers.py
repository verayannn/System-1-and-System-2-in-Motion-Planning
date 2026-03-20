"""
S1_layers.py — Generate System-1 hierarchical layers

Hierarchy (tree):
  ConsensusDyn (cluster_id)
    -> Dyn (dyn_id; global id)
       -> ObsType (currently you can keep only ["car"])
          -> ObsInstance (center locations)

End-to-end:
1) Generate K consensus continuous-time stable dynamics A (Hurwitz).
2) For each consensus A0, sample (n_per_consensus-1) perturbed stable A's such that
   the induced complex "M-map" matrices within that cluster are simultaneously alpha-alignable
   (checked via a CVXPY feasibility problem).
3) Build a hierarchical DB (Consensus -> Dyn -> ObsType -> ObsInstance).
4) Save DB + metadata to JSON.

Dependencies:
- numpy
- cvxpy
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cvxpy as cp


# =============================================================================
# 0) Core helpers for alignment feasibility (CVXPY)
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
    """
    Check simultaneous alpha-alignment feasibility for matrices M_i by solving:

      Find complex K s.t. for each i:
        Re(M_i K) - (M_i M_i^*)   ⪰ 0
        |Im(M_i K)|              ⪯ tan(alpha) * Re(M_i K)

    Returns:
      feasible (bool), K (np.ndarray or None)
    """
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
    """
    REAL-valued surrogate of frequency response (for S1).
    """
    n = A.shape[0]
    jwI_minus_A = 1j * omega0 * np.eye(n) - A
    M_complex = C @ np.linalg.inv(jwI_minus_A) @ B
    return np.real(M_complex)



# =============================================================================
# 1) Stable dynamics sampling utilities
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
    """
    Sample a continuous-time stable A (Hurwitz).
    """
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


def _perturb_A(A0: np.ndarray, rng: np.random.Generator, scale: float = 0.10) -> np.ndarray:
    """
    Small perturbation around A0, enforce Hurwitz by shifting if needed.
    """
    m = A0.shape[0]
    dA = rng.standard_normal((m, m))
    dA = _normalize(dA) * float(rng.uniform(0.0, scale))
    A = A0 + dA

    if not _is_hurwitz(A):
        lam = np.linalg.eigvals(A)
        shift = max(0.0, float(np.max(np.real(lam)) + 1e-3))
        A = A - shift * np.eye(m)

    return A


# =============================================================================
# 2) Generate clustered dynamics constrained by alpha-alignability
# =============================================================================

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
    """
    Returns:
      clusters_A: list[K] of list[n_per] of A matrices
      clusters_M: list[K] of list[n_per] of M matrices
      consensus_As: list[K] (each is the first element of each cluster)
      meta: dict of parameters and sampling stats
    """
    if K_consensus <= 0 or n_per_consensus <= 0:
        raise ValueError("K_consensus and n_per_consensus must be positive integers.")

    rng = np.random.default_rng(seed)
    phase_centers = np.deg2rad(np.array(phase_centers_deg, dtype=float))

    # Default B,C for 2D systems (override if needed)
    if B is None:
        B = np.array([[0.0],
                      [1.0]])
    if C is None:
        C = np.array([[1.0, 0.0]])

    # Step 1: sample consensus A's
    consensus_As: List[np.ndarray] = []
    consensus_thetas: List[float] = []
    for _ in range(K_consensus):
        theta = float(rng.choice(phase_centers))
        A0, th = _make_stable_A(m=m, rng=rng, theta=theta)
        consensus_As.append(A0)
        consensus_thetas.append(float(th))

    # Step 2: grow each cluster while preserving alpha-alignability of M's
    clusters_A: List[List[np.ndarray]] = []
    clusters_M: List[List[np.ndarray]] = []
    tries_used: List[List[int]] = []

    for c_idx, A0 in enumerate(consensus_As):
        M0 = system_to_M(A0, B, C, omega0=omega0)

        cluster_A = [A0]
        cluster_M = [M0]
        cluster_tries = [0]  # consensus item uses 0 tries

        while len(cluster_A) < n_per_consensus:
            accepted = False

            for t in range(1, max_tries_per_member + 1):
                A_cand = _perturb_A(A0, rng=rng, scale=perturb_scale)
                M_cand = system_to_M(A_cand, B, C, omega0=omega0)

                feasible, _ = is_alignable(
                    cluster_M + [M_cand],
                    alpha_within,
                    solver=solver,
                    verbose=verbose,
                )
                if feasible:
                    cluster_A.append(A_cand)
                    cluster_M.append(M_cand)
                    cluster_tries.append(t)
                    accepted = True
                    break

            if not accepted:
                raise RuntimeError(
                    f"[Cluster {c_idx}] Failed to sample an alpha-alignable member.\n"
                    f"Suggestions: increase max_tries_per_member, increase alpha_within, "
                    f"or reduce perturb_scale."
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
        "phase_centers_deg": phase_centers_deg,
        "B": B.tolist(),
        "C": C.tolist(),
        "tries_used": tries_used,
    }
    return clusters_A, clusters_M, consensus_As, meta


# =============================================================================
# 3) S1 hierarchy data structures
# =============================================================================

@dataclass
class ObsInstance:
    center: Tuple[float, float]        # (cx, cy) in continuous coordinates
    timestamp: float = 0.0
    traj_ref: Optional[Any] = None     # later: index/id to stored trajectory


@dataclass
class ObsType:
    name: str
    dyn_id: int
    instances: List[ObsInstance] = field(default_factory=list)


@dataclass
class Dyn:
    dyn_id: int                         # global dyn id (index into M_list)
    cluster_id: int
    M: np.ndarray
    obs_types: Dict[str, ObsType] = field(default_factory=dict)


@dataclass
class ConsensusDyn:
    cluster_id: int
    cd_dyn_id: int                      # global dyn id for consensus representative
    M_cd: np.ndarray
    dyn_children: Dict[int, Dyn] = field(default_factory=dict)  # dyn_id -> Dyn


@dataclass
class HierarchicalDB:
    consensus_nodes: Dict[int, ConsensusDyn] = field(default_factory=dict)  # cluster_id -> ConsensusDyn
    dyn_to_cluster: Dict[int, int] = field(default_factory=dict)            # dyn_id -> cluster_id
    dyn_nodes: Dict[int, Dyn] = field(default_factory=dict)                 # dyn_id -> Dyn


# =============================================================================
# 4) Uniform obstacle centers (by quadrant grid)
# =============================================================================

def generate_uniform_obstacle_centers(
    *,
    grid_height: float = 20,
    grid_width: float = 20,
    grids_per_quadrant: int = 3,
) -> List[Tuple[float, float]]:
    """
    Evenly spaced centers in a plane centered at origin.

    Positive half-extent in x1 is 2*grid_height, and in x2 is 2*grid_width.
    4 quadrants × (grids_per_quadrant^2) cells each → 4*grids_per_quadrant^2 centers.
    """
    half_h = 2.0 * float(grid_height)
    half_w = 2.0 * float(grid_width)

    x_pos = (np.arange(grids_per_quadrant) + 0.5) * (half_h / grids_per_quadrant)
    y_pos = (np.arange(grids_per_quadrant) + 0.5) * (half_w / grids_per_quadrant)

    centers: List[Tuple[float, float]] = []
    quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    for sx, sy in quadrants:
        for cx in x_pos:
            for cy in y_pos:
                centers.append((float(sx * cx), float(sy * cy)))

    return centers


# =============================================================================
# 5) Build DB for single-obstacle case (UPDATED: takes clusters_M directly)
# =============================================================================

def build_hierarchical_db_single_obstacle(
    *,
    clusters_M: List[List[np.ndarray]],
    obstacle_type_names: Optional[List[str]] = None,
    grid_height: float = 5,
    grid_width: float = 5,
    grids_per_quadrant: int = 3,
    num_positions_per_type: Optional[int] = 36,
    seed: int = 0,
    shuffle_centers_per_type: bool = False,
) -> Tuple[HierarchicalDB, Dict[str, Any]]:
    """
    Build S1 DB for single-obstacle case using uniformly spaced positions.

    Inputs:
      - clusters_M: list of clusters, each is a list of M matrices (cluster member dynamics)
                   clusters_M[cluster_id][local_idx] = M

    We create global dyn_id by flattening clusters in order:
      dyn_id = 0..(K*n_per - 1)

    Returns:
      db: HierarchicalDB
      db_info: metadata (counts + build time)
    """
    t0 = time.time()
    random.seed(seed)

    if obstacle_type_names is None:
        obstacle_type_names = ["car"]  # you said only car for now

    centers = generate_uniform_obstacle_centers(
        grid_height=grid_height,
        grid_width=grid_width,
        grids_per_quadrant=grids_per_quadrant,
    )
    total_centers = len(centers)
    if num_positions_per_type is None:
        num_positions_per_type = total_centers
    num_positions_per_type = min(int(num_positions_per_type), total_centers)

    db = HierarchicalDB()

    dyn_id = 0
    n_clusters = len(clusters_M)
    n_dyn_total = 0

    for cluster_id, cluster_Ms in enumerate(clusters_M):
        if len(cluster_Ms) == 0:
            raise ValueError(f"Cluster {cluster_id} is empty.")

        # consensus representative is local 0 by construction
        cd_dyn_id = dyn_id  # because first member we add in this cluster will get this dyn_id
        cd_node = ConsensusDyn(
            cluster_id=int(cluster_id),
            cd_dyn_id=int(cd_dyn_id),
            M_cd=cluster_Ms[0]
        )
        db.consensus_nodes[int(cluster_id)] = cd_node

        for local_idx, M in enumerate(cluster_Ms):
            dyn_node = Dyn(
                dyn_id=int(dyn_id),
                cluster_id=int(cluster_id),
                M=M,
            )

            for name in obstacle_type_names:
                obs_type = ObsType(name=name, dyn_id=int(dyn_id))

                selected = centers[:]
                if shuffle_centers_per_type:
                    random.shuffle(selected)
                selected = selected[:num_positions_per_type]

                for (cx, cy) in selected:
                    obs_type.instances.append(ObsInstance(center=(cx, cy)))

                dyn_node.obs_types[name] = obs_type

            cd_node.dyn_children[int(dyn_id)] = dyn_node
            db.dyn_nodes[int(dyn_id)] = dyn_node
            db.dyn_to_cluster[int(dyn_id)] = int(cluster_id)

            dyn_id += 1
            n_dyn_total += 1

    t1 = time.time()

    db_info = {
        "seed": seed,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "grids_per_quadrant": grids_per_quadrant,
        "total_centers_available": total_centers,
        "num_positions_per_type": num_positions_per_type,
        "obstacle_type_names": obstacle_type_names,
        "counts": {
            "n_consensus": n_clusters,
            "n_dyn_total": n_dyn_total,
            "n_types_total": n_dyn_total * len(obstacle_type_names),
            "n_instances_total": n_dyn_total * len(obstacle_type_names) * num_positions_per_type,
            "max_leaves": n_dyn_total * len(obstacle_type_names) * num_positions_per_type,
        },
        "build_wall_time_sec": float(t1 - t0),
    }

    return db, db_info


# =============================================================================
# 6) Materialize a concrete obstacle from DB
# =============================================================================

OBST_RADIUS_FACTORS: Dict[str, float] = {
    "car": 1.0,
    "wall": 1.5,
    "static_cone": 0.5,
    "pedestrian": 0.8,
    "truck": 1.8,
}


def build_single_obstacle_from_db(
    *,
    db: HierarchicalDB,
    dyn_id: int,
    obs_type_name: str = "car",
    instance_idx: int = 0,
    base_cell_radius: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Build ONE circular obstacle for a given dyn_id, obstacle type, and instance index.
    """
    if dyn_id not in db.dyn_nodes:
        raise KeyError(f"dyn_id={dyn_id} not in db.")

    dyn_node = db.dyn_nodes[dyn_id]
    if obs_type_name not in dyn_node.obs_types:
        raise ValueError(f"Obstacle type '{obs_type_name}' not found for dyn_id {dyn_id}")

    obs_type = dyn_node.obs_types[obs_type_name]
    if not (0 <= instance_idx < len(obs_type.instances)):
        raise IndexError(f"instance_idx {instance_idx} out of range for type '{obs_type_name}'")

    cx, cy = obs_type.instances[instance_idx].center
    factor = OBST_RADIUS_FACTORS.get(obs_type_name, 1.0)
    radius = float(base_cell_radius) * float(factor)

    return [{"center": (cx, cy), "radius": radius, "type": obs_type_name}]

# =============================================================================
# 7) Clean “one config → generate DB” wrapper (UPDATED: also output A_list for MPC)
# =============================================================================

@dataclass
class LayerGenConfig:
    # Dynamics generation
    K_consensus: int = 5
    n_per_consensus: int = 10
    m: int = 2
    seed: int = 1
    alpha_within_deg: float = 5.0
    omega0: float = 1.0
    solver: str = "SCS"
    verbose: bool = False
    max_tries_per_member: int = 500
    perturb_scale: float = 0.10
    phase_centers_deg: Tuple[float, ...] = (0, 25, -25, 50, -50, 75)

    # System-to-M map matrices
    B: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None

    # DB + obstacles
    obstacle_type_names: List[str] = field(default_factory=lambda: ["car"])
    grid_height: float = 5
    grid_width: float = 5
    grids_per_quadrant: int = 3
    num_positions_per_type: Optional[int] = 36
    db_seed: int = 42
    shuffle_centers_per_type: bool = False


def generate_layers(cfg: LayerGenConfig) -> Tuple[HierarchicalDB, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      db: HierarchicalDB
      out: dict containing clusters + meta + (NEW) A_list for MPC
      db_info: DB structure + build time
    """
    alpha_within = float(np.deg2rad(cfg.alpha_within_deg))

    clusters_A, clusters_M, consensus_As, dynamics_meta = generate_consensus_then_clustered_dynamics(
        K_consensus=cfg.K_consensus,
        n_per_consensus=cfg.n_per_consensus,
        m=cfg.m,
        seed=cfg.seed,
        alpha_within=alpha_within,
        omega0=cfg.omega0,
        solver=cfg.solver,
        verbose=cfg.verbose,
        max_tries_per_member=cfg.max_tries_per_member,
        perturb_scale=cfg.perturb_scale,
        phase_centers_deg=cfg.phase_centers_deg,
        B=cfg.B,
        C=cfg.C,
    )

    # --- NEW: flatten A's into a dyn_id-aligned list (dyn_id = 0..N-1) ---
    A_list: List[np.ndarray] = []
    for c in range(len(clusters_A)):
        for j in range(len(clusters_A[c])):
            A_list.append(np.array(clusters_A[c][j], dtype=float))

    db, db_info = build_hierarchical_db_single_obstacle(
        clusters_M=clusters_M,
        obstacle_type_names=cfg.obstacle_type_names,
        grid_height=cfg.grid_height,
        grid_width=cfg.grid_width,
        grids_per_quadrant=cfg.grids_per_quadrant,
        num_positions_per_type=cfg.num_positions_per_type,
        seed=cfg.db_seed,
        shuffle_centers_per_type=cfg.shuffle_centers_per_type,
    )

    # Pack what you'll likely want later
    out = {
        "clusters_A": clusters_A,
        "clusters_M": clusters_M,
        "consensus_As": consensus_As,
        "dynamics_meta": dynamics_meta,
        "A_list": A_list,  # <-- NEW (for MPC)
        "cfg": {
            **{k: v for k, v in cfg.__dict__.items() if k not in ("B", "C")},
            "B": None if cfg.B is None else cfg.B.tolist(),
            "C": None if cfg.C is None else cfg.C.tolist(),
        },
    }
    return db, out, db_info


# =============================================================================
# 8) Save DB to JSON (UPDATED: include A per dyn for MPC)
# =============================================================================

def hierarchical_db_to_dict(db: HierarchicalDB, A_list: List[np.ndarray]) -> Dict[str, Any]:
    """
    Convert HierarchicalDB into a JSON-serializable dictionary.
    Includes:
      - A (real 2x2) per dyn_id for MPC
      - M (your S1 matrix)
      - obstacle instances
    """
    consensus_nodes: Dict[str, Any] = {}
    for cid, cd in db.consensus_nodes.items():
        consensus_nodes[str(cid)] = {
            "cluster_id": int(cd.cluster_id),
            "cd_dyn_id": int(cd.cd_dyn_id),
            "M_cd": cd.M_cd.tolist(),
            "dyn_children": list(map(int, cd.dyn_children.keys())),
        }

    dyn_nodes: Dict[str, Any] = {}
    for dyn_id, dyn in db.dyn_nodes.items():
        dyn_id = int(dyn_id)
        if dyn_id >= len(A_list):
            raise IndexError(
                f"A_list length {len(A_list)} < dyn_id {dyn_id}. "
                "Make sure A_list is flattened in the same dyn_id order as DB."
            )

        A = np.array(A_list[dyn_id], dtype=float)

        dyn_nodes[str(dyn_id)] = {
            "dyn_id": int(dyn.dyn_id),
            "cluster_id": int(dyn.cluster_id),

            # --- NEW: dynamics matrix for MPC ---
            "A": A.tolist(),

            # --- Existing S1 representation ---
            "M": dyn.M.tolist(),

            # --- Obstacle instances ---
            "obs_types": {
                name: [
                    {
                        "center": [float(inst.center[0]), float(inst.center[1])],
                        "timestamp": float(inst.timestamp),
                        "traj_ref": inst.traj_ref,
                    }
                    for inst in obs.instances
                ]
                for name, obs in dyn.obs_types.items()
            },
        }

    return {
        "consensus_nodes": consensus_nodes,
        "dyn_to_cluster": {str(k): int(v) for k, v in db.dyn_to_cluster.items()},
        "dyn_nodes": dyn_nodes,
    }


def save_hierarchical_db_to_json(
    db: HierarchicalDB,
    db_info: Dict[str, Any],
    dynamics_info: Dict[str, Any],
    A_list: List[np.ndarray],
    filepath: str,
) -> None:
    """
    Save HierarchicalDB + db_info + dynamics_info + A_list into a single JSON file.
    """
    payload = {
        "db_info": db_info,
        "dynamics_info": dynamics_info,
        "db": hierarchical_db_to_dict(db, A_list=A_list),
    }
    path = Path(filepath)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# 9) Demo runner (UPDATED: pass A_list to saver)
# =============================================================================

def _print_summary(db: HierarchicalDB, out: Dict[str, Any], db_info: Dict[str, Any]) -> None:
    print("\n=== GENERATED LAYERS SUMMARY ===")
    print(f"Clusters (consensus nodes): {len(db.consensus_nodes)}")

    any_cluster = next(iter(db.consensus_nodes.values()))
    print(f"Dynamics per cluster: {len(any_cluster.dyn_children)}")

    any_dyn = next(iter(any_cluster.dyn_children.values()))
    print(f"Obstacle types per dyn: {len(any_dyn.obs_types)}")

    any_type = next(iter(any_dyn.obs_types.values()))
    print(f"Instances per (dyn,type): {len(any_type.instances)}")

    print(f"Within-cluster alpha: {out['dynamics_meta']['alpha_within_deg']:.2f} deg")
    print(f"omega0 (M-map): {out['dynamics_meta']['omega0']}")
    print(f"DB build time: {db_info['build_wall_time_sec']:.4f} sec")
    print(f"Max leaves: {db_info['counts']['max_leaves']}")

    # NEW: verify A_list alignment
    if "A_list" in out:
        print(f"A_list length: {len(out['A_list'])} (should equal #dyn={len(db.dyn_nodes)})\n")
    else:
        print("A_list missing in out (MPC will not run).\n")


if __name__ == "__main__":
    cfg = LayerGenConfig(
        K_consensus=5, ## 2
        n_per_consensus=10, ## 5
        alpha_within_deg=5.0,
        omega0=1.0,
        solver="SCS",
        verbose=False,
        perturb_scale=0.10,
        max_tries_per_member=500,
        obstacle_type_names=["car"],
        grid_height=5,
        grid_width=5,
        grids_per_quadrant=3,            # 3x3 per quadrant -> 36 total centers
        num_positions_per_type=36,
        db_seed=42,
    )

    db, out, db_info = generate_layers(cfg)
    _print_summary(db, out, db_info)

    # Example: materialize one obstacle (dyn_id 0, first instance)
    obstacle = build_single_obstacle_from_db(
        db=db,
        dyn_id=0,
        obs_type_name="car",
        instance_idx=0,
        base_cell_radius=0.5,
    )
    print("Example obstacle:", obstacle)

    # Save DB (UPDATED: include A_list for MPC)
    save_hierarchical_db_to_json(
        db=db,
        db_info=db_info,
        dynamics_info=out["dynamics_meta"],
        A_list=out["A_list"],
        filepath="S1_database_single_obstacle.json",
    )
    print("Saved:", "S1_database_single_obstacle.json")
