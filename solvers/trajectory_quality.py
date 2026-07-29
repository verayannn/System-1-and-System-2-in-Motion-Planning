
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

DURATION_INVARIANT_QUALITY_VERSION = "duration_invariant_v1"
QUALITY_DEFINITION_VERSION = DURATION_INVARIANT_QUALITY_VERSION

# Evaluation body radius. Held common across solvers so the clearance score
# measures the executed trajectory, not each solver's own inflation setting.
QUALITY_BODY_RADIUS = 0.25
QUALITY_REFERENCE_PATH_RESOLUTION = 0.10
# The reference path must lower-bound what any solver could achieve, so it is
# planned against the bare obstacles rather than against an inflated copy.
# Inflating it by the body radius makes it longer than the executed paths and
# the efficiency ratio then saturates at 1.0 for every solver.
QUALITY_REFERENCE_PATH_CLEARANCE = 0.02

# Dimensionless jerk of an ideal minimum-jerk point-to-point movement:
# integral of squared jerk is 720 L^2 / T^5 and peak speed is 1.875 L / T, so
# DLJ = T^3 / v_peak^2 * integral = 720 / 1.875^2.
MIN_JERK_DLJ = 720.0 / (1.875 ** 2)
MIN_JERK_LDLJ = -math.log(MIN_JERK_DLJ)

# Spectral arc length of an ideal minimum-jerk speed profile. Computed once at
# import so the reference tracks the same estimator used on real data.
_SPARC_AMP_THRESHOLD = 0.05
_SPARC_MAX_CUTOFF_HZ = 10.0


def _as_xy(states: Any) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("states must be an (N>=2, >=2) array")
    return arr[:, :2]


def speed_profile(xy: np.ndarray, dt: float) -> np.ndarray:
    """Return the piecewise speed implied by the executed sample sequence."""
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / float(dt)


def log_dimensionless_jerk(speed: np.ndarray, dt: float) -> float:
    """Log dimensionless jerk of a speed profile (larger, i.e. less negative, is smoother)."""
    speed = np.asarray(speed, dtype=float)
    if speed.size < 3:
        return float("nan")
    peak = float(np.max(np.abs(speed)))
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan")
    duration = float(speed.size * dt)
    jerk = np.diff(speed, n=2) / (float(dt) ** 2)
    integral = float(np.sum(jerk ** 2) * dt)
    if integral <= 0.0:
        # A perfectly constant speed profile has zero jerk; treat as ideal.
        return MIN_JERK_LDLJ
    scale = (duration ** 3) / (peak ** 2)
    return float(-np.log(scale * integral))


def spectral_arc_length(
    speed: np.ndarray,
    dt: float,
    *,
    pad_level: int = 4,
    amp_threshold: float = _SPARC_AMP_THRESHOLD,
    max_cutoff_hz: float = _SPARC_MAX_CUTOFF_HZ,
) -> float:
    """Spectral arc length (SPARC) of a speed profile.

    Follows Balasubramanian et al. (2015). The cutoff is additionally capped at
    the Nyquist frequency because these trajectories are sampled far more
    coarsely than the motion-capture data the metric was designed for, and the
    mirrored spectrum above Nyquist would otherwise be counted twice.
    """
    speed = np.asarray(speed, dtype=float)
    if speed.size < 4:
        return float("nan")
    fs = 1.0 / float(dt)
    peak = float(np.max(np.abs(speed)))
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan")

    nfft = int(2 ** (math.ceil(math.log2(speed.size)) + pad_level))
    freq = np.arange(0, fs, fs / nfft)
    spectrum = np.abs(np.fft.fft(speed, nfft))
    spectrum = spectrum / np.max(spectrum)

    cutoff = min(float(max_cutoff_hz), 0.5 * fs)
    keep = freq <= cutoff
    if keep.sum() < 3:
        return float("nan")
    f_sel = freq[keep]
    m_sel = spectrum[keep]

    above = np.nonzero(m_sel >= amp_threshold)[0]
    if above.size < 2:
        return float("nan")
    f_sel = f_sel[above[0] : above[-1] + 1]
    m_sel = m_sel[above[0] : above[-1] + 1]
    span = float(f_sel[-1] - f_sel[0])
    if span <= 0.0 or f_sel.size < 2:
        return float("nan")

    return float(-np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(m_sel) ** 2)))


def _min_jerk_sparc_reference(duration: float = 3.0, dt: float = 0.075) -> float:
    tau = np.linspace(0.0, 1.0, int(round(duration / dt)) + 1)
    speed = (30.0 / duration) * (tau ** 2) * ((1.0 - tau) ** 2)
    return spectral_arc_length(speed, dt)


MIN_JERK_SPARC = _min_jerk_sparc_reference()


def rect_clearance(xy: np.ndarray, rects: Sequence[Sequence[float]]) -> np.ndarray:
    """Distance from each sample to the nearest axis-aligned rectangle surface."""
    xy = np.asarray(xy, dtype=float)
    if not rects:
        return np.full(xy.shape[0], np.inf)
    best = np.full(xy.shape[0], np.inf)
    for rect in rects:
        x1, y1, x2, y2 = (float(v) for v in rect[:4])
        lo_x, hi_x = min(x1, x2), max(x1, x2)
        lo_y, hi_y = min(y1, y2), max(y1, y2)
        dx = np.maximum(np.maximum(lo_x - xy[:, 0], xy[:, 0] - hi_x), 0.0)
        dy = np.maximum(np.maximum(lo_y - xy[:, 1], xy[:, 1] - hi_y), 0.0)
        best = np.minimum(best, np.hypot(dx, dy))
    return best


def _nearest_free(blocked: np.ndarray, row: int, col: int) -> Optional[Tuple[int, int]]:
    rows, cols = blocked.shape
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    if not blocked[row, col]:
        return row, col
    for radius in range(1, max(rows, cols)):
        r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
        c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
        window = blocked[r0:r1, c0:c1]
        free = np.argwhere(~window)
        if free.size:
            dr, dc = free[np.argmin(np.abs(free[:, 0] + r0 - row) + np.abs(free[:, 1] + c0 - col))]
            return int(dr + r0), int(dc + c0)
    return None


def shortest_free_path_length(
    start: Sequence[float],
    goal: Sequence[float],
    rects: Sequence[Sequence[float]],
    bounds: Sequence[float],
    *,
    clearance: float = QUALITY_REFERENCE_PATH_CLEARANCE,
    resolution: float = QUALITY_REFERENCE_PATH_RESOLUTION,
) -> Optional[float]:
    """Length of the shortest 8-connected collision-free path, or None if blocked."""
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    resolution = max(float(resolution), 0.05)
    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    if xs.size < 2 or ys.size < 2:
        return None
    xx, yy = np.meshgrid(xs, ys)
    blocked = np.zeros_like(xx, dtype=bool)
    for rect in rects:
        x1, y1, x2, y2 = (float(v) for v in rect[:4])
        blocked |= (
            (xx >= min(x1, x2) - clearance)
            & (xx <= max(x1, x2) + clearance)
            & (yy >= min(y1, y2) - clearance)
            & (yy <= max(y1, y2) + clearance)
        )

    start_cell = _nearest_free(blocked, int(np.argmin(np.abs(ys - start[1]))), int(np.argmin(np.abs(xs - start[0]))))
    goal_cell = _nearest_free(blocked, int(np.argmin(np.abs(ys - goal[1]))), int(np.argmin(np.abs(xs - goal[0]))))
    if start_cell is None or goal_cell is None:
        return None

    rows, cols = blocked.shape
    dist = np.full((rows, cols), np.inf)
    dist[start_cell] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start_cell[0], start_cell[1])]
    steps = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    while queue:
        cost, row, col = heapq.heappop(queue)
        if cost > dist[row, col]:
            continue
        if (row, col) == goal_cell:
            return float(cost * resolution)
        for dr, dc, weight in steps:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                continue
            candidate = cost + weight
            if candidate < dist[nr, nc]:
                dist[nr, nc] = candidate
                heapq.heappush(queue, (candidate, nr, nc))
    return None


def evaluate_trajectory(
    states: Any,
    dt: float,
    scenario: Dict[str, Any],
    *,
    inputs: Any = None,
    reference_path_length: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Score one trajectory on duration-invariant path quality."""
    try:
        xy = _as_xy(states)
    except ValueError:
        return None
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        return None

    control = None
    if inputs is not None:
        candidate = np.asarray(inputs, dtype=float)
        if candidate.ndim == 2 and candidate.shape[1] >= 2:
            control = candidate[:, :2]
    # A synthetic goal-patch state is appended without a matching control.
    if control is not None and xy.shape[0] > control.shape[0] + 1:
        xy = xy[: control.shape[0] + 1]
    if xy.shape[0] < 3:
        return None

    start = np.asarray(scenario.get("start", xy[0]), dtype=float).reshape(-1)[:2]
    goal = np.asarray(scenario.get("goal", xy[-1]), dtype=float).reshape(-1)[:2]
    rects = scenario.get("rectangles") or []
    bounds = scenario.get("bounds") or [-10.0, -10.0, 10.0, 10.0]
    u_max = float(scenario.get("u_max", 3.0))

    path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if path_length <= 1e-9:
        return None

    if reference_path_length is None:
        reference_path_length = shortest_free_path_length(start, goal, rects, bounds)
    straight_line = float(np.linalg.norm(goal - start))
    if reference_path_length is None or not np.isfinite(reference_path_length):
        reference_path_length = straight_line
    # Grid snapping can shave up to half a cell off each endpoint; the straight
    # line is always a valid lower bound on achievable length.
    reference_path_length = max(float(reference_path_length), straight_line)
    path_efficiency = float(np.clip(reference_path_length / path_length, 0.0, 1.0))

    speed = speed_profile(xy, dt)
    ldlj = log_dimensionless_jerk(speed, dt)
    sparc = spectral_arc_length(speed, dt)
    if np.isfinite(sparc) and sparc < 0.0:
        smoothness = float(np.clip(MIN_JERK_SPARC / sparc, 0.0, 1.0))
    else:
        smoothness = float("nan")
    if np.isfinite(ldlj):
        # exp(LDLJ - LDLJ_minjerk) is exactly DLJ_minjerk / DLJ_actual.
        smoothness_ldlj = float(np.clip(math.exp(ldlj - MIN_JERK_LDLJ), 0.0, 1.0))
    else:
        smoothness_ldlj = float("nan")

    clearances = rect_clearance(xy, rects)
    min_clearance = float(np.min(clearances)) if clearances.size else float("inf")
    clearance_score = float(np.clip(min_clearance / QUALITY_BODY_RADIUS, 0.0, 1.0))

    components = [path_efficiency, smoothness, clearance_score]
    if any(not np.isfinite(v) for v in components):
        quality = float("nan")
    else:
        # Geometric mean; a floor keeps a single zero from erasing all signal.
        quality = float(np.exp(np.mean(np.log(np.clip(components, 1e-6, 1.0)))))

    duration = float((xy.shape[0] - 1) * dt)
    peak_speed = float(np.max(speed)) if speed.size else 0.0
    result: Dict[str, float] = {
        "quality_score": quality,
        "path_efficiency": path_efficiency,
        "smoothness": smoothness,
        "clearance_score": clearance_score,
        "smoothness_ldlj": smoothness_ldlj,
        "path_length": path_length,
        "reference_path_length": float(reference_path_length),
        "min_clearance": min_clearance,
        "mean_clearance": float(np.mean(clearances)) if clearances.size else float("inf"),
        "ldlj": float(ldlj),
        "sparc": float(sparc),
        "duration_sec": duration,
        "peak_speed": peak_speed,
        "mean_speed": path_length / duration if duration > 0 else 0.0,
        "final_goal_error": float(np.linalg.norm(xy[-1] - goal)),
    }

    if control is not None and control.shape[0] >= 1:
        norms = np.linalg.norm(control, axis=1)
        result["peak_control_norm"] = float(np.max(norms))
        result["mean_control_norm"] = float(np.mean(norms))
        result["control_saturation_frac"] = float(np.mean(norms > u_max * 1.001))
        result["peak_control_ratio"] = float(np.max(norms) / u_max) if u_max > 0 else float("nan")
    return result
