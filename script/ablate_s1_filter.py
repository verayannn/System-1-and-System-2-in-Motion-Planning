"""Ablation: how much of S1's success comes from the policy vs. from the one-step
safety filter in `choose_safe_control`?

Variants:
  greedy         - the original filter: rank candidates by one-step distance to
                   the goal with a hard penalty on any action that does not close
                   that distance
  policy         - the deviation-minimising filter: the policy action is the
                   command, deflected only as much as safety requires
  greedy_no_net  - the original filter with the policy deleted and replaced by
                   the hand-written nominal goal control. Whatever this scores is
                   the part of the success rate that owes nothing to learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from solvers import s1_nonlinear as nl

VARIANTS = ("greedy", "policy", "greedy_no_net")

_ORIGINAL = nl.choose_safe_control


def _filter(variant: str):
    """Wrap the real filter so the policy input can be ablated away."""

    def wrapped(*, scenario, x_curr, u_pred_local, u_goal_local, heading, dt,
                rects, goal, u_max, collision_margin, mode=None):
        del mode
        proposal = u_goal_local if variant == "greedy_no_net" else u_pred_local
        filter_mode = nl.FILTER_MODE_POLICY if variant == "policy" else nl.FILTER_MODE_GREEDY
        return _ORIGINAL(
            scenario=scenario, x_curr=x_curr,
            u_pred_local=proposal, u_goal_local=u_goal_local,
            heading=heading, dt=dt, rects=rects, goal=goal,
            u_max=u_max, collision_margin=collision_margin, mode=filter_mode,
        )

    return wrapped


def run_mode(mode: str, scenarios: List[Dict[str, Any]], model, norm, device, args) -> Dict[str, Any]:
    nl.choose_safe_control = _filter(mode)
    try:
        solved: List[int] = []
        rows = []
        for idx, scenario in enumerate(scenarios):
            goal_tol = nl.scenario_goal_tol(scenario, float(args["goal_tol"]))
            traj, _controls, info = nl.rollout_policy(
                model, scenario, norm, device,
                total_steps=int(args["total_steps"]),
                action_hold=int(args["action_hold"]),
                stall_patience=int(args["stall_patience"]),
                stall_tol=float(args["stall_tol"]),
                progress_patience=int(args["progress_patience"]),
                progress_tol=float(args["progress_tol"]),
                dt_nom=float(args["dt_nom"]),
                u_max_nom=float(args["u_max_nom"]),
                collision_margin=float(args["collision_margin"]),
                goal_tol=goal_tol,
                grid_n=int(args["grid_n"]),
                n_steps_nom=int(args["n_steps_nom"]),
                buffer_cells=int(args["buffer_cells"]),
                stop_tol=float(args["stop_tol"]),
            )
            goal = nl.scenario_goal(scenario)
            final_dist = float(np.linalg.norm(traj[-1, :2] - goal[:2]))
            ok = bool(final_dist <= max(goal_tol, 0.75))
            if ok:
                solved.append(idx)
            path_len = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)))
            rows.append(dict(sid=idx, success=ok, steps=int(len(traj)),
                             final_dist=final_dist, path_length=path_len))
        return dict(mode=mode, solved=solved, n=len(scenarios), rows=rows)
    finally:
        nl.choose_safe_control = _ORIGINAL


def load_scenarios(path: Path, limit: int) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("scenarios", "problems", "cases", "entries"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise SystemExit(f"Could not find a scenario list in {path}")
    return payload[:limit]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dictionary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default="output/s1_filter_ablation.json")
    a = p.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(1)
    model, norm, meta = nl.load_s1_checkpoint(Path(a.model), device)

    from solvers.S1_memory_neural import _make_args

    args = _make_args(meta)
    scenarios = load_scenarios(Path(a.dictionary), a.limit)
    print(f"scenarios={len(scenarios)} dt_nom={args['dt_nom']} total_steps={args['total_steps']} "
          f"action_hold={args['action_hold']} u_max={args['u_max_nom']}")

    results = {}
    for mode in VARIANTS:
        out = run_mode(mode, scenarios, model, norm, device, args)
        results[mode] = out
        k = len(out["solved"])
        steps = np.mean([r["steps"] for r in out["rows"]])
        print(f"  {mode:14s} success={k}/{out['n']} ({k / out['n']:.3f})  mean_steps={steps:.1f}")

    base = set(results["greedy"]["solved"])
    for mode in VARIANTS[1:]:
        other = set(results[mode]["solved"])
        inter = len(base & other)
        union = max(1, len(base | other))
        print(f"\ngreedy vs {mode}: |∩|={inter} Jaccard={inter / union:.3f}")
        print(f"  greedy-only={sorted(base - other)}")
        print(f"  {mode}-only={sorted(other - base)}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
