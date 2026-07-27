"""Forward/gradient checks for the filter-aware differentiable rollout."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from solvers import s1_nonlinear as runtime


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_s1_nonlinear",
    ROOT / "script" / "train_s1_nonlinear.py",
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def scenario(rectangles: list[tuple[float, float, float, float]]) -> SimpleNamespace:
    return SimpleNamespace(
        rectangles=rectangles,
        goal=np.array([5.0, 0.0], dtype=np.float32),
        nonlinear_dynamics={},
    )


def torch_filter(
    *,
    u_pred: np.ndarray,
    u_goal: np.ndarray,
    rectangles: list[tuple[float, float, float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    rects = torch.tensor(rectangles or [(0.0, 0.0, 0.0, 0.0)], dtype=torch.float32)[None, :, :]
    mask = torch.tensor([[bool(rectangles)]], dtype=torch.bool)
    return trainer.policy_first_safe_control_straight_through(
        u_pred_local=torch.tensor(u_pred, dtype=torch.float32)[None, :],
        u_goal_local=torch.tensor(u_goal, dtype=torch.float32)[None, :],
        position=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        heading=torch.tensor([0.0], dtype=torch.float32),
        drift_global=torch.zeros((1, 2), dtype=torch.float32),
        goal_global=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
        rects=rects,
        rect_mask=mask,
        dt=0.1,
        u_max=3.0,
        collision_margin=0.0,
    )


class PolicyFilterTrainingTest(unittest.TestCase):
    def assert_matches_runtime(
        self,
        u_pred: np.ndarray,
        u_goal: np.ndarray,
        rectangles: list[tuple[float, float, float, float]],
    ) -> None:
        expected_u, expected_next = runtime.choose_safe_control(
            scenario=scenario(rectangles),
            x_curr=np.array([0.0, 0.0], dtype=np.float32),
            # rollout_policy clips the raw network command immediately before
            # calling choose_safe_control.
            u_pred_local=np.clip(u_pred, -3.0, 3.0),
            u_goal_local=u_goal,
            heading=0.0,
            dt=0.1,
            rects=rectangles,
            goal=np.array([5.0, 0.0], dtype=np.float32),
            u_max=3.0,
            collision_margin=0.0,
            mode="policy",
        )
        actual_u, actual_next = torch_filter(
            u_pred=u_pred,
            u_goal=u_goal,
            rectangles=rectangles,
        )
        np.testing.assert_allclose(actual_u.detach().numpy()[0], expected_u, atol=1e-6)
        np.testing.assert_allclose(actual_next.detach().numpy()[0], expected_next, atol=1e-6)

    def test_unobstructed_control_matches_runtime(self) -> None:
        self.assert_matches_runtime(
            np.array([2.0, 0.25], dtype=np.float32),
            np.array([3.0, 0.0], dtype=np.float32),
            [],
        )

    def test_clipped_control_matches_runtime(self) -> None:
        self.assert_matches_runtime(
            np.array([5.0, -4.0], dtype=np.float32),
            np.array([3.0, 0.0], dtype=np.float32),
            [],
        )

    def test_deflection_matches_runtime(self) -> None:
        self.assert_matches_runtime(
            np.array([2.0, 0.0], dtype=np.float32),
            np.array([3.0, 0.0], dtype=np.float32),
            [(0.1, -0.2, 0.3, 0.2)],
        )

    def test_no_safe_candidate_returns_runtime_zero_control(self) -> None:
        self.assert_matches_runtime(
            np.array([2.0, 0.0], dtype=np.float32),
            np.array([3.0, 0.0], dtype=np.float32),
            [(-1.0, -1.0, 1.0, 1.0)],
        )

    def test_straight_through_gradient_reaches_raw_action(self) -> None:
        u_pred = torch.tensor([[2.0, 0.0]], dtype=torch.float32, requires_grad=True)
        safe_u, _ = trainer.policy_first_safe_control_straight_through(
            u_pred_local=u_pred,
            u_goal_local=torch.tensor([[3.0, 0.0]], dtype=torch.float32),
            position=torch.zeros((1, 2), dtype=torch.float32),
            heading=torch.zeros(1, dtype=torch.float32),
            drift_global=torch.zeros((1, 2), dtype=torch.float32),
            goal_global=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            rects=torch.tensor([[[0.1, -0.2, 0.3, 0.2]]], dtype=torch.float32),
            rect_mask=torch.ones((1, 1), dtype=torch.bool),
            dt=0.1,
            u_max=3.0,
            collision_margin=0.0,
        )
        self.assertFalse(torch.allclose(safe_u.detach(), u_pred.detach()))
        safe_u.square().sum().backward()
        self.assertIsNotNone(u_pred.grad)
        self.assertTrue(torch.isfinite(u_pred.grad).all())
        self.assertGreater(float(u_pred.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
