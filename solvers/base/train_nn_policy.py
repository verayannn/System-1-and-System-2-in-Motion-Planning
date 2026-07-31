"""
train_nn_policy.py - neural System-1 policy trainer.

Model:
    (local context, situation vector, local dynamics features, local goal vector) -> u_t

Main improvements:
- Continual learning with --init_model
- Stronger behavior-level losses:
    1. control imitation loss
    2. next-step dynamics loss
    3. direction alignment loss
    4. relative speed matching loss
    5. goal-progress margin loss
- Sample weighting:
    emphasizes near-goal and high-progress samples
- Gradient clipping
- ReduceLROnPlateau scheduler

This is designed to improve closed-loop S1_accept, not just lower control MSE.



python solvers/base/train_nn_policy.py \
  --dataset db/nn_dataset_maze.npz \
  --model_out db/s1_policy_control_cnn.pth \
  --epochs 25 \
  --lambda_u 1.0 \
  --lambda_next 1.0 \
  --lambda_dir 0.5 \
  --lambda_speed 1.0 \
  --lambda_progress 1.0 \
  --progress_fraction 0.9
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Model
# ============================================================

class NeuralSystem1ControlPolicyCNN(nn.Module):
    def __init__(
        self,
        ctx_shape,
        sit_dim,
        dyn_dim,
        goal_dim,
        u_dim,
        hidden=256,
        cnn_channels=128,
        dropout=0.05,
    ):
        super().__init__()
        self.ctx_shape = tuple(ctx_shape)
        self.sit_dim = int(sit_dim)
        self.dyn_dim = int(dyn_dim)
        self.goal_dim = int(goal_dim)
        self.u_dim = int(u_dim)

        _, ctx_dim = self.ctx_shape

        self.ctx_encoder = nn.Sequential(
            nn.Conv1d(ctx_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.sit_encoder = nn.Sequential(
            nn.Linear(self.sit_dim, hidden // 2),
            nn.ReLU(),
        )

        self.dyn_encoder = nn.Sequential(
            nn.Linear(self.dyn_dim, hidden // 2),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, hidden // 4),
            nn.ReLU(),
        )

        fused_dim = cnn_channels + hidden // 2 + hidden // 2 + hidden // 4

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.u_dim),
        )

    def forward(self, ctx, sit, dyn, goal):
        x_ctx = ctx.transpose(1, 2)
        f_ctx = self.ctx_encoder(x_ctx).squeeze(-1)
        f_sit = self.sit_encoder(sit)
        f_dyn = self.dyn_encoder(dyn)
        f_goal = self.goal_encoder(goal)
        z = torch.cat([f_ctx, f_sit, f_dyn, f_goal], dim=1)
        return self.head(z)


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
SOLVER_DIR = THIS_DIR.parent
INSTANCE_DIR = SOLVER_DIR.parent
LEGACY_CODES_ROOT = Path(
    os.environ.get("SOFAI_LEGACY_CODES_ROOT", INSTANCE_DIR)
).expanduser()


def _strip_legacy_maze_prefix(path: Path) -> Path:
    if len(path.parts) >= 3 and path.parts[0] == "maze" and path.parts[1] == "n_s1":
        return Path(*path.parts[2:])
    return path


def resolve_existing_path(path_like: str, *, required: bool = False) -> Path:
    if not path_like:
        return Path()

    path = Path(path_like).expanduser()
    if path.is_absolute():
        if path.exists() or not required:
            return path
        raise FileNotFoundError(f"Required path does not exist: {path}")

    local_suffix = _strip_legacy_maze_prefix(path)
    candidates = [
        Path.cwd() / path,
        INSTANCE_DIR / path,
        SOLVER_DIR / local_suffix,
        LEGACY_CODES_ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Required path does not exist: {path_like}\nSearched:\n  {searched}")
    return INSTANCE_DIR / path


def resolve_output_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) >= 3 and path.parts[0] == "maze" and path.parts[1] == "n_s1":
        return SOLVER_DIR / Path(*path.parts[2:])
    return INSTANCE_DIR / _strip_legacy_maze_prefix(path)


# ============================================================
# Helpers
# ============================================================

def grouped_train_val_split(traj_ids: np.ndarray, val_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(traj_ids)
    rng.shuffle(unique_ids)

    n_val_groups = max(1, int(len(unique_ids) * val_frac))
    val_ids = set(unique_ids[:n_val_groups].tolist())

    val_mask = np.array([tid in val_ids for tid in traj_ids], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def denorm(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x_norm * std + mean


def load_previous_weights_if_possible(model: nn.Module, init_model: str, device: torch.device):
    if not init_model:
        return

    path = Path(init_model)
    if not path.exists():
        print(f"[warn] init_model not found: {path}. Training from scratch.")
        return

    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if "state_dict" not in ckpt:
        print(f"[warn] init_model has no state_dict: {path}. Training from scratch.")
        return

    try:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[init] loaded previous model: {path}")
        if missing:
            print(f"[init] missing keys: {missing}")
        if unexpected:
            print(f"[init] unexpected keys: {unexpected}")
    except RuntimeError as e:
        print(f"[warn] could not load init_model due to architecture mismatch: {e}")
        print("[warn] training from scratch.")


def make_sample_weights(
    goal_den: np.ndarray,
    next_den: np.ndarray,
    near_goal_boost: float = 2.0,
    progress_boost: float = 2.0,
) -> np.ndarray:
    """
    Weight samples that matter more for S1_accept:
    - near-goal samples
    - samples with high expert progress
    """
    eps = 1e-6

    goal_dist_now = np.linalg.norm(goal_den, axis=1)
    goal_dist_next = np.linalg.norm(goal_den - next_den, axis=1)
    expert_progress = np.maximum(goal_dist_now - goal_dist_next, 0.0)

    # Near-goal importance: larger when goal is close
    near_score = 1.0 / (goal_dist_now + 0.5)
    near_score = near_score / (near_score.mean() + eps)

    # Progress importance: larger when expert makes meaningful progress
    prog_score = expert_progress / (expert_progress.mean() + eps)

    w = 1.0 + near_goal_boost * near_score + progress_boost * prog_score
    w = w / (w.mean() + eps)

    return w.astype(np.float32)


def weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return (x * w).mean()


def pointwise_regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str, huber_delta: float) -> torch.Tensor:
    if loss_type == "huber":
        return F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=max(float(huber_delta), 1e-6),
        ).mean(dim=1)
    return ((pred - target) ** 2).mean(dim=1)


def freeze_for_conservative_finetune(model: nn.Module, train_head_only: bool):
    if not train_head_only:
        return
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] train_head_only=True | trainable_params={n_train}/{n_total}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Solvers/nn_dataset_maze.npz")
    ap.add_argument("--model_out", type=str, default="Solvers/s1_policy_control_cnn.pth")

    # Continual learning
    ap.add_argument("--init_model", type=str, default="")

    # Training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cnn_channels", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    # Loss weights
    ap.add_argument("--lambda_u", type=float, default=1.0)
    ap.add_argument("--lambda_next", type=float, default=1.0)
    ap.add_argument("--lambda_dir", type=float, default=0.25)
    ap.add_argument("--lambda_speed", type=float, default=0.25)
    ap.add_argument("--lambda_progress", type=float, default=0.5)
    ap.add_argument("--lambda_u_range", type=float, default=0.0,
                    help="Penalty for predicted denormalized controls outside --behavior_u_clip.")
    ap.add_argument("--loss_type", type=str, choices=["mse", "huber"], default="mse")
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--behavior_u_clip", type=float, default=0.0,
                    help="Clip denormalized predicted controls inside behavior losses; <=0 disables.")
    ap.add_argument("--speed_loss_min_step", type=float, default=0.05,
                    help="Floor in relative speed loss denominator.")
    ap.add_argument("--speed_loss_clip", type=float, default=0.0,
                    help="Clip per-sample relative speed loss; <=0 disables.")
    ap.add_argument("--train_head_only", action="store_true",
                    help="Freeze encoders and fine-tune only the policy head.")

    # Progress shaping
    ap.add_argument("--progress_fraction", type=float, default=0.75,
                    help="Predicted step should achieve this fraction of expert progress.")

    # Sample weighting
    ap.add_argument("--near_goal_boost", type=float, default=2.0)
    ap.add_argument("--progress_boost", type=float, default=2.0)
    ap.add_argument("--max_sample_weight", type=float, default=25.0,
                    help="Clip final per-sample weights after multiplying any dataset-provided sample_weight.")

    args = ap.parse_args()
    args.dataset = str(resolve_existing_path(args.dataset, required=True))
    args.model_out = str(resolve_output_path(args.model_out))
    if args.init_model:
        args.init_model = str(resolve_existing_path(args.init_model, required=False))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(args.dataset, allow_pickle=True)

    X_ctx = torch.from_numpy(data["ctx"]).float()
    X_sit = torch.from_numpy(data["sit"]).float()
    X_dyn = torch.from_numpy(data["dyn"]).float()
    X_goal = torch.from_numpy(data["goal"]).float()
    Y_u = torch.from_numpy(data["u"]).float()
    Y_next = torch.from_numpy(data["next_local"]).float()
    traj_ids = np.asarray(data["traj_id"], dtype=np.int32)

    meta = data["meta"].item() if isinstance(data["meta"], np.ndarray) else data["meta"]

    # --------------------------------------------------------
    # Build sample weights from denormalized goal and next step
    # --------------------------------------------------------
    goal_den_np = (
        data["goal"] * data["norm_goal_std"][None, :]
        + data["norm_goal_mean"][None, :]
    )
    next_den_np = (
        data["next_local"] * data["norm_next_std"][None, :]
        + data["norm_next_mean"][None, :]
    )

    sample_weights_np = make_sample_weights(
        goal_den=goal_den_np,
        next_den=next_den_np,
        near_goal_boost=args.near_goal_boost,
        progress_boost=args.progress_boost,
    )
    if "sample_weight" in data.files:
        dataset_weight = np.asarray(data["sample_weight"], dtype=np.float32).reshape(-1)
        if dataset_weight.shape[0] != sample_weights_np.shape[0]:
            raise ValueError(
                f"sample_weight length {dataset_weight.shape[0]} does not match dataset length {sample_weights_np.shape[0]}"
            )
        sample_weights_np = sample_weights_np * dataset_weight

    if args.max_sample_weight > 0:
        sample_weights_np = np.minimum(sample_weights_np, float(args.max_sample_weight))

    sample_weights_np = sample_weights_np / (sample_weights_np.mean() + 1e-6)
    W = torch.from_numpy(sample_weights_np).float()

    train_mask, val_mask = grouped_train_val_split(traj_ids, args.val_frac, seed=42)

    train_ds = TensorDataset(
        X_ctx[train_mask],
        X_sit[train_mask],
        X_dyn[train_mask],
        X_goal[train_mask],
        Y_u[train_mask],
        Y_next[train_mask],
        W[train_mask],
    )
    val_ds = TensorDataset(
        X_ctx[val_mask],
        X_sit[val_mask],
        X_dyn[val_mask],
        X_goal[val_mask],
        Y_u[val_mask],
        Y_next[val_mask],
        W[val_mask],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = NeuralSystem1ControlPolicyCNN(
        ctx_shape=X_ctx.shape[1:],
        sit_dim=X_sit.shape[1],
        dyn_dim=X_dyn.shape[1],
        goal_dim=X_goal.shape[1],
        u_dim=Y_u.shape[1],
        hidden=args.hidden,
        cnn_channels=args.cnn_channels,
        dropout=args.dropout,
    ).to(device)

    load_previous_weights_if_possible(model, args.init_model, device)
    freeze_for_conservative_finetune(model, train_head_only=bool(args.train_head_only))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Denormalization constants
    dyn_mean = torch.from_numpy(data["norm_dyn_mean"]).float().to(device)
    dyn_std = torch.from_numpy(data["norm_dyn_std"]).float().to(device)

    goal_mean = torch.from_numpy(data["norm_goal_mean"]).float().to(device)
    goal_std = torch.from_numpy(data["norm_goal_std"]).float().to(device)

    u_mean = torch.from_numpy(data["norm_u_mean"]).float().to(device)
    u_std = torch.from_numpy(data["norm_u_std"]).float().to(device)

    next_mean = torch.from_numpy(data["norm_next_mean"]).float().to(device)
    next_std = torch.from_numpy(data["norm_next_std"]).float().to(device)

    dt_nom = float(meta["dt_nom"])
    u_dim = int(meta["u_dim"])

    best_val = float("inf")
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    eps = 1e-6

    def compute_loss(b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w):
        pred_u = model(b_ctx, b_sit, b_dyn, b_goal)

        # ----------------------------------------------------
        # 1. Control imitation loss in normalized u-space
        # ----------------------------------------------------
        loss_u_each = pointwise_regression_loss(pred_u, b_u, args.loss_type, args.huber_delta)
        loss_u = weighted_mean(loss_u_each, b_w)

        # ----------------------------------------------------
        # Denormalize for behavioral losses
        # ----------------------------------------------------
        pred_u_den = denorm(pred_u, u_mean, u_std)
        if args.behavior_u_clip > 0:
            u_lim = float(args.behavior_u_clip)
            pred_u_behavior = torch.clamp(pred_u_den, -u_lim, u_lim)
            loss_u_range_each = torch.relu(torch.abs(pred_u_den) - u_lim).pow(2).mean(dim=1)
        else:
            pred_u_behavior = pred_u_den
            loss_u_range_each = torch.zeros_like(loss_u_each)
        loss_u_range = weighted_mean(loss_u_range_each, b_w)

        goal_den = denorm(b_goal, goal_mean, goal_std)
        next_true = denorm(b_next, next_mean, next_std)

        dyn_den = denorm(b_dyn, dyn_mean, dyn_std)

        # dyn layout:
        # [A_local_flat(4), B_local_flat(2*u_dim), drift_local(2)]
        B_start = 4
        B_end = 4 + 2 * u_dim ######## examine this
        drift_start = B_end
        drift_end = B_end + 2

        B_local_flat = dyn_den[:, B_start:B_end]
        drift_local = dyn_den[:, drift_start:drift_end]

        B_local = B_local_flat.view(-1, 2, u_dim)

        pred_next = dt_nom * (
            drift_local
            + torch.bmm(B_local, pred_u_behavior.unsqueeze(-1)).squeeze(-1)
        )
                


        # ----------------------------------------------------
        # 2. Next-step loss in local position space
        # ----------------------------------------------------
        loss_next_each = pointwise_regression_loss(pred_next, next_true, args.loss_type, args.huber_delta)
        loss_next = weighted_mean(loss_next_each, b_w)

        # ----------------------------------------------------
        # 3. Direction alignment loss
        #    This fixes the flat goal/speed issue better than
        #    only matching control MSE.
        # ----------------------------------------------------
        pred_norm = torch.norm(pred_next, dim=1) + eps
        true_norm = torch.norm(next_true, dim=1) + eps

        cos_sim = (pred_next * next_true).sum(dim=1) / (pred_norm * true_norm)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        loss_dir_each = 1.0 - cos_sim
        loss_dir = weighted_mean(loss_dir_each, b_w)

        # ----------------------------------------------------
        # 4. Relative speed loss
        #    Penalizes both too slow and too fast.
        # ----------------------------------------------------
        denom = torch.clamp(true_norm.detach(), min=float(args.speed_loss_min_step))
        loss_speed_each = ((pred_norm - true_norm) / denom) ** 2
        if args.speed_loss_clip > 0:
            loss_speed_each = torch.clamp(loss_speed_each, max=float(args.speed_loss_clip))
        loss_speed = weighted_mean(loss_speed_each, b_w)

        # ----------------------------------------------------
        # 5. Progress-margin loss
        #    Current state is origin in local frame.
        # ----------------------------------------------------
        dist_now = torch.norm(goal_den, dim=1)
        dist_pred = torch.norm(goal_den - pred_next, dim=1)
        dist_true = torch.norm(goal_den - next_true, dim=1)

        true_progress = torch.relu(dist_now - dist_true)
        pred_progress = dist_now - dist_pred

        required_progress = args.progress_fraction * true_progress
        loss_progress_each = torch.relu(required_progress - pred_progress)
        loss_progress = weighted_mean(loss_progress_each, b_w)

        total_loss = (
            args.lambda_u * loss_u
            + args.lambda_next * loss_next
            + args.lambda_dir * loss_dir
            + args.lambda_speed * loss_speed
            + args.lambda_progress * loss_progress
            + args.lambda_u_range * loss_u_range
        )

        parts = {
            "loss_u": float(loss_u.detach().cpu()),
            "loss_next": float(loss_next.detach().cpu()),
            "loss_dir": float(loss_dir.detach().cpu()),
            "loss_speed": float(loss_speed.detach().cpu()),
            "loss_progress": float(loss_progress.detach().cpu()),
            "loss_u_range": float(loss_u_range.detach().cpu()),
            "mean_pred_step": float(pred_norm.mean().detach().cpu()),
            "mean_true_step": float(true_norm.mean().detach().cpu()),
            "mean_pred_progress": float(pred_progress.mean().detach().cpu()),
            "mean_true_progress": float(true_progress.mean().detach().cpu()),
        }

        return total_loss, parts

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w in train_loader:
            b_ctx = b_ctx.to(device)
            b_sit = b_sit.to(device)
            b_dyn = b_dyn.to(device)
            b_goal = b_goal.to(device)
            b_u = b_u.to(device)
            b_next = b_next.to(device)
            b_w = b_w.to(device)

            optimizer.zero_grad()
            loss, _ = compute_loss(b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        val_parts_sum = {
            "loss_u": 0.0,
            "loss_next": 0.0,
            "loss_dir": 0.0,
            "loss_speed": 0.0,
            "loss_progress": 0.0,
            "loss_u_range": 0.0,
            "mean_pred_step": 0.0,
            "mean_true_step": 0.0,
            "mean_pred_progress": 0.0,
            "mean_true_progress": 0.0,
        }

        with torch.no_grad():
            for b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w in val_loader:
                b_ctx = b_ctx.to(device)
                b_sit = b_sit.to(device)
                b_dyn = b_dyn.to(device)
                b_goal = b_goal.to(device)
                b_u = b_u.to(device)
                b_next = b_next.to(device)
                b_w = b_w.to(device)

                loss, parts = compute_loss(b_ctx, b_sit, b_dyn, b_goal, b_u, b_next, b_w)

                val_loss_sum += loss.item()
                for k, v in parts.items():
                    val_parts_sum[k] += v

        val_loss = val_loss_sum / max(1, len(val_loader))
        for k in val_parts_sum:
            val_parts_sum[k] /= max(1, len(val_loader))

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:03d}/{args.epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"u={val_parts_sum['loss_u']:.4f} | "
                f"next={val_parts_sum['loss_next']:.4f} | "
                f"dir={val_parts_sum['loss_dir']:.4f} | "
                f"speed={val_parts_sum['loss_speed']:.4f} | "
                f"prog={val_parts_sum['loss_progress']:.4f} | "
                f"range={val_parts_sum['loss_u_range']:.4f} | "
                f"pred_step={val_parts_sum['mean_pred_step']:.4f} | "
                f"true_step={val_parts_sum['mean_true_step']:.4f} | "
                f"pred_prog={val_parts_sum['mean_pred_progress']:.4f} | "
                f"true_prog={val_parts_sum['mean_true_progress']:.4f} | "
                f"lr={lr_now:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": "control_cnn_v2_behavioral",
                    "meta": {
                        "ctx_shape": tuple(X_ctx.shape[1:]),
                        "sit_dim": int(X_sit.shape[1]),
                        "dyn_dim": int(X_dyn.shape[1]),
                        "goal_dim": int(X_goal.shape[1]),
                        "u_dim": int(Y_u.shape[1]),
                        "hidden": int(args.hidden),
                        "cnn_channels": int(args.cnn_channels),
                        "dropout": float(args.dropout),
                        "dataset_meta": meta,
                    },
                    "norm": {
                        "ctx_mean": data["norm_ctx_mean"].astype(np.float32),
                        "ctx_std": data["norm_ctx_std"].astype(np.float32),
                        "sit_mean": data["norm_sit_mean"].astype(np.float32),
                        "sit_std": data["norm_sit_std"].astype(np.float32),
                        "dyn_mean": data["norm_dyn_mean"].astype(np.float32),
                        "dyn_std": data["norm_dyn_std"].astype(np.float32),
                        "goal_mean": data["norm_goal_mean"].astype(np.float32),
                        "goal_std": data["norm_goal_std"].astype(np.float32),
                        "u_mean": data["norm_u_mean"].astype(np.float32),
                        "u_std": data["norm_u_std"].astype(np.float32),
                        "next_mean": data["norm_next_mean"].astype(np.float32),
                        "next_std": data["norm_next_std"].astype(np.float32),
                    },
                    "best_val_loss": float(best_val),
                    "continued_from": str(args.init_model),
                    "loss_weights": {
                        "lambda_u": float(args.lambda_u),
                        "lambda_next": float(args.lambda_next),
                        "lambda_dir": float(args.lambda_dir),
                        "lambda_speed": float(args.lambda_speed),
                        "lambda_progress": float(args.lambda_progress),
                        "lambda_u_range": float(args.lambda_u_range),
                        "loss_type": str(args.loss_type),
                        "huber_delta": float(args.huber_delta),
                        "behavior_u_clip": float(args.behavior_u_clip),
                        "speed_loss_min_step": float(args.speed_loss_min_step),
                        "speed_loss_clip": float(args.speed_loss_clip),
                        "train_head_only": bool(args.train_head_only),
                        "progress_fraction": float(args.progress_fraction),
                    },
                },
                str(model_out),
            )

    print(f"Done. Best model saved to {model_out} with val loss {best_val:.6f}")


if __name__ == "__main__":
    main()
