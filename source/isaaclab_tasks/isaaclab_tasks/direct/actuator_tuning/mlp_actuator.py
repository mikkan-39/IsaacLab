# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inference for the standalone servo MLP, used to generate IdealPD position targets.

The MLP is trained by ``scripts/tools/actuator_tuning/train_servo_mlp.py`` as an actuator-net-style
one-step predictor: from a window of the servo's recent tracking **error** and **velocity** it
predicts the next position *increment*. Integrating the increment yields a (delay-free) position
trajectory that we feed as the position target to an :class:`~isaaclab.actuators.IdealPDActuator`.

The feature layout and rollout here **must stay byte-for-byte consistent** with the training script
(``build_mlp`` / ``rollout_segment`` there); they are duplicated rather than imported because the
training script is intentionally Isaac-Lab-free while this module lives inside the task package.
"""

from __future__ import annotations

import numpy as np


def _build_mlp(in_dim: int, hidden, out_dim: int = 1):
    """Reconstruct the training MLP (plain ``nn.Sequential`` of Linear/ReLU)."""
    import torch.nn as nn

    layers: list = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def load_mlp(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load a trained servo-MLP checkpoint and rebuild the model.

    Returns a dict with the eval-mode ``model`` plus the normalization stats and metadata needed
    for :func:`rollout_targets` (``history``, ``x_mean``, ``x_std``, ``y_mean``, ``y_std``, ...).
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    history = int(ckpt["history"])
    model = _build_mlp(2 * history, list(ckpt["hidden"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "device": device,
        "history": history,
        "x_mean": np.asarray(ckpt["x_mean"], dtype=np.float64),
        "x_std": np.asarray(ckpt["x_std"], dtype=np.float64),
        "y_mean": float(ckpt["y_mean"]),
        "y_std": float(ckpt["y_std"]),
        "control_hz": float(ckpt.get("control_hz", 50.0)),
        "ref_lag_steps": float(ckpt.get("ref_lag_steps", 0.0)),
        "velocity_source": str(ckpt.get("velocity_source", "derived")),
    }


def rollout_targets(
    mlp: dict,
    target: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    segment_ids: np.ndarray,
    step_dt: float,
) -> np.ndarray:
    """Closed-loop autoregressive rollout of the MLP, producing a position-target trajectory.

    The rollout is performed independently per excitation segment (matching how the model was
    validated): the first ``history`` steps of each segment are warm-started from the recorded real
    position/velocity, after which the model free-runs on its own predictions. Velocity is taken as
    the causal backward-difference of the predicted position during the free-run.

    Args:
        mlp: Object returned by :func:`load_mlp`.
        target: Commanded joint position per control step (rad). Shape ``(N,)``.
        ref_pos: Recorded real joint position per control step (rad). Used only to warm-start each
            segment. Shape ``(N,)``.
        ref_vel: Recorded real joint velocity per control step (rad/s). Warm-start only. Shape ``(N,)``.
        segment_ids: Per-step segment index (contiguous blocks). Shape ``(N,)``.
        step_dt: Control time-step (seconds).

    Returns:
        Predicted position trajectory to use as the IdealPD position target. Shape ``(N,)`` float32.
    """
    import torch

    model = mlp["model"]
    device = mlp["device"]
    history = mlp["history"]
    x_mean, x_std = mlp["x_mean"], mlp["x_std"]
    y_mean, y_std = mlp["y_mean"], mlp["y_std"]

    target = np.asarray(target, dtype=np.float64)
    pos_warm = np.asarray(ref_pos, dtype=np.float64).copy()
    vel_warm = np.asarray(ref_vel, dtype=np.float64).copy()

    # The MLP was trained on real positions advanced by `ref_lag_steps` (delay-free frame). Warm-start
    # each segment in that same frame so the model's inputs match training, independent of how the env
    # scores (which compares against the raw, delayed recording).
    lag = float(mlp.get("ref_lag_steps", 0.0))
    if lag:
        n = pos_warm.shape[0]
        src = np.clip(np.arange(n, dtype=np.float64) + lag, 0.0, n - 1)
        lo = np.floor(src).astype(int)
        hi = np.clip(lo + 1, 0, n - 1)
        frac = src - lo
        pos_warm = pos_warm[lo] * (1.0 - frac) + pos_warm[hi] * frac
        vel_warm = vel_warm[lo] * (1.0 - frac) + vel_warm[hi] * frac

    pos_pred = pos_warm.copy()
    # warm-start velocity must match how the model was trained (derived = backward-difference of
    # position; recorded = the CSV speed column). The free-run portion always uses derived velocity.
    if mlp["velocity_source"] == "recorded":
        vel_pred = vel_warm
    else:
        vel_pred = np.zeros_like(pos_pred)
        vel_pred[1:] = (pos_pred[1:] - pos_pred[:-1]) / step_dt

    for s in np.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == s)
        a, b = int(idx[0]), int(idx[-1]) + 1
        for k in range(a + history - 1, b - 1):
            win = slice(k - history + 1, k + 1)
            e = (target[win] - pos_pred[win])[::-1]
            v = vel_pred[win][::-1]
            feat = np.concatenate([e, v])
            xn = (feat - x_mean) / x_std
            with torch.no_grad():
                dn = model(torch.as_tensor(xn, dtype=torch.float32, device=device).unsqueeze(0)).item()
            pos_pred[k + 1] = pos_pred[k] + (dn * y_std + y_mean)
            vel_pred[k + 1] = (pos_pred[k + 1] - pos_pred[k]) / step_dt

    return pos_pred.astype(np.float32)
