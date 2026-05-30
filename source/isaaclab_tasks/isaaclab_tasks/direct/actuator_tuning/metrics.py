# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-matching metrics for actuator parameter tuning.

The scoring is position-first (velocity from differentiated encoder data is noisy and
de-emphasized). Transient and reversal segments of the *commanded* trajectory are weighted
more heavily because servo "feel" matters most at startup/braking/reversals.

Score (lower is better)::

    score = tracking

``tracking`` depends on ``score_mode``:

    position_only  : weighted_pos_mse
    position_heavy : 0.9 * weighted_pos_mse + 0.1 * weighted_vel_mse
    balanced       : 0.7 * weighted_pos_mse + 0.3 * weighted_vel_mse

``spike_pos_err`` (a high percentile of |position error|) and ``max_abs_pos`` are still computed
and reported as diagnostics, but no longer enter the score: penalizing worst-case spikes was
rewarding timid, lagging, low-amplitude solutions over ones that actually track the motion.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

SCORE_MODES = ("position_only", "position_heavy", "balanced")

_TRACKING_WEIGHTS = {
    "position_only": (1.0, 0.0),
    "position_heavy": (0.9, 0.1),
    "balanced": (0.7, 0.3),
}


@dataclass
class WeightSpec:
    """Configuration for the per-step error weighting masks."""

    move_eps: float = 0.05
    """Target speed threshold (rad/s) above which the joint is considered moving."""

    transient_window_s: float = 0.3
    """Duration after a movement start that receives the transient weight."""

    transient_weight: float = 2.0
    """Multiplier applied during the post-movement-start window."""

    reversal_weight: float = 2.0
    """Multiplier applied at direction reversals / sudden acceleration changes."""

    accel_thresh: float = 5.0
    """|d2(target)/dt2| (rad/s^2) above which a sample counts as a sudden accel change."""


def compute_weights(target: np.ndarray, step_dt: float, spec: WeightSpec) -> tuple[np.ndarray, np.ndarray]:
    """Build per-step error weights and a boolean reversal mask from the commanded trajectory.

    Args:
        target: Commanded position per control step (rad). Shape ``(N,)``.
        step_dt: Control time-step (seconds).
        spec: Weighting configuration.

    Returns:
        ``(weights, reversal_mask)`` where ``weights`` is ``(N,)`` float and ``reversal_mask``
        is ``(N,)`` bool marking reversal / sudden-accel samples.
    """
    n = target.shape[0]
    weights = np.ones(n, dtype=np.float64)
    if n < 3:
        return weights, np.zeros(n, dtype=bool)

    tvel = np.gradient(target, step_dt)
    taccel = np.gradient(tvel, step_dt)

    moving = np.abs(tvel) > spec.move_eps

    # movement-start = rising edge of `moving`
    starts = np.zeros(n, dtype=bool)
    starts[1:] = moving[1:] & ~moving[:-1]
    starts[0] = moving[0]

    # transient window after each start
    win = max(1, int(round(spec.transient_window_s / step_dt)))
    transient = np.zeros(n, dtype=bool)
    start_idx = np.flatnonzero(starts)
    for s in start_idx:
        transient[s : min(n, s + win)] = True

    # reversal = sign change of target velocity while moving on at least one side
    sign = np.sign(tvel)
    reversal = np.zeros(n, dtype=bool)
    reversal[1:] = (sign[1:] * sign[:-1] < 0) & (np.abs(tvel[1:]) > spec.move_eps)
    # sudden acceleration spikes
    accel_spike = np.abs(taccel) > spec.accel_thresh
    reversal_mask = reversal | accel_spike

    weights[transient] *= spec.transient_weight
    weights[reversal_mask] *= spec.reversal_weight
    return weights, reversal_mask


def _weighted_mse(err: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean-squared error along the last axis. ``err`` is ``(..., N)``."""
    w = weights / np.sum(weights)
    return np.sum((err**2) * w, axis=-1)


def compute_scores(
    sim_pos: np.ndarray,
    sim_vel: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    weights: np.ndarray,
    reversal_mask: np.ndarray,
    score_mode: str = "position_only",
    spike_percentile: float = 99.0,
) -> dict[str, np.ndarray]:
    """Compute the score and its components for one or more simulated trajectories.

    Args:
        sim_pos: Simulated positions. Shape ``(E, N)`` or ``(N,)``.
        sim_vel: Simulated velocities. Same shape as ``sim_pos``.
        ref_pos: Real measured positions. Shape ``(N,)``.
        ref_vel: Real measured velocities. Shape ``(N,)``.
        weights: Per-step weights. Shape ``(N,)``.
        reversal_mask: Boolean reversal mask. Shape ``(N,)``.
        score_mode: One of :data:`SCORE_MODES`.
        spike_percentile: Percentile of ``|position error|`` used as the spike term in the score
            (default 99). Robust to a single unavoidable step-instant that ``max`` would latch onto.

    Returns:
        Dict of metric name -> array shaped ``(E,)`` (or scalar for ``(N,)`` input).
    """
    if score_mode not in _TRACKING_WEIGHTS:
        raise ValueError(f"Unknown score_mode '{score_mode}'. Expected one of {SCORE_MODES}.")
    w_pos, w_vel = _TRACKING_WEIGHTS[score_mode]

    e_pos = sim_pos - ref_pos
    e_vel = sim_vel - ref_vel
    abs_e_pos = np.abs(e_pos)

    weighted_pos_mse = _weighted_mse(e_pos, weights)
    weighted_vel_mse = _weighted_mse(e_vel, weights)
    max_abs_pos = np.max(abs_e_pos, axis=-1)
    # robust worst-case spike: high percentile of |error| instead of the raw max
    spike_pos_err = np.percentile(abs_e_pos, spike_percentile, axis=-1)

    tracking = w_pos * weighted_pos_mse + w_vel * weighted_vel_mse
    # spike_pos_err is intentionally NOT in the score (diagnostic only): penalizing worst-case
    # spikes rewarded timid/lagging/low-amplitude fits over ones that track the motion.
    score = tracking

    # reversal-only position MSE (used to decide whether to unlock friction in pass 2)
    if np.any(reversal_mask):
        rev_w = weights * reversal_mask
        reversal_pos_mse = _weighted_mse(e_pos, rev_w)
    else:
        reversal_pos_mse = np.zeros_like(np.atleast_1d(max_abs_pos))
        if np.ndim(max_abs_pos) == 0:
            reversal_pos_mse = np.float64(0.0)

    return {
        "score": score,
        "tracking": tracking,
        "weighted_pos_mse": weighted_pos_mse,
        "weighted_vel_mse": weighted_vel_mse,
        "spike_pos_err": spike_pos_err,
        "max_abs_pos": max_abs_pos,
        "reversal_pos_mse": reversal_pos_mse,
    }
