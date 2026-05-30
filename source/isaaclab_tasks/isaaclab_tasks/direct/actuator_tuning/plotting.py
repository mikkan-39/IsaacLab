# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay plots of target vs. real vs. simulated joint trajectories.

A scalar score can hide "same MSE but wrong feel" failures, so we always render overlays for
the top-k candidates. Uses the non-interactive ``Agg`` backend so it works headless.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_overlay(
    path: str,
    t_rel,
    target,
    ref_pos,
    sim_pos,
    *,
    title: str = "",
    metrics: dict | None = None,
) -> str:
    """Save a two-panel overlay plot (trajectory + position error).

    Args:
        path: Output PNG path.
        t_rel: Time axis (seconds), shape ``(N,)``.
        target: Commanded trajectory, shape ``(N,)``.
        ref_pos: Real measured trajectory, shape ``(N,)``.
        sim_pos: Simulated trajectory, shape ``(N,)``.
        title: Optional plot title.
        metrics: Optional dict of scalar metrics to annotate.

    Returns:
        The output path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[3, 1])

    ax0.plot(t_rel, target, label="target (cmd)", color="tab:gray", linewidth=1.0, linestyle="--")
    ax0.plot(t_rel, ref_pos, label="real", color="tab:blue", linewidth=1.5)
    ax0.plot(t_rel, sim_pos, label="sim", color="tab:orange", linewidth=1.5, alpha=0.9)
    ax0.set_ylabel("position (rad)")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    full_title = title
    if metrics:
        annot = "  ".join(
            f"{k}={metrics[k]:.4g}"
            for k in ("score", "weighted_pos_mse", "max_abs_pos", "reversal_pos_mse")
            if k in metrics
        )
        full_title = f"{title}\n{annot}" if title else annot
    if full_title:
        ax0.set_title(full_title, fontsize=10)

    err = sim_pos - ref_pos
    ax1.plot(t_rel, err, color="tab:red", linewidth=1.0)
    ax1.axhline(0.0, color="k", linewidth=0.5)
    ax1.set_ylabel("sim - real (rad)")
    ax1.set_xlabel("time (s)")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
