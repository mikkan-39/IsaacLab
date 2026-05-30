# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-logic tests for the actuator tuning environment (no simulation app required).

Covers the CSV zero-order-hold downsampling and the trajectory-matching metrics.
"""

import csv
import numpy as np
import os
import tempfile

import pytest

from isaaclab_tasks.direct.actuator_tuning import metrics
from isaaclab_tasks.direct.actuator_tuning.csv_replay import load_servo_trajectory


def _write_csv(path, timestamps, target, position, speed):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "target_rad", "position_rad", "error_rad", "speed_rad_s"])
        for t, tg, p, s in zip(timestamps, target, position, speed):
            w.writerow([t, tg, p, tg - p, s])


def test_zero_order_hold_downsample():
    # 1000 Hz recording, downsample to 50 Hz (every 20th sample held)
    n = 1000
    dt_raw = 0.001
    t0 = 1780085388.0
    ts = t0 + np.arange(n) * dt_raw
    target = np.sin(np.arange(n) * 0.01)
    position = target * 0.5
    speed = np.cos(np.arange(n) * 0.01)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.csv")
        _write_csv(path, ts, target, position, speed)
        traj = load_servo_trajectory(path, control_hz=50.0)

    # 1000 samples * 0.001 s = ~0.999 s span -> 50 control steps + 1
    assert traj.step_dt == pytest.approx(0.02)
    assert traj.num_steps == 50
    # first control step aligns with the first recorded sample
    assert traj.target[0] == pytest.approx(target[0])
    assert traj.ref_pos[0] == pytest.approx(position[0])
    # control step k=1 (t=0.02 s) holds raw index 20
    assert traj.target[1] == pytest.approx(target[20])


def test_max_duration_cap():
    n = 500
    ts = 1780085388.0 + np.arange(n) * 0.002  # 500 Hz, 1.0 s span
    vals = np.arange(n, dtype=float)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.csv")
        _write_csv(path, ts, vals, vals, vals)
        traj = load_servo_trajectory(path, control_hz=50.0, max_duration_s=0.5)
    # 0.5 s at 50 Hz -> 25 steps + 1
    assert traj.num_steps == 26
    assert traj.t_rel[-1] == pytest.approx(0.5)


def test_weights_emphasize_transients_and_reversals():
    step_dt = 0.02
    n = 200
    t = np.arange(n) * step_dt
    # a triangle-ish motion that starts moving then reverses
    target = np.concatenate([np.zeros(20), np.linspace(0, 1.0, 60), np.linspace(1.0, -1.0, 60), np.full(60, -1.0)])
    spec = metrics.WeightSpec()
    weights, reversal = metrics.compute_weights(target, step_dt, spec)

    assert weights.shape == (n,)
    # steady-state (held) regions stay at weight 1
    assert weights[-1] == pytest.approx(1.0)
    # some samples are up-weighted (transients / reversals exist)
    assert np.any(weights > 1.0)
    # a reversal is detected around the peak
    assert np.any(reversal)


def test_score_position_only_zero_when_perfect():
    n = 100
    step_dt = 0.02
    target = np.sin(np.arange(n) * 0.1)
    weights, reversal = metrics.compute_weights(target, step_dt, metrics.WeightSpec())
    ref_pos = target.copy()
    ref_vel = np.cos(np.arange(n) * 0.1)
    # perfect position tracking, but wrong velocity
    sim_pos = ref_pos.copy()
    sim_vel = np.zeros_like(ref_vel)

    out = metrics.compute_scores(sim_pos, sim_vel, ref_pos, ref_vel, weights, reversal, score_mode="position_only")
    assert out["weighted_pos_mse"] == pytest.approx(0.0, abs=1e-12)
    assert out["max_abs_pos"] == pytest.approx(0.0, abs=1e-12)
    assert out["score"] == pytest.approx(0.0, abs=1e-12)
    # velocity error is logged but does not affect the position_only score
    assert out["weighted_vel_mse"] > 0.0


def test_score_batched_shapes():
    n, e = 80, 4
    step_dt = 0.02
    target = np.sin(np.arange(n) * 0.1)
    weights, reversal = metrics.compute_weights(target, step_dt, metrics.WeightSpec())
    ref_pos = target.copy()
    ref_vel = np.cos(np.arange(n) * 0.1)
    sim_pos = np.tile(ref_pos, (e, 1)) + np.random.randn(e, n) * 0.01
    sim_vel = np.tile(ref_vel, (e, 1))

    out = metrics.compute_scores(sim_pos, sim_vel, ref_pos, ref_vel, weights, reversal, score_mode="balanced")
    assert out["score"].shape == (e,)
    assert out["max_abs_pos"].shape == (e,)


def test_invalid_score_mode():
    n = 10
    a = np.zeros(n)
    w = np.ones(n)
    with pytest.raises(ValueError):
        metrics.compute_scores(a, a, a, a, w, w.astype(bool), score_mode="nope")
