# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and resample a real-servo recording for replay in the tuning environment.

The CSV is expected to have the header::

    timestamp,target_rad,position_rad,error_rad,speed_rad_s

Recorded at a high rate (50-1000 Hz). For replay we downsample to the control rate
(default 50 Hz) using a zero-order hold (nearest past sample), without interpolation.
"""

from __future__ import annotations

import csv
import numpy as np
from dataclasses import dataclass


@dataclass
class ServoTrajectory:
    """A control-rate-resampled servo trajectory.

    All arrays are 1-D with the same length ``num_steps`` and are indexed by control step.
    """

    t_rel: np.ndarray
    """Control-step times relative to the first sample (seconds)."""

    target: np.ndarray
    """Commanded joint position at each control step (rad)."""

    ref_pos: np.ndarray
    """Measured real joint position at each control step (rad)."""

    ref_vel: np.ndarray
    """Measured real joint speed at each control step (rad/s)."""

    step_dt: float
    """Control time-step (seconds)."""

    @property
    def num_steps(self) -> int:
        return int(self.target.shape[0])

    @property
    def duration(self) -> float:
        return float(self.t_rel[-1] - self.t_rel[0]) if self.num_steps else 0.0


def _read_csv(csv_path: str) -> dict[str, np.ndarray]:
    """Read the raw recording into column arrays."""
    cols: dict[str, list[float]] = {
        "timestamp": [],
        "target_rad": [],
        "position_rad": [],
        "error_rad": [],
        "speed_rad_s": [],
    }
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV '{csv_path}' is missing required columns {missing}. Found: {reader.fieldnames}"
            )
        for row in reader:
            # skip malformed / empty trailing rows
            if not row.get("timestamp"):
                continue
            for c in cols:
                cols[c].append(float(row[c]))
    return {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}


def load_servo_trajectory(
    csv_path: str,
    control_hz: float,
    *,
    max_duration_s: float | None = None,
    ref_lag_steps: float = 0.0,
) -> ServoTrajectory:
    """Load a recording and resample it onto a uniform control-rate grid.

    Args:
        csv_path: Path to the recording CSV.
        control_hz: Target replay/control rate (must be lower than the recording rate).
        max_duration_s: Optional cap on the replayed duration (seconds from the start).
        ref_lag_steps: Advance the real signals (``ref_pos``/``ref_vel``) earlier by this many
            control steps to compensate for the real servo's measured transport lag. The commanded
            ``target`` is left untouched. Fractional values are linearly interpolated. 0 = no shift.

    Returns:
        A :class:`ServoTrajectory` resampled with a zero-order hold (nearest past sample).
    """
    raw = _read_csv(csv_path)
    ts = raw["timestamp"]
    if ts.size < 2:
        raise ValueError(f"CSV '{csv_path}' has too few rows ({ts.size}).")

    # absolute-time frame: relative to the first timestamp
    t0 = ts[0]
    t_rel_raw = ts - t0

    total_dur = float(t_rel_raw[-1])
    if max_duration_s is not None:
        total_dur = min(total_dur, float(max_duration_s))

    step_dt = 1.0 / float(control_hz)
    num_steps = int(np.floor(total_dur / step_dt)) + 1
    t_ctrl = np.arange(num_steps, dtype=np.float64) * step_dt

    # zero-order hold: for each control time, take the last raw sample with t_rel <= t_ctrl.
    # np.searchsorted gives the insertion index; subtract 1 for the last <= value.
    idx = np.searchsorted(t_rel_raw, t_ctrl, side="right") - 1
    idx = np.clip(idx, 0, t_rel_raw.size - 1)

    target = raw["target_rad"][idx].copy()
    ref_pos = raw["position_rad"][idx].copy()
    ref_vel = raw["speed_rad_s"][idx].copy()

    # advance the real signals to remove the servo's (known, fixed) transport lag; target unchanged
    if ref_lag_steps:
        src = np.clip(np.arange(num_steps, dtype=np.float64) + float(ref_lag_steps), 0.0, num_steps - 1)
        lo = np.floor(src).astype(int)
        hi = np.clip(lo + 1, 0, num_steps - 1)
        frac = src - lo
        ref_pos = ref_pos[lo] * (1.0 - frac) + ref_pos[hi] * frac
        ref_vel = ref_vel[lo] * (1.0 - frac) + ref_vel[hi] * frac

    return ServoTrajectory(
        t_rel=t_ctrl,
        target=target,
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        step_dt=step_dt,
    )
