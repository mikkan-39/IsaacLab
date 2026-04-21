# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RTv6-only constants (decoupled from manager-based RTv5 task configs)."""

# Mirrors historical RTv5 `controllableJointsRegex` — edit here for RTv6 experiments.
CONTROLLABLE_JOINTS_REGEX = r"^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL)).*$"

GAIT_FREQ_RANGE = (1.0, 1.5)  # Hz — per-env gait clock for phase observation
