# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RTv6-only constants (decoupled from manager-based RTv5 task configs)."""

import math

# Mirrors RTv5 `controllableJointsRegex` — used for domain-randomization events and some rewards.
CONTROLLABLE_JOINTS_REGEX = r"^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL)).*$"

# --- Gait clock (shared by actuation targets and gait_phase observation) ---
GAIT_FREQ = 1.25  # Hz, fixed for all envs

# --- Sinusoidal target limits (policy outputs are in [-1, 1] before scaling) ---
AMPLITUDE_LIMIT = 1.0
OFFSET_LIMIT = 0.3
PHASE_OFFSET_LIMIT = math.pi / 2

# --- Leg joints (policy controls right leg only; left is mirrored) ---
RIGHT_LEG_JOINT_NAMES: tuple[str, ...] = (
    "base_link_to_RightHipBracket_revolute",
    "RightHipBracket_to_HipBulkR_revolute",
    "HipBulkR_to_HipR_revolute",
    "HipR_to_TibiaR_revolute",
    "TibiaR_to_FootJointR_revolute",
    "FootJointR_to_RightFoot_revolute",
)

LEFT_LEG_JOINT_NAMES: tuple[str, ...] = (
    "base_link_to_LeftHipBracket_revolute",
    "LeftHipBracket_to_HipBulkL_revolute",
    "HipBulkL_to_HipL_revolute",
    "HipL_to_TibiaL_revolute",
    "TibiaL_to_FootJointL_revolute",
    "FootJointL_to_LeftFoot_revolute",
)

# (right_name, left_name, invert_left_target)
LEG_JOINT_PAIRS: tuple[tuple[str, str, bool], ...] = tuple(
    zip(
        RIGHT_LEG_JOINT_NAMES,
        LEFT_LEG_JOINT_NAMES,
        ( # Flipped left target signs
            True,
            False,
            True,
            True,
            False,
            True,
        ),
    )
)

NUM_RIGHT_LEG_JOINTS = len(RIGHT_LEG_JOINT_NAMES)

# Per-joint amplitude floor (rad), same order as RIGHT_LEG_JOINT_NAMES. Applied after scaling to AMPLITUDE_LIMIT.
AMPLITUDE_MINIMUMS: tuple[float, ...] = (
    0.05,  # base_link_to_RightHipBracket_revolute
    0.05,  # RightHipBracket_to_HipBulkR_revolute
    0.20,  # HipBulkR_to_HipR_revolute
    0.20,  # HipR_to_TibiaR_revolute
    0.05,  # TibiaR_to_FootJointR_revolute
    0.05,  # FootJointR_to_RightFoot_revolute
)

if len(AMPLITUDE_MINIMUMS) != NUM_RIGHT_LEG_JOINTS:
    raise ValueError("AMPLITUDE_MINIMUMS must have one entry per RIGHT_LEG_JOINT_NAMES joint.")
if any(m < 0.0 or m > AMPLITUDE_LIMIT for m in AMPLITUDE_MINIMUMS):
    raise ValueError("Each AMPLITUDE_MINIMUMS entry must be in [0, AMPLITUDE_LIMIT].")

ACTIONS_PER_JOINT = 3  # amplitude, phase_offset, offset (in that order)
GAIT_ACTION_DIM = NUM_RIGHT_LEG_JOINTS * ACTIONS_PER_JOINT

# Observation corruption
PROJECTED_GRAVITY_OBS_NOISE_STD = 0.025
