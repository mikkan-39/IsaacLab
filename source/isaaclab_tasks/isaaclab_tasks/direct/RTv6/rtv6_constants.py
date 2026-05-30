# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RTv6-only constants (decoupled from manager-based RTv5 task configs)."""

import math

# Mirrors RTv5 `controllableJointsRegex` — used for domain-randomization events and some rewards.
CONTROLLABLE_JOINTS_REGEX = r"^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL)).*$"

# --- Gait clock (shared by actuation targets and gait_phase observation) ---
GAIT_FREQ = 1

# --- Sinusoidal target limits (policy outputs are in [-1, 1] before scaling) ---
AMPLITUDE_LIMIT = 1.0
OFFSET_LIMIT = 0.3
PHASE_OFFSET_LIMIT = math.pi

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

# Per left joint (same order as RIGHT_LEG_JOINT_NAMES):
# - invert_left_target: negate (offset + amp*sin) before adding default_L
# - invert_left_amplitude: negate amp*sin only on the left leg (offset unchanged)
INVERT_LEFT_TARGET: tuple[bool, ...] = (
    True,
    False,
    True,
    True,
    False,
    False,
)
INVERT_LEFT_AMPLITUDE: tuple[bool, ...] = (
    False,
    False,
    False,
    False,
    False,
    False,
)

# (right_name, left_name, invert_left_target, invert_left_amplitude)
LEG_JOINT_PAIRS: tuple[tuple[str, str, bool, bool], ...] = tuple(
    zip(
        RIGHT_LEG_JOINT_NAMES,
        LEFT_LEG_JOINT_NAMES,
        INVERT_LEFT_TARGET,
        INVERT_LEFT_AMPLITUDE,
    )
)

NUM_RIGHT_LEG_JOINTS = len(RIGHT_LEG_JOINT_NAMES)

# TEMP: ankle = sign * (knee + hip_bulk) after soft-limit clamp (see :meth:`RTv6SinusoidalGaitController.apply_to_sim`).
ANKLE_PARALLEL_FROM_HIP_KNEE = True
ANKLE_PARALLEL_SIGN_RIGHT = 1.0
ANKLE_PARALLEL_SIGN_LEFT = -1.0
HIP_BULK_JOINT_INDEX = RIGHT_LEG_JOINT_NAMES.index("HipBulkR_to_HipR_revolute")
KNEE_JOINT_INDEX = RIGHT_LEG_JOINT_NAMES.index("HipR_to_TibiaR_revolute")
ANKLE_JOINT_INDEX = RIGHT_LEG_JOINT_NAMES.index("TibiaR_to_FootJointR_revolute")

# Per-joint amplitude floor (rad), same order as RIGHT_LEG_JOINT_NAMES. Applied after scaling to AMPLITUDE_LIMIT.
AMPLITUDE_MINIMUMS: tuple[float, ...] = (
    0.0,  # base_link_to_RightHipBracket_revolute
    0.2,  # RightHipBracket_to_HipBulkR_revolute
    0.15,  # HipBulkR_to_HipR_revolute
    0.3,  # HipR_to_TibiaR_revolute
    0.15,  # TibiaR_to_FootJointR_revolute
    0.2,  # FootJointR_to_RightFoot_revolute
)

OFFSETS_BASELINE: tuple[float, ...] = (
    0.0,  # base_link_to_RightHipBracket_revolute +outward
    -0.05,  # RightHipBracket_to_HipBulkR_revolute -outward
    0.1,  # HipBulkR_to_HipR_revolute            +forward
    0.0,  # HipR_to_TibiaR_revolute              -bend
    -0.05,  # TibiaR_to_FootJointR_revolute       -forward
    0.05,  # FootJointR_to_RightFoot_revolute:    +inward
)

PHASE_OFFSETS_BASELINE: tuple[float, ...] = (
    0,  # base_link_to_RightHipBracket_revolute
    0,  # RightHipBracket_to_HipBulkR_revolute
    math.pi/2,  # HipBulkR_to_HipR_revolute
    0,  # HipR_to_TibiaR_revolute
    math.pi/2,  # TibiaR_to_FootJointR_revolute
    math.pi,  # FootJointR_to_RightFoot_revolute
)

# Seconds after episode start before swing amplitude turns on (offset still active). Same order as right leg.
START_TIME: tuple[float, ...] = (
    1/GAIT_FREQ * 0.0,  # base_link_to_RightHipBracket_revolute
    1/GAIT_FREQ * 0.0,  # RightHipBracket_to_HipBulkR_revolute
    1/GAIT_FREQ * 1.25,  # HipBulkR_to_HipR_revolute
    1/GAIT_FREQ * 1.0,  # HipR_to_TibiaR_revolute
    1/GAIT_FREQ * 1.25,  # TibiaR_to_FootJointR_revolute
    1/GAIT_FREQ * 0.0,  # FootJointR_to_RightFoot_revolute
)

if len(AMPLITUDE_MINIMUMS) != NUM_RIGHT_LEG_JOINTS:
    raise ValueError("AMPLITUDE_MINIMUMS must have one entry per RIGHT_LEG_JOINT_NAMES joint.")
if any(m < 0.0 or m > AMPLITUDE_LIMIT for m in AMPLITUDE_MINIMUMS):
    raise ValueError("Each AMPLITUDE_MINIMUMS entry must be in [0, AMPLITUDE_LIMIT].")
if len(INVERT_LEFT_TARGET) != NUM_RIGHT_LEG_JOINTS or len(INVERT_LEFT_AMPLITUDE) != NUM_RIGHT_LEG_JOINTS:
    raise ValueError("INVERT_LEFT_* tuples must have one entry per RIGHT_LEG_JOINT_NAMES joint.")
if len(START_TIME) != NUM_RIGHT_LEG_JOINTS:
    raise ValueError("START_TIME must have one entry per RIGHT_LEG_JOINT_NAMES joint.")
if any(t < 0.0 for t in START_TIME):
    raise ValueError("Each START_TIME entry must be >= 0.")

ACTIONS_PER_JOINT = 3  # amplitude, phase_offset, offset (in that order)
GAIT_ACTION_DIM = NUM_RIGHT_LEG_JOINTS * ACTIONS_PER_JOINT

# --- Foot clearance raycast (sole patch under each foot link) ---
# Distance from foot link origin to lowest sole point, along foot link -Z (tune if reward is biased).
FOOT_SOLE_Z_OFFSET_B = -0.027
FOOTPRINT_HALF_LENGTH = 0.055
FOOTPRINT_HALF_WIDTH = 0.030
FOOTPRINT_RAY_RESOLUTION = 0.025
FOOT_RAY_CAST_LIFT = 0.02
FOOT_RAY_MAX_DIST = 0.5

# Observation corruption
PROJECTED_GRAVITY_OBS_NOISE_STD = 0.025
