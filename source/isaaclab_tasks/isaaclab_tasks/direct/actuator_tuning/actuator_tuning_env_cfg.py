# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the actuator parameter tuning environment."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import RT_CFG

# Default recording shipped alongside the environment.
_DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "servo_recording.csv")


@configclass
class ActuatorTuningEnvCfg(DirectRLEnvCfg):
    """Replays a recorded servo trajectory on a fixed-base RT robot to tune DCMotor parameters.

    The base link is fully fixed in place. One joint (``joint_name``) replays the recorded
    ``target_rad`` at the control rate; every other joint is frozen at its default pose. Each
    parallel environment can hold a different DCMotor parameter vector (see the tuning scripts).
    """

    # -- replay / scoring settings --
    joint_name: str = "base_link_to_Neck_revolute"
    """Exact USD joint name to drive with the recorded targets."""

    override_joint_pos_limits: tuple[float, float] | None = (-math.pi, math.pi)
    """If set, overwrite every joint's position limits with this (low, high) range in radians.

    The RT USD ships tight angular limits on its joints. For trajectory replay/identification we
    widen them (default +/- pi) so the recorded targets are never clipped by the solver. Set to
    None to keep the USD limits.
    """

    csv_path: str = _DEFAULT_CSV
    """Path to the servo recording CSV."""

    actuator_name: str = "ST3215-HS"
    """Key of the DCMotor actuator group in the robot config."""

    control_hz: float = 50.0
    """Control/replay rate in Hz (lower than the recording rate)."""

    solver_velocity_limit: float = 20.0
    """Fixed PhysX solver velocity cap (rad/s) for the driven joint, decoupled from the tuned
    ``velocity_limit``.

    The tuned ``velocity_limit`` only shapes the DCMotor torque-speed curve; this hard cap is held
    high (well above the real servo's measured peak speed) so the search cannot use a low solver
    cap to fake reversal lag. Set above your measured maximum joint speed with margin.
    """

    max_duration_s: float | None = None
    """Optional cap on replayed duration (seconds). None replays the full recording."""

    score_mode: str = "position_only"
    """Scoring mode: ``position_only`` | ``position_heavy`` | ``balanced``."""

    spike_percentile: float = 99.0
    """Percentile of |position error| used as the spike term (diagnostic only; not in the score)."""

    ref_lag_steps: float = 1.0
    """Advance the *real* reference (ref_pos/ref_vel) by this many control steps before scoring.

    The real servo has a measured transport lag (~4-16 ms; ~1 step at 50 Hz). Rather than adding a
    matching dead-time to the simulated command, we time-shift the recorded real signal earlier by
    this amount so the comparison is about the actuator *dynamics*, not the (known, fixed) comms
    delay. NOTE: this removes the delay from the *comparison*, not from the simulated model -- for
    sim2real deployment, re-introduce the delay (a 1-step command buffer) on the deployed sim.
    """

    segment_len_s: float = 10.0
    """Duration of each independent excitation segment in the recording (steps/sines/sawtooths)."""

    lag_weight: float = 2.0
    """Weight of the mean per-segment sim-vs-real lag (seconds) added to the score. 0 disables it."""

    lag_max_s: float = 0.3
    """Half-width of the per-segment lag cross-correlation search window (seconds)."""

    # -- error-weighting (transients / reversals emphasized) --
    weight_move_eps: float = 0.05
    weight_transient_window_s: float = 0.3
    weight_transient: float = 2.0
    weight_reversal: float = 2.0
    weight_accel_thresh: float = 5.0

    # -- standard direct-env settings --
    decimation: int = 4
    episode_length_s: float = 1.0e6  # termination is driven by replay length, not this
    action_space: int = 1  # dummy; replay is internal
    observation_space: int = 1  # dummy
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=4)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=1.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot is spawned manually in the env's _setup_scene (cartpole-style)
    robot_cfg: ArticulationCfg = MISSING

    def __post_init__(self) -> None:
        # control rate => physics dt: step_dt = decimation * sim.dt = 1 / control_hz
        self.sim.dt = 1.0 / (self.control_hz * self.decimation)
        self.sim.render_interval = self.decimation

        # fixed-base RT robot with a plain DCMotor and a known (zero) friction baseline
        robot = RT_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore
        robot.spawn.articulation_props.fix_root_link = True
        actuators = copy.deepcopy(robot.actuators)
        act = actuators[self.actuator_name]
        act.friction = 0.0
        act.dynamic_friction = 0.0
        act.viscous_friction = 0.0
        robot.actuators = actuators
        self.robot_cfg = robot
