# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the actuator parameter tuning environment."""

from __future__ import annotations

import copy
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
    joint_name: str = MISSING
    """Exact USD joint name to drive with the recorded targets."""

    csv_path: str = _DEFAULT_CSV
    """Path to the servo recording CSV."""

    actuator_name: str = "ST3215-HS"
    """Key of the DCMotor actuator group in the robot config."""

    control_hz: float = 50.0
    """Control/replay rate in Hz (lower than the recording rate)."""

    max_duration_s: float | None = None
    """Optional cap on replayed duration (seconds). None replays the full recording."""

    score_mode: str = "position_only"
    """Scoring mode: ``position_only`` | ``position_heavy`` | ``balanced``."""

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
