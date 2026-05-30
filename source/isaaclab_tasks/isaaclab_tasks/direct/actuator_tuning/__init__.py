# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator parameter tuning environment.

Replays a recorded real-servo trajectory on a fixed-base RT robot and scores how well the
simulated DCMotor tracks it, for grid/sample search and regression fitting of motor parameters.
"""

import gymnasium as gym

from .actuator_tuning_env import ActuatorTuningEnv
from .actuator_tuning_env_cfg import ActuatorTuningEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Actuator-Tuning-RT-v0",
    entry_point=f"{__name__}.actuator_tuning_env:ActuatorTuningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actuator_tuning_env_cfg:ActuatorTuningEnvCfg",
    },
)
