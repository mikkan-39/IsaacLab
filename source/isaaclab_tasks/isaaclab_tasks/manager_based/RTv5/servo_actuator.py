from __future__ import annotations

"""Identified IdealPD servo actuator for the RT robot (MLP + IdealPD workflow).

The PD gains were identified by the MLP+IdealPD tuning workflow
(``scripts/tools/actuator_tuning`` -> ``logs/actuator_tuning/pd01/best_grid.json``). The
:class:`MlpJointPositionAction` term supplies the (delay-free) MLP position setpoint and this PD
tracks it, re-creating the servo's transport lag. ``effort_limit`` / ``velocity_limit_sim`` match the
tuning env's IdealPD defaults so the identified gains behave identically here.
"""

import copy

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# Identified gains (logs/actuator_tuning/pd01/best_grid.json).
SERVO_STIFFNESS = 212.408
SERVO_DAMPING = 2.6585
SERVO_ARMATURE = 0.012599
# Match the tuning env's IdealPD defaults (ideal_pd_effort_limit=10.0, solver_velocity_limit=20.0).
SERVO_EFFORT_LIMIT = 10.0
SERVO_VELOCITY_LIMIT_SIM = 20.0


def apply_identified_servo(robot_cfg: ArticulationCfg, actuator_key: str = "ST3215-HS") -> ArticulationCfg:
    """Replace the robot's actuator group with the identified IdealPD servo model (in place).

    Returns the same ``robot_cfg`` for convenience.
    """
    actuators = copy.deepcopy(robot_cfg.actuators)
    base = actuators[actuator_key]
    actuators[actuator_key] = IdealPDActuatorCfg(
        joint_names_expr=list(base.joint_names_expr),
        stiffness=SERVO_STIFFNESS,
        damping=SERVO_DAMPING,
        armature=SERVO_ARMATURE,
        effort_limit=SERVO_EFFORT_LIMIT,
        effort_limit_sim=1.0e9,
        velocity_limit_sim=SERVO_VELOCITY_LIMIT_SIM,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    robot_cfg.actuators = actuators
    return robot_cfg
