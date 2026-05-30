# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply per-environment DCMotor parameters to a single actuated joint.

Each environment in the vectorized scene holds a *different* parameter vector but replays the
*same* trajectory. This module writes a batch of parameter values into both the explicit
:class:`~isaaclab.actuators.DCMotor` model tensors and the physics solver (where required).

Field coupling:

* ``effort_limit`` and ``effort_limit_sim`` share their value.
* ``velocity_limit`` shapes the DCMotor torque-speed curve only. ``velocity_limit_sim`` (the hard
  PhysX solver speed cap) is decoupled and held at a fixed high value by the env, so the search
  cannot abuse it as a brick-wall cap that fakes the real servo's reversal lag.
"""

from __future__ import annotations

import torch

# Parameters searched in pass 1 (friction locked at 0).
PASS1_PARAMS = ("stiffness", "damping", "velocity_limit", "effort_limit", "saturation_effort", "armature")
# Friction triple, searched only in pass 2.
FRICTION_PARAMS = ("friction", "dynamic_friction", "viscous_friction")
# All tunable parameters.
ALL_PARAMS = PASS1_PARAMS + FRICTION_PARAMS


def ensure_tensor_saturation(actuator) -> None:
    """Promote ``DCMotor._saturation_effort`` from a scalar to a per-(env, joint) tensor.

    The stock model stores ``saturation_effort`` as a Python float. To tune it per environment
    we replace it with a tensor broadcastable against the velocity/effort limit tensors.
    """
    sat = actuator._saturation_effort
    if not torch.is_tensor(sat):
        actuator._saturation_effort = torch.full_like(actuator.velocity_limit, float(sat))


def apply_params(
    robot,
    actuator_name: str,
    joint_idx: int,
    env_ids: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> None:
    """Write a batch of DCMotor parameters for one joint into the actuator model and the sim.

    Args:
        robot: The articulation asset.
        actuator_name: Key of the actuator group in ``robot.actuators``.
        joint_idx: Global joint index of the joint being tuned.
        env_ids: Environment indices to update. Shape ``(E,)``.
        params: Mapping of parameter name -> value tensor shaped ``(E,)``.
    """
    actuator = robot.actuators[actuator_name]
    ensure_tensor_saturation(actuator)

    j0, j1 = joint_idx, joint_idx + 1
    jslice = slice(j0, j1)
    joint_ids = [joint_idx]

    def col(name: str) -> torch.Tensor:
        return params[name].to(robot.device).reshape(-1, 1)

    # --- PD gains (used by the explicit actuator model only) ---
    if "stiffness" in params:
        actuator.stiffness[env_ids, jslice] = col("stiffness")
    if "damping" in params:
        actuator.damping[env_ids, jslice] = col("damping")

    # --- saturation effort (stall torque of the DC motor model) ---
    if "saturation_effort" in params:
        actuator._saturation_effort[env_ids, jslice] = col("saturation_effort")

    # --- effort limit (continuous torque); effort_limit_sim shares the value ---
    if "effort_limit" in params:
        v = col("effort_limit")
        actuator.effort_limit[env_ids, jslice] = v
        actuator.effort_limit_sim[env_ids, jslice] = v
        robot.write_joint_effort_limit_to_sim(v, joint_ids=joint_ids, env_ids=env_ids)

    # --- velocity limit: shapes the DCMotor torque-speed curve ONLY ---
    # velocity_limit_sim (the hard PhysX solver cap) is intentionally NOT written here. It is held
    # at a fixed high/physical value by the env (see ActuatorTuningEnv) so the search cannot abuse
    # it as a brick-wall speed cap. Only the torque-speed-curve no-load speed is tuned.
    if "velocity_limit" in params:
        actuator.velocity_limit[env_ids, jslice] = col("velocity_limit")

    # --- armature (physics solver parameter) ---
    if "armature" in params:
        v = col("armature")
        actuator.armature[env_ids, jslice] = v
        robot.write_joint_armature_to_sim(v, joint_ids=joint_ids, env_ids=env_ids)

    # --- friction triple ---
    # PhysX requires static friction >= dynamic friction. The search samples these coefficients
    # independently, so clamp dynamic down to static here (per env) to keep every batch valid.
    fric = {k: params[k] for k in FRICTION_PARAMS if k in params}
    if fric:
        static = col("friction") if "friction" in fric else actuator.friction[env_ids, jslice]
        dyn = col("dynamic_friction") if "dynamic_friction" in fric else actuator.dynamic_friction[env_ids, jslice]
        dyn = torch.minimum(dyn, static)
        vis = col("viscous_friction") if "viscous_friction" in fric else None
        actuator.friction[env_ids, jslice] = static
        actuator.dynamic_friction[env_ids, jslice] = dyn
        if vis is not None:
            actuator.viscous_friction[env_ids, jslice] = vis
        robot.write_joint_friction_coefficient_to_sim(
            static,
            joint_dynamic_friction_coeff=dyn,
            joint_viscous_friction_coeff=vis,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

    # Recompute the velocity at which the torque-speed curve crosses the continuous-torque limit.
    # Depends on velocity_limit, effort_limit, and saturation_effort, so refresh after any change.
    actuator._vel_at_effort_lim[env_ids, jslice] = actuator.velocity_limit[env_ids, jslice] * (
        1.0 + actuator.effort_limit[env_ids, jslice] / actuator._saturation_effort[env_ids, jslice]
    )
