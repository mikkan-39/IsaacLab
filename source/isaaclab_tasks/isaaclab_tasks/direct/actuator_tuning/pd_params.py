# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply per-environment IdealPD (explicit PD) parameters to a single actuated joint.

Used by the actuator-tuning env when ``actuator_model == "ideal_pd"``. The joint is driven toward
the MLP-predicted position target by an :class:`~isaaclab.actuators.IdealPDActuator`; the search
tunes the PD gains (and optionally armature / effort clip / friction) so the joint's lagged response
lands on the recorded real-servo trajectory.

For an explicit actuator the PD gains and effort clip are consumed by the actuator model itself
(``compute()``), not the PhysX drive, so ``stiffness``/``damping``/``effort_limit`` are written only
to the actuator tensors. ``armature`` and the friction triple are solver parameters and are written
through to the simulation.
"""

from __future__ import annotations

import torch

# Parameters tunable for the IdealPD model.
PD_PARAMS = ("stiffness", "damping", "effort_limit", "armature")
FRICTION_PARAMS = ("friction", "dynamic_friction", "viscous_friction")
ALL_PARAMS = PD_PARAMS + FRICTION_PARAMS


def apply_params(
    robot,
    actuator_name: str,
    joint_idx: int,
    env_ids: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> None:
    """Write a batch of IdealPD parameters for one joint into the actuator model and the sim.

    Args:
        robot: The articulation asset.
        actuator_name: Key of the actuator group in ``robot.actuators``.
        joint_idx: Global joint index of the joint being tuned.
        env_ids: Environment indices to update. Shape ``(E,)``.
        params: Mapping of parameter name (subset of :data:`ALL_PARAMS`) -> value tensor ``(E,)``.
    """
    actuator = robot.actuators[actuator_name]
    jslice = slice(joint_idx, joint_idx + 1)
    joint_ids = [joint_idx]

    def col(name: str) -> torch.Tensor:
        return params[name].to(robot.device).reshape(-1, 1)

    # --- PD gains: explicit actuator consumes these directly (not the PhysX drive) ---
    if "stiffness" in params:
        actuator.stiffness[env_ids, jslice] = col("stiffness")
    if "damping" in params:
        actuator.damping[env_ids, jslice] = col("damping")

    # --- effort clip: used by IdealPDActuator._clip_effort (model-internal, no sim write) ---
    if "effort_limit" in params:
        actuator.effort_limit[env_ids, jslice] = col("effort_limit")

    # --- armature (physics solver parameter) ---
    if "armature" in params:
        v = col("armature")
        actuator.armature[env_ids, jslice] = v
        robot.write_joint_armature_to_sim(v, joint_ids=joint_ids, env_ids=env_ids)

    # --- friction triple (PhysX requires static >= dynamic; clamp before writing) ---
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
