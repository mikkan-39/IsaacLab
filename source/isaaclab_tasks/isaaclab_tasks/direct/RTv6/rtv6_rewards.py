# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Weighted sum of RTv5 rough rewards (``weight * f(env)`` per term).

The environment multiplies the return by ``step_dt`` to match :class:`~isaaclab.managers.RewardManager`.
"""

from __future__ import annotations

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def compute_rtv6_rough_reward(env) -> torch.Tensor:
    """Mirror :class:`isaaclab_tasks.manager_based.RTv5.rough_env_cfg.RTv5Rewards` (active terms only).

    SceneEntityCfg instances are resolved in :meth:`RTv6RoughEnv._init_rtv6_after_physics`.
    """
    e = env
    r = torch.zeros(e.num_envs, device=e.device)
    r += 0.5 * mdp.is_alive(e)
    r += -200.0 * mdp.is_terminated(e)
    r += 1.0 * mdp.track_lin_vel_xy_yaw_frame_exp(e, command_name="base_velocity", std=0.15)
    r += 1.0 * mdp.track_ang_vel_z_world_exp(e, command_name="base_velocity", std=0.5)
    r += 1.0 * mdp.contact_gating_reward(
        e,
        command_name="base_velocity",
        sensor_cfg=e._r6_gait_contact_sensor,
        stance_ratio=0.6,
    )
    r += -0.05 * mdp.lin_vel_z_l2(e, asset_cfg=e._r6_lin_vel_z_asset)
    r += -1.0 * mdp.flat_orientation_l2(e, asset_cfg=e._r6_base_flat_asset)
    r += -0.05 * mdp.ang_vel_xy_l2(e, asset_cfg=e._r6_base_ang_asset)
    r += -0.1 * mdp.action_rate_l2(e)
    r += -0.5 * mdp.action_clip_violation(e, clip_min=-1.57, clip_max=1.57)
    r += -3.0e-4 * mdp.joint_vel_l2(e, asset_cfg=e._r6_speed_asset)
    r += -1.5e-5 * mdp.joint_torques(e, asset_cfg=e._r6_torque_asset)
    r += -5.0e-4 * mdp.joint_torques(e, asset_cfg=e._r6_torque_feet_asset)
    r += -0.3 * mdp.joint_pos_limits(e, asset_cfg=e._r6_dof_limits_asset)
    r += -2.5e-7 * mdp.joint_acc_l2(e)
    r += -1.5 * mdp.feet_slide(e, sensor_cfg=e._r6_feet_slide_sensor, asset_cfg=e._r6_feet_slide_robot)
    r += -0.2 * mdp.joint_same_direction_deviation_penalty(e, asset_cfg=e._r6_hip_spread)
    r += -0.5 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_hip_rotate)
    r += -0.2 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_feet_main)
    r += -0.2 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_feet_secondary)
    r += -0.2 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_arms)
    return r
