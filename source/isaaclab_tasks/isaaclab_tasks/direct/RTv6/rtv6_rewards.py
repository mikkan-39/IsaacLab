# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Weighted sum of RTv5 rough rewards (``weight * f(env)`` per term).

Mirrors :class:`isaaclab_tasks.manager_based.RTv5.rough_env_cfg.RTv5Rewards` active terms.
The environment multiplies each term by ``step_dt`` before summing (matches :class:`~isaaclab.managers.RewardManager`).
"""
from __future__ import annotations

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rtv6_constants import AMPLITUDE_LIMIT, NUM_RIGHT_LEG_JOINTS, OFFSET_LIMIT, PHASE_OFFSET_LIMIT
from .rtv6_feet_clearance import feet_clearance_capped


def gait_amplitude_out_of_bounds_penalty(env) -> torch.Tensor:
    """Per-joint L1 violation (rad) when scaled policy amplitude is outside [mins, AMPLITUDE_LIMIT]."""
    params = env.action_manager.action.view(env.num_envs, NUM_RIGHT_LEG_JOINTS, 3)
    amp_rad = params[..., 0] * AMPLITUDE_LIMIT
    mins = env._gait_ctrl._amplitude_minimums
    below = torch.clamp(mins - amp_rad, min=0.0)
    above = torch.clamp(amp_rad - AMPLITUDE_LIMIT, min=0.0)
    return (below + above).abs().sum(dim=-1)


def gait_phase_over_limit_penalty(env) -> torch.Tensor:
    """Per-joint L1 excess (rad) when |policy phase| > 1 before scaling."""
    params = env.action_manager.action.view(env.num_envs, NUM_RIGHT_LEG_JOINTS, 3)
    raw_phase = params[..., 1]
    excess_norm = torch.clamp(raw_phase.abs() - 1.0, min=0.0)
    return (excess_norm * PHASE_OFFSET_LIMIT).abs().sum(dim=-1)


def gait_offset_over_limit_penalty(env) -> torch.Tensor:
    """Per-joint L1 excess (rad) when |policy offset| > 1 before scaling."""
    params = env.action_manager.action.view(env.num_envs, NUM_RIGHT_LEG_JOINTS, 3)
    raw_offset = params[..., 2]
    excess_norm = torch.clamp(raw_offset.abs() - 1.0, min=0.0)
    return (excess_norm * OFFSET_LIMIT).abs().sum(dim=-1)


# Names must match keys in :func:`compute_rtv6_rough_reward_terms` (logged as ``Episode_Reward/<name>``).
REWARD_TERM_NAMES: tuple[str, ...] = (
    "alive_reward",
    "termination_penalty",
    "track_lin_vel",
    "track_ang_vel",
    "gait_contact",
    # "feet_clearance",
    "lin_vel_z",
    "base_pos",
    "base_ang_vel",
    # "actions_cost_diff",
    "gait_amp_out_of_bounds",
    "gait_phase_over_limit",
    "gait_offset_over_limit",
    "speed_cost",
    "torque_cost",
    "torque_cost_feet",
    "joint_deviation_hip_spread",
    "joint_deviation_hip_rotate",
    "joint_deviation_feet_main",
    "joint_deviation_feet_secondary",
)


def compute_rtv6_rough_reward_terms(env) -> dict[str, torch.Tensor]:
    """Per-term reward before ``step_dt`` scaling (``weight * f(env)`` each).
    SceneEntityCfg instances are resolved in :meth:`RTv6RoughEnv._init_rtv6_after_physics`.
    """
    e = env
    return {
        "alive_reward": 0.5 * mdp.is_alive(e),
        "termination_penalty": -200.0 * mdp.is_terminated(e),
        "track_lin_vel": 5.0
        * mdp.track_lin_vel_xy_yaw_frame_exp(e, command_name="base_velocity", std=0.1, std_y=0.2),
        "track_ang_vel": 3.0 * mdp.track_ang_vel_z_world_exp(e, command_name="base_velocity", std=0.25),
        # "stand_still": -0.4 * mdp.stand_still_joint_deviation_l1(
        #     e,
        #     command_name="base_velocity",
        #     command_threshold=0.02,
        #     asset_cfg=e._r6_speed_asset,
        # ),
        "gait_contact": 3.0 * mdp.contact_gating_reward(
            e,
            command_name="base_velocity",
            sensor_cfg=e._r6_gait_contact_sensor,
            stance_ratio=0.6,
        ),
        # "feet_clearance": 5.0
        # * feet_clearance_capped(
        #     e,
        #     sensor_cfg=e._r6_gait_contact_sensor,
        #     asset_cfg=e._r6_feet_clearance_asset,
        #     target_height=0.05,
        #     min_air_time=0.02,
        # ),
        "lin_vel_z": -0.05 * mdp.lin_vel_z_l2(e, asset_cfg=e._r6_lin_vel_z_asset),
        "base_pos": -3.0 * mdp.flat_orientation_l2(e, asset_cfg=e._r6_base_flat_asset),
        "base_ang_vel": -0.25 * mdp.ang_vel_xy_l2(e, asset_cfg=e._r6_base_ang_asset),
        # "actions_cost_diff": -0.02 * mdp.action_rate_l2(e),
        "gait_amp_out_of_bounds": -2.0 * gait_amplitude_out_of_bounds_penalty(e),
        "gait_phase_over_limit": -0.5 * gait_phase_over_limit_penalty(e),
        "gait_offset_over_limit": -0.5 * gait_offset_over_limit_penalty(e),
        "speed_cost": -3.0e-4 * mdp.joint_vel_l2(e, asset_cfg=e._r6_speed_asset),
        "torque_cost": -1.5e-5 * mdp.joint_torques(e, asset_cfg=e._r6_torque_asset),
        "torque_cost_feet": -5.0e-4 * mdp.joint_torques(e, asset_cfg=e._r6_torque_feet_asset),
        # "joint_deviation_l1": -0.01 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_speed_asset),
        # "dof_limits": -1.0 * mdp.joint_pos_limits(e, asset_cfg=e._r6_dof_limits_asset),
        # "dof_limits_knees": -5.0 * mdp.joint_pos_limits(e, asset_cfg=e._r6_dof_limits_knees_asset),
        "joint_deviation_hip_spread": -0.5 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_hip_spread),
        "joint_deviation_hip_rotate": -0.15 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_hip_rotate),
        "joint_deviation_feet_main": -0.05 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_feet_main),
        "joint_deviation_feet_secondary": -0.2 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_feet_secondary),
        # "joint_deviation_arms": -0.2 * mdp.joint_deviation_l1(e, asset_cfg=e._r6_arms),
    }


def compute_rtv6_rough_reward(env) -> torch.Tensor:
    """Sum of :func:`compute_rtv6_rough_reward_terms` (caller applies ``* step_dt``)."""
    terms = compute_rtv6_rough_reward_terms(env)
    return torch.stack(list(terms.values()), dim=0).sum(dim=0)
