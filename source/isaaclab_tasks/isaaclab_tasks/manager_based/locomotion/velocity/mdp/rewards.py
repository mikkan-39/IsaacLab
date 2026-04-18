# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_joint_pos_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded joint target using joint position relative to default pose.

    Using relative joint position keeps the reward consistent with:
    - `joint_pos_rel` observations
    - `JointPositionActionCfg(..., use_default_offset=True)`
    """
    asset = env.scene[asset_cfg.name]

    # Command from SineJointCommand: shape (num_envs, 1) for this setup.
    target = env.command_manager.get_command(command_name)

    # Compare in the same coordinate frame as observations/actions (relative to default pose).
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Mean absolute error keeps scale stable if joint count changes later.
    error = joint_pos_rel - target
    return -torch.mean(torch.abs(error), dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def action_clip_violation(
    env: ManagerBasedRLEnv, clip_min: float = -1.0, clip_max: float = 1.0
) -> torch.Tensor:
    """Penalize actions that exceed the clip bounds.

    This function penalizes the policy for outputting actions outside the clipping range.
    Since actions are clipped before being applied, the policy doesn't inherently learn
    to stay within bounds. This penalty encourages bounded outputs.

    The penalty is computed as the sum of squared violations for each action dimension.
    """
    # Get raw actions from all action terms (before clipping/processing)
    raw_actions_list = []
    for term in env.action_manager._terms.values():
        if hasattr(term, 'raw_actions'):
            raw_actions_list.append(term.raw_actions)
    
    if not raw_actions_list:
        return torch.zeros(env.num_envs, device=env.device)
    
    raw_actions = torch.cat(raw_actions_list, dim=1)
    
    # Compute violation: amount by which actions exceed bounds
    lower_violation = torch.clamp(clip_min - raw_actions, min=0.0)  # Positive if below min
    upper_violation = torch.clamp(raw_actions - clip_max, min=0.0)  # Positive if above max
    
    # Sum of squared violations
    violation = torch.sum(torch.square(lower_violation) + torch.square(upper_violation), dim=1)
    
    return violation


def action_magnitude_l1(
    env: ManagerBasedRLEnv, action_ids: list[int] | None = None
) -> torch.Tensor:
    """Return L1 norm of actions for specified action indices.
    
    Args:
        env: The environment.
        action_ids: List of action indices to include. If None, uses all actions.
    
    Returns:
        Sum of absolute action values for specified indices. Shape: (num_envs,)
    """
    actions = env.action_manager.action
    if action_ids is not None:
        actions = actions[:, action_ids]
    return torch.sum(torch.abs(actions), dim=1)


def action_magnitude_l2(
    env: ManagerBasedRLEnv, action_ids: list[int] | None = None
) -> torch.Tensor:
    """Return L2 norm squared of actions for specified action indices.
    
    Args:
        env: The environment.
        action_ids: List of action indices to include. If None, uses all actions.
    
    Returns:
        Sum of squared action values for specified indices. Shape: (num_envs,)
    """
    actions = env.action_manager.action
    if action_ids is not None:
        actions = actions[:, action_ids]
    return torch.sum(torch.square(actions), dim=1)


def feet_step_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward feet for covering horizontal distance during the swing phase.

    Records each foot's world-frame XY position at liftoff. When the foot
    lands again, rewards the horizontal distance traveled. This directly
    incentivizes longer steps: shuffling barely moves the foot, proper
    walking swings it far.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_pos_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    buf_key = "_feet_liftoff_pos"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, foot_pos_xy.clone())
    liftoff_buf: torch.Tensor = getattr(env, buf_key)

    liftoff_buf[first_air] = foot_pos_xy[first_air]

    step_dist = torch.norm(foot_pos_xy - liftoff_buf, dim=-1)
    reward = torch.sum(step_dist.square() * first_contact.float(), dim=1)

    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05

    # update per-foot last step distance (used by feet_step_symmetry)
    dist_key = "_last_step_dist"
    if not hasattr(env, dist_key):
        setattr(env, dist_key, torch.zeros_like(step_dist))
    last_dist: torch.Tensor = getattr(env, dist_key)
    last_dist[first_contact] = step_dist[first_contact]

    return reward


def feet_step_symmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    ang_vel_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize asymmetric step distances between left and right feet, but only when moving straight.

    Uses the per-foot last step distances tracked by ``feet_step_distance``.
    Returns the absolute difference — a larger value means more asymmetry.
    Must be used with a negative weight.

    Penalizes asymmetry only when:
    - The robot is commanded to move forward (lin_vel > 0.1)
    - The robot is moving relatively straight (ang_vel < ang_vel_threshold)

    When turning (ang_vel >= ang_vel_threshold), asymmetry is allowed.

    Requires ``feet_step_distance`` to be active in the same reward config.

    Args:
        env: The environment.
        command_name: Name of the command manager.
        sensor_cfg: Sensor configuration (unused, for API consistency).
        ang_vel_threshold: Angular velocity threshold (rad/s) above which asymmetry is allowed. Defaults to 0.2.
    """
    if not hasattr(env, "_last_step_dist"):
        return torch.zeros(env.num_envs, device=env.device)

    import torch

    cmd = env.command_manager.get_command(command_name)
    lin_vel_xy = torch.norm(cmd[:, :2], dim=1)
    ang_vel = cmd[:, 2]

    # Only penalize asymmetry when moving straight
    moving_straight = (lin_vel_xy > 0.1) & (torch.abs(ang_vel) < ang_vel_threshold)

    last_dist: torch.Tensor = getattr(env, "_last_step_dist")
    penalty = torch.abs(last_dist[:, 0] - last_dist[:, 1])
    penalty *= moving_straight.float()

    return penalty


def feet_max_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    speed_multiplier: float = 2.5,
) -> torch.Tensor:
    """Penalize feet moving faster than a command-adaptive threshold using squared L2 penalty.

    Discourages unnatural, high-speed foot motion relative to commanded locomotion speed.
    Useful for preventing the policy from achieving locomotion through fast, jittery motion.

    Args:
        env: The environment.
        command_name: Name of the command manager.
        asset_cfg: Configuration for the feet bodies.
        speed_multiplier: Multiplier on commanded linear velocity. Threshold = speed_multiplier * cmd_lin_vel.
                         Defaults to 2.5 (feet can move up to 2.5x commanded speed).

    Returns:
        Penalty tensor of shape (num_envs,). Zero if foot speed <= threshold, else (speed - threshold)^2.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    foot_lin_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)  # (num_envs, num_feet)

    # Get commanded linear velocity magnitude
    cmd = env.command_manager.get_command(command_name)
    cmd_lin_vel = torch.norm(cmd[:, :2], dim=1)  # (num_envs,)

    # Adaptive threshold: 2.5x the commanded speed
    threshold = speed_multiplier * cmd_lin_vel.unsqueeze(1)  # (num_envs, 1)

    # Penalty: (max(0, speed - threshold))^2
    excess_speed = torch.clamp(foot_lin_vel - threshold, min=0.0)
    penalty = torch.sum(excess_speed**2, dim=1)  # Sum across feet

    return penalty


def joint_direction_change_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_deadband: float = 0.2,
) -> torch.Tensor:
    """Penalize per-step joint velocity direction changes with a deadband.

    This term detects sign flips in joint velocity between two consecutive
    policy steps. A deadband avoids penalizing tiny sign changes caused by
    encoder noise around zero velocity.

    Args:
        env: The environment.
        asset_cfg: Joint subset to evaluate.
        vel_deadband: Minimum absolute velocity required (both previous and current)
            for a sign change to be considered real. Units: rad/s.

    Returns:
        Per-env penalty equal to the count of valid direction changes across joints.
    """
    asset = env.scene[asset_cfg.name]
    curr_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    buf_key = "_prev_joint_vel_for_direction_change"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, curr_vel.clone())
    prev_vel: torch.Tensor = getattr(env, buf_key)

    # Sign flip if product is negative; deadband filters noisy zero-crossings.
    flipped = (curr_vel * prev_vel) < 0.0
    above_deadband = (torch.abs(curr_vel) > vel_deadband) & (torch.abs(prev_vel) > vel_deadband)
    valid_flip = flipped & above_deadband

    # Cache for next step.
    prev_vel.copy_(curr_vel)

    return torch.sum(valid_flip.float(), dim=1)


def contact_gating_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    gait_freq: float = 1.5,
    stance_ratio: float = 0.6,
) -> torch.Tensor:
    """Reward for matching foot contact state to a gait phase clock.

    A phase clock at ``gait_freq`` Hz defines stance/swing windows per foot.
    Right foot uses the raw phase; left foot is offset by pi (alternating).
    The stance window occupies ``stance_ratio`` of each cycle.

    Returns +1 per foot for correct contact state, -1 for wrong.
    Total range [-2, +2]. Gated by velocity command (inactive when standing).
    """
    phase = 2.0 * torch.pi * gait_freq * env.episode_length_buf.float() * env.step_dt
    stance_end = stance_ratio * 2.0 * torch.pi

    right_phase = phase % (2.0 * torch.pi)
    left_phase = (phase + torch.pi) % (2.0 * torch.pi)

    right_should_contact = right_phase < stance_end
    left_should_contact = left_phase < stance_end

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    right_correct = (right_should_contact == in_contact[:, 0]).float() * 2.0 - 1.0
    left_correct = (left_should_contact == in_contact[:, 1]).float() * 2.0 - 1.0

    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=1) > 0.05).float()

    return (right_correct + left_correct) * moving


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical power (|velocity * torque|) across joints.

    Directly penalizes energy expenditure. Rapid shuffling wastes power
    through constant acceleration/deceleration, while smooth long strides
    have lower power due to constant-velocity swing phases.
    """
    asset = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(vel * torque), dim=1)


def step_frequency_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_freq: float = 3.0,
) -> torch.Tensor:
    """Quadratic penalty when step frequency deviates from a target.

    Uses cumulative step counting per episode. Resets automatically at
    the start of each episode via ``episode_length_buf``.

    Only active when the robot is commanded to move (||vel_xy|| > 0.05).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    if not hasattr(env, "_sfp_count"):
        env._sfp_count = torch.zeros(env.num_envs, device=env.device)
        env._sfp_elapsed = torch.zeros(env.num_envs, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._sfp_count[just_reset] = 0.0
    env._sfp_elapsed[just_reset] = 0.0

    env._sfp_count += first_contact.any(dim=1).float()
    env._sfp_elapsed += env.step_dt

    freq = env._sfp_count / env._sfp_elapsed.clamp(min=env.step_dt)

    cmd = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd[:, :2], dim=1) > 0.05

    penalty = (freq - target_freq) ** 2 * moving.float()
    return penalty
