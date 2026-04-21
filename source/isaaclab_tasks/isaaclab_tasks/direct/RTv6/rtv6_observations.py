# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Direct observation tensors for RTv6 (no ObservationManager / RTv5 MDP obs helpers)."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .rtv6_constants import GAIT_FREQ_RANGE


def get_gait_freq(env) -> torch.Tensor:
    """Per-env gait clock frequency (Hz); lazy-initialized like RTv5 ``_get_gait_freq``."""
    if not hasattr(env, "_gait_freq"):
        env._gait_freq = torch.empty(env.num_envs, device=env.device).uniform_(*GAIT_FREQ_RANGE)
    return env._gait_freq


def resample_gait_freq(env, env_ids) -> None:
    """Resample gait clock frequency for ``env_ids`` (used on episode reset)."""
    freq = get_gait_freq(env)
    n = len(env_ids) if not isinstance(env_ids, slice) else env.num_envs
    freq[env_ids] = torch.empty(n, device=env.device).uniform_(*GAIT_FREQ_RANGE)


def gait_phase_sincos(env) -> torch.Tensor:
    """sin/cos gait phase, shape ``(num_envs, 2)``."""
    freq = get_gait_freq(env)
    phase = 2.0 * torch.pi * freq * env.episode_length_buf.float() * env.step_dt
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def projected_gravity_b(robot: Articulation) -> torch.Tensor:
    """Gravity direction in the root/body frame (same as ``mdp.projected_gravity`` for the robot)."""
    return robot.data.projected_gravity_b


def joint_pos_rel_subset(robot: Articulation, joint_ids: list[int] | torch.Tensor | slice) -> torch.Tensor:
    """Joint positions relative to defaults for the selected DOFs."""
    return robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]


def gait_metrics(
    env,
    env_ids,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
    min_air_time: float = 0.04,
) -> dict[str, float]:
    """Passive gait metrics (same behavior as RTv5 ``gait_metrics``); RTv6-owned, no RTv5 imports."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    if not hasattr(env, "_gait_steps_right"):
        env._gait_steps_right = torch.zeros(env.num_envs, device=env.device)
        env._gait_steps_left = torch.zeros(env.num_envs, device=env.device)
        env._gait_elapsed = torch.zeros(env.num_envs, device=env.device)
        env._gait_prev_air_time = torch.zeros(env.num_envs, 2, device=env.device)
        env._gait_swing_sum = torch.zeros(env.num_envs, device=env.device)
        env._gait_swing_count = torch.zeros(env.num_envs, device=env.device)

    real_step_right = first_contact[:, 0] & (env._gait_prev_air_time[:, 0] > min_air_time)
    real_step_left = first_contact[:, 1] & (env._gait_prev_air_time[:, 1] > min_air_time)

    env._gait_steps_right += real_step_right.float()
    env._gait_steps_left += real_step_left.float()
    env._gait_elapsed += env.step_dt

    elapsed = env._gait_elapsed.clamp(min=0.1)
    total_steps = env._gait_steps_right + env._gait_steps_left
    freq = total_steps / elapsed

    max_steps = torch.max(env._gait_steps_right, env._gait_steps_left).clamp(min=1.0)
    min_steps = torch.min(env._gait_steps_right, env._gait_steps_left)
    symmetry = min_steps / max_steps

    for foot_idx, real_step in enumerate([real_step_right, real_step_left]):
        env._gait_swing_sum += torch.where(
            real_step, env._gait_prev_air_time[:, foot_idx], torch.zeros_like(env._gait_swing_sum)
        )
        env._gait_swing_count += real_step.float()

    mean_swing = env._gait_swing_sum / env._gait_swing_count.clamp(min=1.0)

    env._gait_prev_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids].clone()

    env._gait_steps_right[env_ids] = 0.0
    env._gait_steps_left[env_ids] = 0.0
    env._gait_elapsed[env_ids] = 0.0
    env._gait_swing_sum[env_ids] = 0.0
    env._gait_swing_count[env_ids] = 0.0
    resample_gait_freq(env, env_ids)

    gait_freq = get_gait_freq(env)

    return {
        "step_freq_hz": freq.mean().item(),
        "symmetry": symmetry.mean().item(),
        "mean_swing_s": mean_swing.mean().item(),
        "steps_right": env._gait_steps_right.mean().item(),
        "steps_left": env._gait_steps_left.mean().item(),
        "gait_clock_hz": gait_freq.mean().item(),
    }
