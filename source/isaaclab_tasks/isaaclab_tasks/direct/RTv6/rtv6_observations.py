# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Direct observation tensors for RTv6."""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .rtv6_constants import GAIT_FREQ, PROJECTED_GRAVITY_OBS_NOISE_STD


def gait_phase_sincos(env) -> torch.Tensor:
    """sin/cos of the global gait clock, shape ``(num_envs, 2)``."""
    phase = 2.0 * math.pi * GAIT_FREQ * env.episode_length_buf.float() * env.step_dt
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def projected_gravity_b(robot: Articulation) -> torch.Tensor:
    """Gravity direction in the root/body frame."""
    return robot.data.projected_gravity_b


def projected_gravity_obs(robot: Articulation) -> torch.Tensor:
    """Noisy projected gravity (std=0.025)."""
    g = projected_gravity_b(robot)
    if PROJECTED_GRAVITY_OBS_NOISE_STD > 0.0:
        g = g + torch.randn_like(g) * PROJECTED_GRAVITY_OBS_NOISE_STD
    return g


def gait_metrics(
    env,
    env_ids,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
    min_air_time: float = 0.04,
) -> dict[str, float]:
    """Passive gait metrics for curriculum logging."""
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

    return {
        "step_freq_hz": freq.mean().item(),
        "symmetry": symmetry.mean().item(),
        "mean_swing_s": mean_swing.mean().item(),
        "steps_right": env._gait_steps_right.mean().item(),
        "steps_left": env._gait_steps_left.mean().item(),
        "gait_clock_hz": GAIT_FREQ,
    }
