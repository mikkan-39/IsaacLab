# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Sinusoidal joint position targets: policy sets right-leg wave parameters; left leg is mirrored."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation

from .rtv6_constants import (
    AMPLITUDE_LIMIT,
    AMPLITUDE_MINIMUMS,
    GAIT_ACTION_DIM,
    GAIT_FREQ,
    LEG_JOINT_PAIRS,
    NUM_RIGHT_LEG_JOINTS,
    OFFSET_LIMIT,
    PHASE_OFFSET_LIMIT,
    RIGHT_LEG_JOINT_NAMES,
    OFFSETS_BASELINE,
    PHASE_OFFSETS_BASELINE,
    START_TIME,
)


class RTv6SinusoidalGaitController:
    """Right-leg gait parameters → per-joint position targets for both legs."""

    action_dim = GAIT_ACTION_DIM

    def __init__(self, robot: Articulation, num_envs: int, device: torch.device) -> None:
        self._asset = robot
        self.num_envs = num_envs
        self.device = device

        right_ids, right_names = robot.find_joints(list(RIGHT_LEG_JOINT_NAMES), preserve_order=True)
        if len(right_names) != NUM_RIGHT_LEG_JOINTS:
            raise RuntimeError(
                f"Expected {NUM_RIGHT_LEG_JOINTS} right leg joints, matched {len(right_names)}: {right_names}"
            )

        left_ids: list[int] = []
        invert_left: list[bool] = []
        invert_left_amplitude: list[bool] = []
        for _r_name, l_name, invert_target, invert_amp in LEG_JOINT_PAIRS:
            lid, _ = robot.find_joints([l_name], preserve_order=True)
            if len(lid) != 1:
                raise RuntimeError(f"Left leg joint not found: {l_name}")
            left_ids.append(lid[0])
            invert_left.append(invert_target)
            invert_left_amplitude.append(invert_amp)

        self._right_joint_ids = list(right_ids)
        self._left_joint_ids = left_ids
        self._all_joint_ids = self._right_joint_ids + self._left_joint_ids
        self._invert_left = torch.tensor(invert_left, device=device, dtype=torch.bool)
        self._invert_left_amplitude = torch.tensor(invert_left_amplitude, device=device, dtype=torch.bool)

        self._default_right = robot.data.default_joint_pos[:, self._right_joint_ids].clone()
        self._default_left = robot.data.default_joint_pos[:, self._left_joint_ids].clone()

        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=device)
        self._amplitude = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)
        self._phase_offset = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)
        self._offset = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)

        self._two_pi_f = 2.0 * math.pi * GAIT_FREQ
        self._amplitude_minimums = torch.tensor(AMPLITUDE_MINIMUMS, device=device, dtype=torch.float32).view(1, -1)
        self._offsets_baseline = torch.tensor(OFFSETS_BASELINE, device=device, dtype=torch.float32).view(1, -1)
        self._phase_offsets_baseline = torch.tensor(PHASE_OFFSETS_BASELINE, device=device, dtype=torch.float32).view(1, -1)
        self._start_time = torch.tensor(START_TIME, device=device, dtype=torch.float32).view(1, -1)

        # Soft joint limits (same as DelayedBacklashJointPositionAction) — clamp targets before sim write.
        soft_lim = robot.data.soft_joint_pos_limits[0, self._all_joint_ids].clone()  # (12, 2)
        self._jp_min = soft_lim[:, 0].unsqueeze(0).expand(num_envs, -1).contiguous()
        self._jp_max = soft_lim[:, 1].unsqueeze(0).expand(num_envs, -1).contiguous()

        # Per-joint offset bounds so wave center (default + offset, mirrored on left) stays inside soft limits.
        soft_r = robot.data.soft_joint_pos_limits[:, self._right_joint_ids, :]
        soft_l = robot.data.soft_joint_pos_limits[:, self._left_joint_ids, :]
        off_min_r = soft_r[..., 0] - self._default_right
        off_max_r = soft_r[..., 1] - self._default_right
        off_min_l = soft_l[..., 0] - self._default_left
        off_max_l = soft_l[..., 1] - self._default_left
        off_min_l_inv = self._default_left - soft_l[..., 1]
        off_max_l_inv = self._default_left - soft_l[..., 0]
        off_min_l = torch.where(self._invert_left.unsqueeze(0), off_min_l_inv, off_min_l)
        off_max_l = torch.where(self._invert_left.unsqueeze(0), off_max_l_inv, off_max_l)
        policy_bound = torch.tensor([-OFFSET_LIMIT, OFFSET_LIMIT], device=device, dtype=torch.float32)
        off_min = torch.maximum(torch.maximum(off_min_r, off_min_l), policy_bound[0])
        off_max = torch.minimum(torch.minimum(off_max_r, off_max_l), policy_bound[1])
        invalid = off_min > off_max
        self._offset_min = torch.where(invalid, 0.0, off_min)
        self._offset_max = torch.where(invalid, 0.0, off_max)

        # Buffers for Isaac Lab live plots (updated in :meth:`apply_to_sim`).
        self.vis_amplitude = self._amplitude
        self.vis_phase_offset = self._phase_offset
        self.vis_offset = self._offset
        self.vis_wave_right = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)
        self.vis_wave_left = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)
        self.vis_target_right = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)
        self.vis_target_left = torch.zeros(num_envs, NUM_RIGHT_LEG_JOINTS, device=device)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Parse policy output once per control step. Order per joint: amp, phase, offset."""
        self._raw_actions[:] = actions
        params = actions.view(self.num_envs, NUM_RIGHT_LEG_JOINTS, 3)
        # raw_amp = params[..., 0]
        # raw_phase = params[..., 1]
        # raw_offset = params[..., 2]

        # TEMP: fixed from constants; policy channels ignored.
        raw_amp = self._amplitude_minimums.expand(self.num_envs, -1)
        raw_phase = self._phase_offsets_baseline.expand(self.num_envs, -1)
        raw_offset = self._offsets_baseline.expand(self.num_envs, -1)

        amp = torch.clamp(raw_amp, min=0.0, max=1.0) * AMPLITUDE_LIMIT
        self._amplitude[:] = torch.maximum(amp, self._amplitude_minimums)
        # TEMP baselines are radians; do not clamp to [-1, 1] and scale (that maps ±π/2 → ±π, collapsing phases).
        self._phase_offset[:] = raw_phase
        self._offset[:] = torch.clamp(raw_offset, self._offset_min, self._offset_max)

    def apply_to_sim(self, sim_time_s: torch.Tensor) -> None:
        """Recompute and write position targets for all leg joints (call every physics step).

        Args:
            sim_time_s: Per-env simulation time in seconds, shape ``(num_envs,)``.
        """
        clock = self._two_pi_f * sim_time_s.unsqueeze(-1)
        sin_r = torch.sin(clock + self._phase_offset)
        amp_active = self._amplitude * (sim_time_s.unsqueeze(-1) >= self._start_time).to(self._amplitude.dtype)
        target_right = self._default_right + self._offset + amp_active * sin_r

        sin_l = torch.sin(clock + self._phase_offset + math.pi)
        amp_l = torch.where(self._invert_left_amplitude.unsqueeze(0), -amp_active, amp_active)
        wave_left = self._offset + amp_l * sin_l
        # Optionally negate full wave (offset + swing); then add default_L.
        wave_left_mirrored = torch.where(self._invert_left.unsqueeze(0), -wave_left, wave_left)

        target_left = self._default_left + wave_left_mirrored

        self.vis_wave_right[:] = self._offset + amp_active * sin_r
        self.vis_wave_left[:] = wave_left_mirrored

        targets = torch.cat([target_right, target_left], dim=1)
        torch.clamp(targets, self._jp_min, self._jp_max, out=targets)
        self.vis_target_right[:] = targets[:, :NUM_RIGHT_LEG_JOINTS]
        self.vis_target_left[:] = targets[:, NUM_RIGHT_LEG_JOINTS:]
        self._asset.set_joint_position_target(targets, joint_ids=self._all_joint_ids)

    def reset(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            return
        if isinstance(env_ids, slice):
            self._raw_actions.zero_()
            self._amplitude.zero_()
            self._phase_offset.zero_()
            self._offset.zero_()
            return
        if isinstance(env_ids, torch.Tensor):
            eid = env_ids
        else:
            eid = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[eid] = 0.0
        self._amplitude[eid] = 0.0
        self._phase_offset[eid] = 0.0
        self._offset[eid] = 0.0
