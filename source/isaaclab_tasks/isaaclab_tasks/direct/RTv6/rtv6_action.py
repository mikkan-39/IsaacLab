# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Direct RL joint action: LPF → affine (scale/offset) → backlash → noise → FIFO delay → sim targets.

Behavior matches the former ``DelayedBacklashJointPositionAction`` + ``JointPositionAction`` stack without
:class:`ActionManager` or RTv5 config imports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.utils import configclass


@configclass
class RTv6JointActionCfg:
    """Defaults aligned with RTv5 ``DelayedBacklashJointPositionActionCfg`` / ``ActionsCfg``."""

    joint_names: list[str] | None = None
    """Regex list passed to :meth:`Articulation.find_joints`. ``None`` uses :data:`CONTROLLABLE_JOINTS_REGEX`."""

    scale: float = 1.0
    use_default_offset: bool = True
    preserve_order: bool = True

    min_delay_steps: int = 2
    max_delay_steps: int = 4
    backlash_deg: float = 1.0
    action_noise_std: float = 0.01
    action_lpf_alpha: float = 0.4


class RTv6DelayedJointPositionController:
    """Tensor-only controller; exposes ``raw_actions`` / ``processed_actions`` like an action term."""

    def __init__(
        self,
        cfg: RTv6JointActionCfg,
        robot: Articulation,
        joint_name_patterns: list[str],
        num_envs: int,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self._asset = robot
        self.num_envs = num_envs
        self.device = device

        self._joint_ids, self._joint_names = robot.find_joints(joint_name_patterns, preserve_order=cfg.preserve_order)
        self.action_dim = len(self._joint_ids)
        if self.action_dim == 0:
            raise RuntimeError("RTv6DelayedJointPositionController: no joints matched the given patterns.")

        self._scale = float(cfg.scale)
        if cfg.use_default_offset:
            self._offset = robot.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros(num_envs, self.action_dim, device=device)

        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        buf_len = cfg.max_delay_steps + 1
        self._buf = torch.zeros(buf_len, num_envs, self.action_dim, device=device)
        default = self._offset.clone()
        self._buf[:] = default.unsqueeze(0)
        self._buf_head = 0

        self._delay = torch.randint(cfg.min_delay_steps, cfg.max_delay_steps + 1, (num_envs,), device=device)

        self._backlash_rad = cfg.backlash_deg * (math.pi / 180.0)
        self._gear_pos = default.clone()
        self._last_dir = torch.zeros(num_envs, self.action_dim, device=device)

        self._noise_std = cfg.action_noise_std
        self._env_arange = torch.arange(num_envs, device=device)

        self._lpf_alpha = float(cfg.action_lpf_alpha)
        self._lpf_state = torch.zeros(num_envs, self.action_dim, device=device)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Once per env step, after optional env-level action noise."""
        self._raw_actions[:] = actions

        if self._lpf_alpha >= 1.0 - 1e-9:
            filtered = actions
        else:
            one_m = 1.0 - self._lpf_alpha
            self._lpf_state.mul_(one_m).add_(actions, alpha=self._lpf_alpha)
            filtered = self._lpf_state

        self._processed_actions = filtered * self._scale + self._offset

        target = self._processed_actions

        if self._backlash_rad > 0:
            delta = target - self._gear_pos
            direction = torch.sign(delta)
            dir_changed = (direction != self._last_dir) & (self._last_dir != 0)
            movement = torch.where(
                dir_changed,
                torch.clamp(torch.abs(delta) - self._backlash_rad, min=0) * direction,
                delta,
            )
            self._gear_pos = self._gear_pos + movement
            self._last_dir = torch.where(delta != 0, direction, self._last_dir)
            output = self._gear_pos.clone()
        else:
            output = target.clone()

        if self._noise_std > 0:
            output = output + torch.randn_like(output) * self._noise_std

        self._buf[self._buf_head] = output
        self._buf_head = (self._buf_head + 1) % self._buf.shape[0]

    def apply_to_sim(self) -> None:
        """Call once per physics substep."""
        read_idx = (self._buf_head - 1 - self._delay) % self._buf.shape[0]
        delayed = self._buf[read_idx, self._env_arange]
        self._asset.set_joint_position_target(delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            return
        if isinstance(env_ids, torch.Tensor):
            eid = env_ids
        else:
            eid = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._raw_actions[eid] = 0.0

        default_vals = self._offset[eid]
        self._gear_pos[eid] = default_vals
        self._last_dir[eid] = 0.0
        self._buf[:, eid] = default_vals.unsqueeze(0)

        n = eid.shape[0]
        self._delay[eid] = torch.randint(
            self.cfg.min_delay_steps,
            self.cfg.max_delay_steps + 1,
            (n,),
            device=self.device,
        )
        self._lpf_state[eid] = 0.0
