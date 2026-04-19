from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedBacklashJointPositionAction(JointPositionAction):
    """Joint position action with transport delay, backlash dead-zone, and action noise.

    Models three real-world servo characteristics that are absent from the default
    ``JointPositionAction``:

    **Transport delay** – A per-environment FIFO ring buffer delays the commanded
    position by a random number of RL steps (resampled on episode reset).  At 50 Hz
    control this lets you model 40-80 ms of total pipeline latency (bus scheduling +
    servo internal response dead-time).

    **Backlash** – On direction reversal the first ``backlash_deg`` degrees of
    commanded movement are absorbed by the gear dead-zone and never reach the
    simulated joint.  Continuing in the same direction applies full movement.
    Matches the model used by the Bimo project for ST3215 serial bus servos.

    **Action noise** – Gaussian noise (in radians) added *after* backlash filtering
    and *before* entering the delay buffer, modelling servo positioning jitter.

    The ``processed_actions`` property still returns the un-delayed, un-backlashed
    value (what the policy intended) so that observation terms like ``last_action``
    and penalties like ``action_rate_l2`` remain correct.
    """

    cfg: DelayedBacklashJointPositionActionCfg

    def __init__(self, cfg: DelayedBacklashJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        buf_len = cfg.max_delay_steps + 1
        self._buf = torch.zeros(buf_len, self.num_envs, self.action_dim, device=self.device)

        default = self._offset if isinstance(self._offset, torch.Tensor) else torch.full_like(self._buf[0], self._offset)
        self._buf[:] = default.unsqueeze(0)
        self._buf_head = 0

        self._delay = torch.randint(
            cfg.min_delay_steps, cfg.max_delay_steps + 1,
            (self.num_envs,), device=self.device,
        )

        self._backlash_rad = cfg.backlash_deg * (math.pi / 180.0)
        self._gear_pos = default.clone()
        self._last_dir = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        self._noise_std = cfg.action_noise_std
        self._env_arange = torch.arange(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

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

    def apply_actions(self):
        read_idx = (self._buf_head - 1 - self._delay) % self._buf.shape[0]
        delayed = self._buf[read_idx, self._env_arange]
        self._asset.set_joint_position_target(delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            return

        if isinstance(self._offset, torch.Tensor):
            default_vals = self._offset[env_ids]
        else:
            default_vals = self._offset

        self._gear_pos[env_ids] = default_vals
        self._last_dir[env_ids] = 0.0

        if isinstance(default_vals, torch.Tensor):
            self._buf[:, env_ids] = default_vals.unsqueeze(0)
        else:
            self._buf[:, env_ids] = default_vals

        n = len(env_ids) if hasattr(env_ids, "__len__") else self.num_envs
        self._delay[env_ids] = torch.randint(
            self.cfg.min_delay_steps, self.cfg.max_delay_steps + 1,
            (n,), device=self.device,
        )


@configclass
class DelayedBacklashJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`DelayedBacklashJointPositionAction`."""

    class_type: type[ActionTerm] = DelayedBacklashJointPositionAction

    min_delay_steps: int = 2
    """Minimum action delay in RL steps (inclusive). At 50 Hz, 1 step = 20 ms."""

    max_delay_steps: int = 4
    """Maximum action delay in RL steps (inclusive). At 50 Hz, 4 steps = 80 ms."""

    backlash_deg: float = 1.0
    """Backlash dead-zone in degrees. On direction reversal this much movement is
    absorbed before the joint target starts tracking again."""

    action_noise_std: float = 0.0
    """Gaussian noise std (radians) added to the commanded joint position.
    Set to ~0.01 for ~0.6 deg positioning jitter typical of bus servos."""
