from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators.actuator_pd import DCMotor
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .backlash_dc_motor_cfg import BacklashDCMotorCfg


class BacklashDCMotor(DCMotor):
    """DC motor actuator with command delay, gear backlash, and position noise.

    Combines three real-servo behaviors into a single actuator model:

    1. **Command delay** (configurable physics steps, randomized per-env on reset)
    2. **Gear backlash** dead-band on position targets
    3. **Gaussian noise** on the effective target position

    Pipeline::

        DesiredPos → DelayBuffer → BacklashDeadBand → GaussianNoise → DCMotor PD → TorqueSpeedClip → Effort

    The backlash is modelled as a symmetric dead-band of width ``backlash_rad``.
    When the commanded direction reverses, the output shaft stays stationary until
    the input has traversed the full backlash width, then follows with a constant
    offset of ``backlash_rad / 2``.
    """

    cfg: BacklashDCMotorCfg

    def __init__(self, cfg: BacklashDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # -- delay buffers (same approach as DelayedPDActuator)
        self._positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # -- backlash state
        self._gear_position = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self._half_backlash = cfg.backlash_rad / 2.0
        # -- convenience
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # randomise per-env delay
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self._positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self._velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self._efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        self._positions_delay_buffer.reset(env_ids)
        self._velocities_delay_buffer.reset(env_ids)
        self._efforts_delay_buffer.reset(env_ids)
        # reset backlash state
        self._gear_position[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # 1) apply command delay
        control_action.joint_positions = self._positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self._velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self._efforts_delay_buffer.compute(control_action.joint_efforts)

        # 2) apply backlash dead-band to position targets
        if control_action.joint_positions is not None:
            target = control_action.joint_positions
            upper = self._gear_position + self._half_backlash
            lower = self._gear_position - self._half_backlash
            self._gear_position = torch.where(
                target > upper,
                target - self._half_backlash,
                torch.where(target < lower, target + self._half_backlash, self._gear_position),
            )
            # 3) additive Gaussian noise (does not mutate _gear_position)
            effective_target = self._gear_position
            if self.cfg.noise_std > 0.0:
                effective_target = effective_target + self.cfg.noise_std * torch.randn_like(effective_target)
            control_action.joint_positions = effective_target

        # 4) DCMotor: PD torque + torque-speed saturation clip
        return super().compute(control_action, joint_pos, joint_vel)
