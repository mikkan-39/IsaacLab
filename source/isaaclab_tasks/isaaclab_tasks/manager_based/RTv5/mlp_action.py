from __future__ import annotations

"""Joint position action whose target is shaped by a learned servo MLP.

The policy commands an **absolute** joint position target (standard ``JointPositionAction``
processing: ``target = default + scale * action``, clipped). That target is fed to a trained servo
MLP that predicts the **delay-free** position the real servo would reach, and that prediction is used
as the position setpoint for an :class:`~isaaclab.actuators.IdealPDActuator` (configured on the
robot). The PD's near-constant lag re-introduces the servo's transport delay, and the MLP supplies
the frequency-dependent component an analytical PD/DCMotor cannot.

Because the MLP captures the servo's lag (and the data it learned from already contains the servo's
backlash), this term deliberately does **not** add a transport-delay buffer or a backlash dead-zone
(unlike :class:`DelayedBacklashJointPositionAction`).

The MLP advances once per control step (``process_actions``, 50 Hz); ``apply_actions`` writes the
held setpoint each physics substep.
"""

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.RTv5.servo_mlp_reference import ServoMlpReference

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MlpJointPositionAction(JointPositionAction):
    """Absolute joint position action with a learned-MLP servo reference (see module docstring)."""

    cfg: MlpJointPositionActionCfg

    def __init__(self, cfg: MlpJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        default = self._offset if isinstance(self._offset, torch.Tensor) else torch.full(
            (self.num_envs, self.action_dim), float(self._offset), device=self.device
        )
        self._default_pos = default.clone()

        # learned servo reference; its prediction is the setpoint applied each physics substep
        self._ref = ServoMlpReference(
            cfg.mlp_checkpoint, self.num_envs, self.action_dim, self.device, control_dt=env.step_dt
        )
        self._ref.reset(slice(None), default)
        self._setpoint = default.clone()

    def process_actions(self, actions: torch.Tensor):
        # standard absolute position target (target = default + scale * action, then clipped)
        super().process_actions(actions)
        # learned servo reference -> position setpoint for the IdealPD actuator
        self._setpoint = self._ref.step(self._processed_actions)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._setpoint, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            return
        default_vals = self._default_pos[env_ids]
        self._setpoint[env_ids] = default_vals
        self._ref.reset(env_ids, default_vals)


@configclass
class MlpJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`MlpJointPositionAction`.

    Standard ``JointPositionActionCfg`` semantics: the policy commands an absolute target via
    ``scale``/``offset``. ``use_default_offset=True`` makes the action a position offset from the
    default pose and seeds the MLP reference at that pose.
    """

    class_type: type[ActionTerm] = MlpJointPositionAction

    mlp_checkpoint: str = ""
    """Path to a trained servo MLP checkpoint (``servo_mlp.pt`` from ``train_servo_mlp.py``)."""
