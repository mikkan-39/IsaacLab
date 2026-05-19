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
    """Delta-integrated joint position action with transport delay, backlash, and noise.

    Pipeline per step:

    1. **Raw action LPF** – EMA on the raw policy output (``a_t = (1-α)a_{t-1} + α a_raw``)
       attenuates jerk in the policy's signal *before* anything downstream sees it.
    2. **Delta integration** – The (filtered) raw action is treated as a *delta* in
       joint-target space: ``cmd_t = clamp(cmd_{t-1} + delta_scale * a_filt, soft_lim)``.
       This bounds the per-step slew rate of the commanded position and prevents the
       policy from issuing target jumps no real servo could follow.
       Solves Tier-2 item #4 (absolute-target vs delta) from the sim2real analysis:
       slew-rate-bounded targets are dramatically more robust to observation noise
       than absolute targets and match the Bimo control scheme.
    3. **Backlash dead-zone** – On direction reversal the first ``backlash_deg``
       degrees of commanded movement are absorbed by the gear lash and never reach
       the simulated joint. Continuing in the same direction applies full movement.
    4. **Position jitter** – Optional Gaussian noise (radians) added after backlash
       and before the delay buffer, modelling ST3215-class servo step quantization
       and positioning jitter. Solves Tier-2 item #5 (action noise disabled).
    5. **Transport delay** – Per-env FIFO ring buffer delays the final command by
       a random number of RL steps (resampled on reset).

    ``processed_actions`` returns the *un-delayed* integrated target (the policy's
    intended joint target) so observation terms like ``last_action`` and penalties
    like ``action_rate_l2`` operate on the same quantity the policy can reason about.
    """

    cfg: DelayedBacklashJointPositionActionCfg

    def __init__(self, cfg: DelayedBacklashJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Default joint positions per env (shape: num_envs x action_dim) used as the
        # starting integrated target and as the seed for every history buffer below.
        default = self._offset if isinstance(self._offset, torch.Tensor) else torch.full(
            (self.num_envs, self.action_dim), float(self._offset), device=self.device
        )

        # Delay ring buffer holds the *post-backlash, post-noise* joint targets that
        # will eventually be applied. Length = max_delay_steps + 1 (oldest slot + max
        # lookback). Seeded to default so the very first apply() reads a sane value.
        buf_len = cfg.max_delay_steps + 1
        self._buf = torch.zeros(buf_len, self.num_envs, self.action_dim, device=self.device)
        self._buf[:] = default.unsqueeze(0)
        self._buf_head = 0

        # Per-env transport delay sampled at reset. Modeled in RL steps; at 50 Hz
        # control 1 step = 20 ms, so [2, 4] = 40-80 ms pipeline latency.
        self._delay = torch.randint(
            cfg.min_delay_steps, cfg.max_delay_steps + 1,
            (self.num_envs,), device=self.device,
        )

        self._backlash_rad = cfg.backlash_deg * (math.pi / 180.0)
        self._gear_pos = default.clone()
        self._last_dir = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        self._noise_std = cfg.action_noise_std
        self._env_arange = torch.arange(self.num_envs, device=self.device)

        self._lpf_alpha = float(cfg.action_lpf_alpha)
        self._lpf_state = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # Integrated commanded joint target. Persists across env steps. Always
        # clamped into the soft joint position limits so the policy cannot drift
        # the target arbitrarily far from physically reachable values.
        self._integrated_target = default.clone()
        self._delta_scale = float(cfg.delta_scale)

        # Soft joint limits used to clamp the integrated target. Shape (num_joints, 2).
        # We materialize per-env copies so reset() can selectively re-seed sub-ranges.
        soft_lim = self._asset.data.soft_joint_pos_limits[0, self._joint_ids].clone()  # (J, 2)
        self._jp_min = soft_lim[:, 0].unsqueeze(0).expand(self.num_envs, -1).contiguous()
        self._jp_max = soft_lim[:, 1].unsqueeze(0).expand(self.num_envs, -1).contiguous()

        # Cache default joint pos for reset() (handles both scalar and tensor _offset).
        self._default_pos = default.clone()

        # Reset-time history randomization magnitude (Tier-3 item #11):
        # On a fraction of resets, seed the buffers with a small random offset from
        # default rather than exact default. Trains the policy to tolerate startup
        # states where the action history is not what it expects (e.g. handed
        # control from a stand-up controller on the real robot).
        self._history_jitter_std = float(cfg.reset_history_jitter_std)
        self._history_jitter_prob = float(cfg.reset_history_jitter_prob)

    def process_actions(self, actions: torch.Tensor):
        # Hard-clip raw policy output to per-joint bounds from cfg.clip. This
        # bounds the per-step delta authority and removes the need for an
        # action_clip_violation reward (the policy can no longer exceed the
        # bounds, so penalizing violation is meaningless). Done BEFORE the LPF
        # and integration so downstream stages always see in-range actions.
        if self.cfg.clip is not None:
            actions = torch.clamp(actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
            
        # LPF on raw policy output. Filters jerk in the *delta* signal before it
        # gets integrated, which keeps the slew rate well-behaved even if the
        # policy network produces noisy outputs.
        if self._lpf_alpha >= 1.0 - 1e-9:
            filtered = actions
        else:
            self._lpf_state.mul_(1.0 - self._lpf_alpha).add_(actions, alpha=self._lpf_alpha)
            filtered = self._lpf_state

        # Persist raw action for observation/reward terms (last_action, action_rate_l2).
        self._raw_actions[:] = actions

        # Delta integration with soft-limit clamping.
        self._integrated_target.add_(filtered * self._delta_scale)
        torch.clamp(self._integrated_target, self._jp_min, self._jp_max, out=self._integrated_target)
        self._processed_actions[:] = self._integrated_target

        # Backlash dead-zone applied to the integrated target -> gear position.
        if self._backlash_rad > 0:
            delta = self._integrated_target - self._gear_pos
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
            output = self._integrated_target.clone()

        # Position jitter (radians). ~0.005-0.01 rad matches ST3215-class servo
        # quantization (~0.087° per count) and physical positioning noise.
        if self._noise_std > 0:
            output = output + torch.randn_like(output) * self._noise_std

        # Push final value into delay buffer.
        self._buf[self._buf_head] = output
        self._buf_head = (self._buf_head + 1) % self._buf.shape[0]

    def apply_actions(self):
        # Read each env's delayed slot. _delay is per-env so we gather per row.
        read_idx = (self._buf_head - 1 - self._delay) % self._buf.shape[0]
        delayed = self._buf[read_idx, self._env_arange]
        self._asset.set_joint_position_target(delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Parent zeros _raw_actions for the reset envs.
        super().reset(env_ids)
        if env_ids is None:
            return

        default_vals = self._default_pos[env_ids]

        # Restart the integrator at the default pose, otherwise the policy would
        # inherit the previous episode's terminal target on reset.
        self._integrated_target[env_ids] = default_vals
        self._gear_pos[env_ids] = default_vals
        self._last_dir[env_ids] = 0.0
        self._lpf_state[env_ids] = 0.0

        # Seed the delay ring buffer. By default this is exact-default values.
        # With probability `reset_history_jitter_prob`, perturb the buffer with
        # Gaussian noise of std `reset_history_jitter_std` (radians). This is
        # Bimo's "raw degrees in history at reset" trick adapted to our space:
        # it teaches the policy to recover from arbitrary startup states where
        # the action history is not exactly the default pose. Tier-3 item #11.
        n = len(env_ids) if hasattr(env_ids, "__len__") else self.num_envs
        if self._history_jitter_std > 0.0 and self._history_jitter_prob > 0.0:
            jitter_mask = (torch.rand(n, device=self.device) < self._history_jitter_prob).float().unsqueeze(-1)
            jitter = torch.randn(n, self.action_dim, device=self.device) * self._history_jitter_std
            seed_vals = default_vals + jitter * jitter_mask
            self._buf[:, env_ids] = seed_vals.unsqueeze(0)
        else:
            self._buf[:, env_ids] = default_vals.unsqueeze(0)

        # Per-env transport delay resampled each reset.
        self._delay[env_ids] = torch.randint(
            self.cfg.min_delay_steps, self.cfg.max_delay_steps + 1,
            (n,), device=self.device,
        )


@configclass
class DelayedBacklashJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`DelayedBacklashJointPositionAction`.

    Note: ``scale`` from the parent is unused in delta mode; the per-step delta
    magnitude is controlled by ``delta_scale``. ``use_default_offset=True`` is
    recommended so the integrator starts at the default pose.
    """

    class_type: type[ActionTerm] = DelayedBacklashJointPositionAction

    delta_scale: float = 0.05
    """Per-step delta magnitude in radians for a unit raw action.

    A raw action of 1.0 moves the integrated target by ``delta_scale`` rad per
    control step. At 50 Hz, ``delta_scale=0.05`` caps slew rate at ~143 deg/s
    when the policy issues unit actions, which is a sane upper bound for
    ST3215-class servos under load. Lower for slower/more cautious gaits.
    """

    min_delay_steps: int = 2
    """Minimum transport delay in RL steps (inclusive). 50 Hz: 1 step = 20 ms."""

    max_delay_steps: int = 4
    """Maximum transport delay in RL steps (inclusive). 50 Hz: 4 steps = 80 ms."""

    backlash_deg: float = 1.0
    """Backlash dead-zone in degrees. On direction reversal this much movement is
    absorbed before the joint target starts tracking again."""

    action_noise_std: float = 0.0
    """Gaussian noise std (radians) added to the commanded joint position after
    backlash and before the delay buffer. ~0.005-0.01 is typical for bus servos."""

    action_lpf_alpha: float = 1.0
    """EMA weight on *new* raw actions: ``a_t = (1-α)*a_{t-1} + α*a_raw``.
    ``1.0`` = no filtering (default). ``0.2`` matches ``a_t = 0.8*a_{t-1} + 0.2*a_raw``."""

    reset_history_jitter_std: float = 0.0
    """Std (radians) of Gaussian perturbation applied to the delay buffer at reset
    with probability ``reset_history_jitter_prob``. Models hand-off from a
    non-policy controller (e.g. stand-up routine) where action history is not
    exactly the policy's expected default."""

    reset_history_jitter_prob: float = 0.0
    """Probability that a given reset perturbs the delay buffer (see above)."""
