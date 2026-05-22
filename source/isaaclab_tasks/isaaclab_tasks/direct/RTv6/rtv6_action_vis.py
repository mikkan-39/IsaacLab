# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Live GUI plots for RTv6 gait actions (ManagerLiveVisualizer-compatible adapter)."""

from __future__ import annotations

from collections.abc import Sequence

from .rtv6_action import RTv6SinusoidalGaitController


class RTv6GaitActionVisManager:
    """Adapter so :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` can plot RTv6 gait signals.

    Not a full :class:`~isaaclab.managers.ManagerBase`; only implements the surface the visualizer reads.
    """

    has_debug_vis_implementation = True

    def __init__(self, gait_ctrl: RTv6SinusoidalGaitController, policy_action: object) -> None:
        self._gait_ctrl = gait_ctrl
        self._policy_action = policy_action  # tensor view with shape (num_envs, action_dim)
        self.num_envs = gait_ctrl.num_envs

    @property
    def active_terms(self) -> list[str]:
        return [
            "policy_action",
            "clipped_amplitude",
            "clipped_phase",
            "clipped_offset",
            "wave_right",
            "wave_left_mirrored",
            "target_right",
            "target_left",
        ]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        gc = self._gait_ctrl
        return [
            ("policy_action", self._policy_action[env_idx].detach().cpu().tolist()),
            ("clipped_amplitude", gc.vis_amplitude[env_idx].detach().cpu().tolist()),
            ("clipped_phase", gc.vis_phase_offset[env_idx].detach().cpu().tolist()),
            ("clipped_offset", gc.vis_offset[env_idx].detach().cpu().tolist()),
            ("wave_right", gc.vis_wave_right[env_idx].detach().cpu().tolist()),
            ("wave_left_mirrored", gc.vis_wave_left[env_idx].detach().cpu().tolist()),
            ("target_right", gc.vis_target_right[env_idx].detach().cpu().tolist()),
            ("target_left", gc.vis_target_left[env_idx].detach().cpu().tolist()),
        ]
