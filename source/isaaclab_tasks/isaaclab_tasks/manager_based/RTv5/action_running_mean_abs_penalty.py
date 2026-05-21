# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Penalty on per-dimension |episode-mean(action)| for delta / speed-style control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gather_action_vector(env: ManagerBasedRLEnv, use_raw: bool) -> torch.Tensor:
    """Return (num_envs, action_dim), preferring concatenated raw_actions when available."""
    if use_raw:
        chunks: list[torch.Tensor] | None = []
        for term in env.action_manager._terms.values():
            if not hasattr(term, "raw_actions"):
                chunks = None
                break
            chunks.append(term.raw_actions)
        if chunks is not None and len(chunks) == len(env.action_manager._terms) and len(chunks) > 0:
            return torch.cat(chunks, dim=1)
    return env.action_manager.action


class ActionRunningMeanAbsPenalty(ManagerTermBase):
    """Penalize sum of absolute per-action dimension episode means within the current episode.

    Accumulates ``sum(a)`` and a step count each time the gate is active; the reward value is
    ``sum_i |mean_i|`` with ``mean = sum/count``. Defaults to raw policy outputs (falls back to the
    applied action vector when a term exposes no ``raw_actions``).

    With ``command_name`` set and ``min_linear_cmd``, sums only steps where planar command magnitude
    is at least that value (walking), so standing still does not pull means toward zero.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._use_raw: bool = bool(p.get("use_raw_actions", True))
        self._command_name: str | None = p.get("command_name", None)
        self._min_linear_cmd: float = float(p.get("min_linear_cmd", 0.06))

        adim = env.action_manager.total_action_dim
        self._sum = torch.zeros(env.num_envs, adim, device=env.device)
        self._count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._reset_buffers(env_ids)

    def _reset_buffers(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            self._sum.zero_()
            self._count.zero_()
            return
        if isinstance(env_ids, slice):
            self._sum[env_ids] = 0.0
            self._count[env_ids] = 0
            return
        eid = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        if eid.numel() == 0:
            return
        self._sum[eid] = 0.0
        self._count[eid] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        use_raw_actions: bool = True,
        command_name: str | None = None,
        min_linear_cmd: float = 0.06,
    ) -> torch.Tensor:
        del use_raw_actions, command_name, min_linear_cmd  # from cfg.params; use self._*

        act = _gather_action_vector(env, self._use_raw)

        if self._command_name is not None:
            cmd = env.command_manager.get_command(self._command_name)
            active = (torch.norm(cmd[:, :2], dim=1) >= self._min_linear_cmd).unsqueeze(1).to(act.dtype)
        else:
            active = torch.ones(env.num_envs, 1, device=env.device, dtype=act.dtype)

        self._sum += act * active
        self._count += active.squeeze(1).long()

        mean = self._sum / self._count.unsqueeze(1).clamp(min=1)
        penalty = torch.sum(torch.abs(mean), dim=-1)
        penalty = torch.where(self._count > 0, penalty, torch.zeros_like(penalty))
        return penalty
