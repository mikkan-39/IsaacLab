# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RTv6 RSL-RL runner hook for custom policy initialization."""

from __future__ import annotations

import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner


def zero_init_actor_mean(policy: nn.Module) -> None:
    """Zero the last :class:`nn.Linear` in the actor MLP (Gaussian mean head in RSL-RL 3.x)."""
    last_linear: nn.Linear | None = None
    for module in policy.actor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise RuntimeError("Could not find a Linear layer in policy.actor.")
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)


class RTv6OnPolicyRunner(OnPolicyRunner):
    """On-policy runner that zero-inits the actor mean head when ``policy.zero_actor_mean`` is set."""

    def _construct_algorithm(self, obs: TensorDict):
        policy_cfg = self.cfg.get("policy", {})
        zero_mean = policy_cfg.pop("zero_actor_mean", False)
        alg = super()._construct_algorithm(obs)
        if zero_mean:
            zero_init_actor_mean(alg.policy)
        return alg
