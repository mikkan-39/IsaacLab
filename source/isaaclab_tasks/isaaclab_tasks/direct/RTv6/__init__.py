# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL (RTv6) — mirrors manager-based RTv5 rough/flat MDP."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Rough-RTv6-v0",
    entry_point=f"{__name__}.rtv6_env:RTv6RoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rtv6_rough_env_cfg:RTv6RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RTv6RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-RTv6-Play-v0",
    entry_point=f"{__name__}.rtv6_env:RTv6RoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rtv6_rough_env_cfg:RTv6RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RTv6RoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-RTv6-v0",
    entry_point=f"{__name__}.rtv6_env:RTv6RoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rtv6_flat_env_cfg:RTv6FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RTv6FlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-RTv6-Play-v0",
    entry_point=f"{__name__}.rtv6_env:RTv6RoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rtv6_flat_env_cfg:RTv6FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RTv6FlatPPORunnerCfg",
    },
)
