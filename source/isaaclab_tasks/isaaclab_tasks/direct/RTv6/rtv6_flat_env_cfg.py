# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rtv6_rough_env_cfg import RTv6RoughEnvCfg


@configclass
class RTv6FlatEnvCfg(RTv6RoughEnvCfg):
    """Flat ground; no terrain curriculum (matches RTv5 flat)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.enable_terrain_curriculum = False


@configclass
class RTv6FlatEnvCfg_PLAY(RTv6FlatEnvCfg):
    """Smaller flat scene for play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
