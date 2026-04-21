# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.RTv5.velocity_env_cfg import EventCfg, MySceneCfg

from .rtv6_action import RTv6JointActionCfg


@configclass
class RTv6CommandsCfg:
    """Velocity command spec (mirrors RTv5 ``CommandsCfg.base_velocity``; no RTv5 import)."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.35), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class RTv6RoughEnvCfg(DirectRLEnvCfg):
    """Direct RL analogue of :class:`isaaclab_tasks.manager_based.RTv5.rough_env_cfg.RTv5RoughEnvCfg`."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2)

    events: EventCfg = EventCfg()

    commands: RTv6CommandsCfg = RTv6CommandsCfg()
    joint_action: RTv6JointActionCfg = RTv6JointActionCfg()

    decimation: int = 4
    episode_length_s: float = 15.0

    observation_space: int | dict = 1
    action_space: int | dict = 1

    # Set False on flat terrain (no generator curriculum).
    enable_terrain_curriculum: bool = True
    enable_gait_curriculum: bool = True

    def __post_init__(self) -> None:
        from isaaclab_assets import RT_CFG

        # Mirror LocomotionVelocityRoughEnvCfg + RTv5RoughEnvCfg.
        self.scene.robot = RT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = "cuda:0"
        self.sim.enable_scene_query_support = False
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.viewer.eye = (0.3, 1.4, 0.2)
        self.viewer.env_index = 14
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if self.enable_terrain_curriculum:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RTv6RoughEnvCfg_PLAY(RTv6RoughEnvCfg):
    """Smaller rough scene for play / debugging."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

