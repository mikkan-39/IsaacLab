# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import RT_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.RT.locomotion_env import LocomotionEnv


@configclass
class RTEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 22
    observation_space = 78
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation) # TODO dt
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = RT_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore
    joint_gears: list = [
        -1.0, #   'base_link_to_LeftHipBracket_revolute', 
        1.0, #   'base_link_to_Neck_revolute', 
        1.0, #   'base_link_to_RightHipBracket_revolute', 
        1.0, #   'base_link_to_shoulder_joint_v1Mirror_revolute', 
        1.0, #   'base_link_to_shoulder_joint_v1_revolute', 
        1.0, #   'LeftHipBracket_to_HipBulkL_revolute', 
        1.0, #   'Neck_to_Head_revolute', 
        1.0, #   'RightHipBracket_to_HipBulkR_revolute', 
        1.0, #   'shoulder_joint_v1Mirror_to_ShoulderL_revolute', 
        1.0, #   'shoulder_joint_v1_to_ShoulderR_revolute', 
        -1.0, #   'HipBulkL_to_HipL_revolute', 
        1.0, #   'HipBulkR_to_HipR_revolute', 
        1.0, #   'ShoulderL_to_ElbowL_revolute', 
        1.0, #   'ShoulderR_to_ElbowR_revolute', 
        1.0, #   'HipL_to_TibiaL_revolute', 
        -1.0, #   'HipR_to_TibiaR_revolute', 
        1.0, #   'ElbowL_to_ArmL_revolute', 
        1.0, #   'ElbowR_to_ArmR_revolute', 
        1.0, #   'TibiaL_to_FootJointL_revolute', 
        1.0, #   'TibiaR_to_FootJointR_revolute', 
        1.0, #   'FootJointL_to_LeftFoot_revolute', 
        1.0, #   'FootJointR_to_RightFoot_revolute'
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.25

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class RTEnv(LocomotionEnv):
    cfg: RTEnvCfg

    def __init__(self, cfg: RTEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
