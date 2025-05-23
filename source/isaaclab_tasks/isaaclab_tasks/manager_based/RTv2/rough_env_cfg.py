# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import RTV2_CFG  # isort: skip

@configclass
class RTv2Rewards(RewardsCfg):
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.5, 
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot.*"),
            "threshold": 0.4,
        },
    )
    # leg_accel = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=0.125e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*to_HipR.*", ".*to_HipL.*", ".*to_Tibia.*"])},
    # )

    both_feet_contacts = RewTerm(
        func=mdp.undesired_contacts_all,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot")},
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot"),
        },
    )
    joint_deviation_feet = RewTerm(
        func=mdp.joint_same_direction_deviation_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])},
    )
    joint_deviation_hip_spread = RewTerm(
        func=mdp.joint_same_direction_deviation_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])},
    )
    # joint_deviation_step = RewTerm(
    #     func=mdp.joint_diff_direction_deviation_penalty,
    #     weight=0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])},
    # )
    # joint_deviation_knees = RewTerm(
    #     func=mdp.two_joint_deviation_penalty,
    #     weight=0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*to_Tibia.*"])},
    # )
    joint_deviation_hip_rotate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_revolute"])},
    )
    # joint_pos_ankle = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])},
    # )
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 "base_link_to_shoulder.*",
    #                 ".*to_Shoulder.*",
    #                 ".*Elbow.*"
    #             ]
    #         )
    #     },
    # )



@configclass
class RTv2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RTv2Rewards = RTv2Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = RTV2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

        # Randomization
        self.events.push_robot = None # type: ignore
        self.events.add_base_mass = None # type: ignore
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), "roll": (-0.0, 0.1), "pitch": (-0.05, 0.05),},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # self.observations.policy.height_scan = None # type: ignore

        # Rewards
        self.rewards.flat_orientation_l2.weight = -1.0
    
        self.rewards.lin_vel_z_l2 = None # type: ignore
        self.rewards.action_rate_l2 = None # type: ignore
        self.rewards.dof_acc_l2 = None # type: ignore
        self.rewards.dof_torques_l2 = None # type: ignore
        self.rewards.undesired_contacts = None # type: ignore
        self.rewards.dof_pos_limits.weight = -0.1

        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.undesired_contacts = None # type: ignore
        # self.rewards.action_rate_l2.weight = 0
        # self.rewards.dof_acc_l2.weight = 0
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*Hip.*", ".*to_FootJoint.*"]
        # )
        # self.rewards.dof_torques_l2.weight = 0
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*Hip.*", ".*FootJoint.*"]
        # )


@configclass
class RTv2RoughEnvCfg_PLAY(RTv2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, -1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None # type: ignore
        self.events.push_robot = None # type: ignore
