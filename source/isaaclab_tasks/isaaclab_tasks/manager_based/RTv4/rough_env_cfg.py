import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets import RT_CFG 

@configclass
class RTv4Rewards:
    
    # alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=10.0, 
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=3.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # added
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )


    # base_pos_x = RewTerm(
    #     func=mdp.flat_orientation_l2_x, 
    #     weight=-5.0, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )
    # base_pos_y = RewTerm(
    #     func=mdp.flat_orientation_l2_y, 
    #     weight=-2.5, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )

    actions_cost_diff = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.001,
    )

    hip_vel_same_sign = RewTerm(
        func=mdp.hip_vel_same_sign, 
        weight=1.0, 
        params={
            "command_name": "base_velocity",
            "asset_cfg_a": SceneEntityCfg("robot", joint_names=[".*to_HipR.*"]),
            "asset_cfg_b": SceneEntityCfg("robot", joint_names=[".*to_HipL.*"])
            })
  
    # torque_cost = RewTerm(
    #     func=mdp.joint_torques_l2, 
    #     weight=-1.5e-3, 
    # )

    # torque_cost_feet = RewTerm(
    #     func=mdp.joint_torques_l2, 
    #     weight=-1.5e-1, 
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])}
    # )

    # Penalize all joint limits except knees.
    dof_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*to_Tibia).*"])},
    )

    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot"),
    #     },
    # )

    joint_deviation_hip_spread = RewTerm(
        func=mdp.joint_same_direction_deviation_penalty,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])},
    )

    joint_deviation_hip_rotate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_revolute"])},
    )

    # joint_deviation_knees = RewTerm(
    #     func=mdp.joint_same_direction_deviation_penalty,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*to_Tibia.*"])},
    # )

    # joint_deviation_hips_all = RewTerm(
    #     func=mdp.joint_same_direction_deviation_penalty,
    #     weight=-15.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])},
    # )

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 "base_link_to_shoulder.*",
    #             ]
    #         )
    #     },
    # )

@configclass
class RTv4RewardsShitty:
    # -- task
    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0) # Maybe?
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=50.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
            "command_name": "base_velocity",
            "threshold": 0.2,
        },
    )



@configclass
class RTv4RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RTv4Rewards = RTv4Rewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

        # Randomization
        pass

# @configclass
# class RTv4RoughEnvCfg_PLAY(RTv4RoughEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.0
#         self.episode_length_s = 40.0
#         # spawn the robot randomly in the grid (instead of their terrain levels)
#         self.scene.terrain.max_init_terrain_level = None
#         # reduce the number of terrains to save memory
#         if self.scene.terrain.terrain_generator is not None:
#             self.scene.terrain.terrain_generator.num_rows = 5
#             self.scene.terrain.terrain_generator.num_cols = 5
#             self.scene.terrain.terrain_generator.curriculum = False

#         self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (-1.0, -1.0)
#         self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
#         self.commands.base_velocity.ranges.heading = (0.0, 0.0)
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
