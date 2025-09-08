import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp2

from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg, controllableJointsRegex

from isaaclab_assets import RT_CFG 

@configclass
class RTv5Rewards:
    
    # progress = RewTerm(func=mdp2.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # move_to_target = RewTerm(
    #     func=mdp2.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    # )
    alive_reward = RewTerm(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": 0.25}
    )

   

    # added
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.1,
    #     },
    # )


    
    # base_pos = RewTerm(
    #     func=mdp.flat_orientation_l2, 
    #     weight=-1.0, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )

    # base_pos_x = RewTerm(
    #     func=mdp.flat_orientation_l2_x, 
    #     weight=-100.0, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )
    # base_pos_y = RewTerm(
    #     func=mdp.flat_orientation_l2_y, 
    #     weight=-15.0, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )

    # base_ang_vel = RewTerm(
    #     func=mdp.ang_vel_xy_l2, 
    #     weight=-0.05, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )

    # actions_cost = RewTerm(
    #     func=mdp.action_l2,
    #     weight=-0.001, 
    # )
    actions_cost_diff = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.1,
    )

    # hip_vel_same_sign = RewTerm(
    #     func=mdp.hip_vel_same_sign, 
    #     weight=1.0, 
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg_a": SceneEntityCfg("robot", joint_names=[".*to_HipR.*"]),
    #         "asset_cfg_b": SceneEntityCfg("robot", joint_names=[".*to_HipL.*"])
    #         })
  
    torque_cost = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.5e-3, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )

    # torque_cost_feet = RewTerm(
    #     func=mdp.joint_torques_l2, 
    #     weight=-1.5e-1, 
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])}
    # )

    # Penalize all joint limits except knees.
    dof_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-0.2, 
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
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])},
    )

    joint_deviation_hip_rotate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
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
    #     weight=-0.4,
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
class RTv5RewardsShitty:
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
class RTv5RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RTv5Rewards = RTv5Rewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

        # Randomization
        pass
