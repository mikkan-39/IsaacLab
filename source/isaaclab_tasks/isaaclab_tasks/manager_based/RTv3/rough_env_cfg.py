from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets import RT_CFG 

@configclass
class RTv3Rewards:
    
    # alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.5, 
        params={"command_name": "base_velocity", "std": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    )


    actions_cost = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01, 
    )
    torque_cost = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-0.05, 
    )
    dof_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-0.01, 
    )
    
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.35,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot.*"),
    #         "threshold": 0.4,
    #     },
    # )
    # both_feet_contacts = RewTerm(
    #     func=mdp.undesired_contacts_all,
    #     weight=-10.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot")},
    # )
    
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot"),
    #     },
    # )
    # joint_deviation_feet = RewTerm(
    #     func=mdp.joint_same_direction_deviation_penalty,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])},
    # )



@configclass
class RTv3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RTv3Rewards = RTv3Rewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

        # Randomization
        pass

# @configclass
# class RTv3RoughEnvCfg_PLAY(RTv3RoughEnvCfg):
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
