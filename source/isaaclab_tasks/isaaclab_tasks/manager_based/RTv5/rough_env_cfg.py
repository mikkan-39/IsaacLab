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
    alive_reward = RewTerm(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.15},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.02,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
        },
    )

    gait_contact = RewTerm(
        func=mdp.contact_gating_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
            "command_name": "base_velocity",
            "stance_ratio": 0.6,
        },
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance_capped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=["RightFoot", "LeftFoot"]),
            "target_height": 0.03,
            "min_air_time":  0.02,
        },
    )

    # -- Replaced by gait_contact --
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=2.0, 
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
    #         "command_name": "base_velocity",
    #         "threshold": 0.4,
    #     },
    # )
    # step_freq = RewTerm(
    #     func=mdp.step_frequency_penalty,
    #     weight=-0.002,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
    #         "command_name": "base_velocity",
    #         "target_freq": 3.0,
    #     },
    # )

    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    )
    
    base_pos = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-3.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    )

    base_ang_vel = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.25, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    )

    # actions_cost = RewTerm(
    #     func=mdp.action_l2,
    #     weight=-0.001, 
    # )

    actions_cost_diff = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.02,
    )

    # joint_dir_change = RewTerm(
    #     func=mdp.joint_direction_change_penalty,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["HipBulkL_to_HipL_revolute", "HipBulkR_to_HipR_revolute"]),
    #         "vel_deadband": 0.1,
    #     },
    # )

    # Tier-2 #4 (companion): with delta-mode actions the raw policy output now
    # represents a per-step *delta*, not an absolute target. A unit raw action
    # is +0.05 rad of integrated movement per step, so the meaningful action
    # range is roughly [-3, 3] sigmas with init_noise_std=1.0. Penalize only
    # gross out-of-distribution deltas; the integrator clamps to soft joint
    # limits anyway, so this is mostly a regularizer against policy drift.
    action_clip_violation = RewTerm(
        func=mdp.action_clip_violation,
        weight=-0.5,
        params={"clip_min": -3.0, "clip_max": 3.0},
    )

    # hip_vel_same_sign = RewTerm(
    #     func=mdp.hip_vel_same_sign, 
    #     weight=1.0, 
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg_a": SceneEntityCfg("robot", joint_names=[".*to_HipR.*"]),
    #         "asset_cfg_b": SceneEntityCfg("robot", joint_names=[".*to_HipL.*"])
    #         })

    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
    #         "threshold": 1.0,
    #     },
    # )

    speed_cost = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-3.0e-4, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex])}
    )

    torque_cost = RewTerm(
        func=mdp.joint_torques, 
        weight=-1.5e-5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex])}
    )

    torque_cost_feet = RewTerm(
        func=mdp.joint_torques, 
        weight=-5.0e-4, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])}
    )

    # power_cost = RewTerm(
    #     func=mdp.joint_power_l1,
    #     weight=-5e-2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex])},
    # )

    # Penalize all joint limits except un-controllable joints and knees.
    dof_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            # controllableJointsRegex.replace(")).*$", "|to_Tibia)).*$")
            controllableJointsRegex
        ])},
    )

    dof_limits_knees = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-5.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*to_Tibia.*"
        ])},
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # TODO: Try to increase this

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot"),
        },
    )

    joint_deviation_hip_spread = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])},
    )

    joint_deviation_hip_rotate = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipBracket_revolute"])},
    )

    # joint_deviation_knees = RewTerm(
    #     func=mdp.joint_same_direction_deviation_penalty,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*to_Tibia.*"])},
    # )

    joint_deviation_feet_main = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*to_FootJoint.*",
                ]
            )
        },
    )

    joint_deviation_feet_secondary = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    "FootJoint.*",
                ]
            )
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    "base_link_to_shoulder.*",
                ]
            )
        },
    )

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

        # Tier-3 #9: bump position-iteration count for the robot articulation.
        # Default solver_position_iteration_count=4 under-resolves edge contacts
        # on thin printed feet, which lets the policy learn "sticky" foot
        # behaviour that doesn't transfer. 8 is the same value Bimo uses and
        # roughly doubles the per-step cost on the robot (negligible at scale).
        robot_cfg = RT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        robot_cfg.spawn.articulation_props.solver_position_iteration_count = 8
        # Already True in the asset; explicit here so future asset edits don't
        # silently disable self-collisions for the legs/arms during swing.
        robot_cfg.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot = robot_cfg
