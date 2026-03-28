import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp2

from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg, controllableJointsRegex

from isaaclab_assets import RT_CFG 

#  [
# 'base_link_to_LeftHipBracket_revolute', 
# 'base_link_to_RightHipBracket_revolute', 
# 'base_link_to_shoulder_joint_v1Mirror_revolute', 
# 'base_link_to_shoulder_joint_v1_revolute', 
# 'LeftHipBracket_to_HipBulkL_revolute', 
# 'RightHipBracket_to_HipBulkR_revolute', 
# 'HipBulkL_to_HipL_revolute', 
# 'HipBulkR_to_HipR_revolute', 
# 'HipL_to_TibiaL_revolute', 
# 'HipR_to_TibiaR_revolute', 
# 'TibiaL_to_FootJointL_revolute', 
# 'TibiaR_to_FootJointR_revolute', 
# 'FootJointL_to_LeftFoot_revolute', 
# 'FootJointR_to_RightFoot_revolute'
# ]

# Controllable joints (14 total, excluding Neck, Elbow, Arm, Shoulders):
# Index 0: base_link_to_LeftHipBracket_revolute  <-- TARGET
# Index 1: base_link_to_RightHipBracket_revolute
# Index 2-3: HipBracket_to_HipBulk (L/R)
# Index 4-5: HipBulk_to_Hip (L/R)
# Index 6-7: Hip_to_Tibia (L/R)
# Index 8-9: Tibia_to_FootJoint (L/R)
# Index 10-11: FootJoint_to_Foot (L/R)
# (adjust indices based on actual preserve_order output)

# name, action index, observation index, action direction, observation direction is the same as action direction for all tested joints
# base_link_to_LeftHipBracket_revolute, 0, 0, + is inward 
# base_link_to_RightHipBracket_revolute, 1, 1, + is outward 
# base_link_to_shoulder_joint_v1Mirror_revolute (left shoulder), 2, 2, + is backward 
# base_link_to_shoulder_joint_v1_revolute (right shoulder), 3, 3, + is forward 
# LeftHipBracket_to_HipBulkL_revolute, 4, 4, + is inward 
# RightHipBracket_to_HipBulkR_revolute, 5, 5, + is inward 
# HipBulkL_to_HipL_revolute, 6, 6, + is backward 
# HipBulkR_to_HipR_revolute, 7, 7, + is forward 
# HipL_to_TibiaL_revolute, 8, 8, + is bend 
# HipR_to_TibiaR_revolute, 9, 9, - is bend 
# TibiaL_to_FootJointL_revolute, 10, 10, - is toe up 
# TibiaR_to_FootJointR_revolute, 11, 11, - is toe up 
# FootJointL_to_LeftFoot_revolute, 12, 12, + is foot inward 
# FootJointR_to_RightFoot_revolute, 13, 13, + is foot inward 

@configclass
class RTv5Rewards:
    # Reward action magnitude for target joint (action index 0)
    # Use positive weight to reward non-zero actions
    target_action = RewTerm(
        func=mdp.action_magnitude_l1,
        weight=10.0,
        params={"action_ids": [13]},  # First action = first controllable joint
    )
    # Penalize action magnitude for all OTHER joints (indices 1-13)
    other_actions = RewTerm(
        func=mdp.action_magnitude_l2,
        weight=-1.0,
        params={"action_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    )
    # Smooth actions
    actions_cost_diff = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01,
    )

    action_clip_violation = RewTerm(
        func=mdp.action_clip_violation,
        weight=-100,
        params={"clip_min": -1.0, "clip_max": 1.0},
    )


@configclass
class RTv5TestTerminations:
    """Only timeout, no other terminations."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[controllableJointsRegex]
            ),
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0),"yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, -0.0),
                "pitch": (-1.0, -1.0),
                "yaw": (-0.0, -0.0),
            },
        },
    )



@configclass
class RTv5TestEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RTv5Rewards = RTv5Rewards()
    terminations: RTv5TestTerminations = RTv5TestTerminations()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = RT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None # type: ignore

