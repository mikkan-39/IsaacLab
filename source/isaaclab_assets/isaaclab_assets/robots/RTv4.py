from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##
RT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Mikkan/Documents/IsaacBS/Robot_URDF/Ididathing.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        # rot=(1, 0, 0, 1),
        # joint_pos={
        #     ".*HipR_to.*": -0.6,
        #     ".*HipL_to.*": 0.6,
        #     ".*to_HipR.*": 0.3,
        #     ".*to_HipL.*": -0.3,
        #     ".*to_FootJointR.*": -0.3,
        #     ".*to_FootJointL.*": -0.3,
        #     ".*to_Arm.*": -1.5,
        #     ".*to_Shoulder.*": -0.5,
        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "ST3215-HS": ImplicitActuatorCfg(
            joint_names_expr=[
                "^(?!.*FootJoint).*",
            ],
            velocity_limit_sim=50.0,
            stiffness={
                ".*": 30.0
            },
            damping={
                ".*": 5.0
            },
            armature={
                ".*": 0.01
            },
        ),
        "ST3215-HS-Feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*FootJoint.*"
            ],
            # effort_limit_sim={
            #     ".*": 0.0
            # },
            velocity_limit_sim=50.0,
            stiffness={
                ".*": 0.5
            },
            damping={
                ".*": 0.5
            },
            armature={
                ".*": 0.01
            },
        ),
    },
)
