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
        usd_path="C:/Users/Mikkan/Documents/IsaacBS/Robot_URDF_Gazebo/Robot_clean/Robot_Force_Corrected.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, # TODO
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34), # TODO
        joint_pos={".*": 0.0},
    ),
    actuators={
        "ST3215-HS": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*": 10.0,
            },
            damping={
                ".*": 1.0,
            },
            effort_limit={
                ".*": 3.0,
            }
        ),
    },
)
