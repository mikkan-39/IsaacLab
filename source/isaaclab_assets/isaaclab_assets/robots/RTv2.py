
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

RTV2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Mikkan/Documents/IsaacBS/Robot_URDF/AnotherAttempt.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*HipR_to.*": -0.4,
            ".*HipL_to.*": 0.6,
            ".*to_HipR.*": 0.2,
            ".*to_HipL.*": -0.3,
            ".*to_FootJointR.*": -0.2,
            ".*to_FootJointL.*": -0.3,

            # ".*HipR_to.*": -0.4,
            # ".*HipL_to.*": 0.4,
            # ".*to_HipR.*": 0.2,
            # ".*to_HipL.*": -0.2,
            # ".*to_FootJointR.*": -0.2,
            # ".*to_FootJointL.*": -0.2,

            ".*to_Arm.*": -1.6,
            # "base_link_to_shoulder_joint_v1_revolute": -0.78,
            # "base_link_to_shoulder_joint_v1Mirror_revolute": 0.78,
            # ".*to_Shoulder.*": -0.3
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "ST3215-HS": ImplicitActuatorCfg(
            joint_names_expr=[
                "^(?!.*FootJoint).*"
                # ".*"
            ],
            effort_limit_sim={
                ".*": 2.0
            },
            velocity_limit=150.0,
            velocity_limit_sim=150.0,
            stiffness={
                ".*": 2.0
            },
            damping={
                ".*": 0.2
            },
            armature={
                ".*": 0.01
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*FootJoint.*"
            ],
            effort_limit={
                ".*to_FootJoint.*": 0.75,
                ".*Foot_revolute.*": 0.75
            },
            velocity_limit=100.0,
            velocity_limit_sim=150.0,
            stiffness={
                ".*": 2.0
            },
            damping={
                ".*": 0.2
            },
            armature={
                ".*": 0.01
            },
        )
    },
)

# actuators={
#         "ST3215-HS": ImplicitActuatorCfg(
#             joint_names_expr=[".*"],
#             stiffness= 100.0,
#             damping=10.0,
#             # friction=0.1,
#             effort_limit=300.0,
#             effort_limit_sim=300.0,
#             # armature=0.001,
#             velocity_limit=200.0,
#             velocity_limit_sim=200.0
#         ),
#     },