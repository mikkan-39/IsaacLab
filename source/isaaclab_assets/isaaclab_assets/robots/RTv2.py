
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
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
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*HipR_to.*": -0.5,
            ".*HipL_to.*": 0.5,
            ".*to_HipR.*": 0.25,
            ".*to_HipL.*": -0.25,
            ".*to_FootJointR.*": -0.25,
            ".*to_FootJointL.*": 0.25,
            ".*to_Arm.*": -1.5,
            "base_link_to_shoulder_joint_v1_revolute": -0.8,
            "base_link_to_shoulder_joint_v1Mirror_revolute": 0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "ST3215-HS": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*"
            ],
            effort_limit={
                ".*": 5.0
                # ".*Neck.*": 0.1
            },
            velocity_limit=100.0,
            stiffness={
                ".*": 3.0
            },
            damping={
                ".*": 0.3
                # ".*Neck.*": 10
            },
            armature={
                ".*": 0.01
            },
        ),
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