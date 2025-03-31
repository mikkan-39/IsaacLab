
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

RTV2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Mikkan/Documents/IsaacBS/Robot_URDF/Robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33), # TODO
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "ST3215-HS": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness= 10.0,
            damping=1.0,
            # friction=0.1,
            effort_limit=3.0,
            # armature=0.001,
            velocity_limit=2
        ),
    },
)