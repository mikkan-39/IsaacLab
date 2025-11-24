import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.minirough import MINI_ROUGH_TERRAINS_CFG  # isort: skip

# controllableJointsRegex = "^(?!.*(Neck|to_Elbow|to_Arm|to_Shoulder|shoulder)).*$"
# controllableJointsRegex = "^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL|Foot)).*$"
controllableJointsRegex = "^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL)).*$"

##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MINI_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            # dynamic_friction=2.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            # mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            # mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(1.0, 1.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING # type: ignore
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=1.0,
    #     debug_vis=False,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.3, 0.3), heading=(-math.pi, math.pi)
    #     ),
    # )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[controllableJointsRegex], 
                                           scale=1.0, 
                                           use_default_offset=True,
                                           preserve_order=True,
                                           clip={".*": (-1.0, 1.0)}
                                           )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_acc = ObsTerm(
            func=mdp.base_lin_acc, 
            # noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_acc = ObsTerm(
            func=mdp.base_ang_acc, 
            # noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            # noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            # noise=Unoise(n_min=-0.01, n_max=0.01), 
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[controllableJointsRegex]
            )}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            # noise=Unoise(n_min=-0.2, n_max=0.2), 
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[controllableJointsRegex]
            )}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
    #         "mass_distribution_params": (0.75, 1.25),
    #         "operation": "scale",
    #     },
    # )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 1.0),
    #     params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        
    # )

    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
    #         "stiffness_distribution_params": (1.0, 30.0),
    #         # "damping_distribution_params": (0.25, 0.5),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )


@configclass
class RewardsCfg:
    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # fall = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    # )
    fall = DoneTerm(
        func=mdp.bad_orientation,
        #45 degrees = 0.78 rad
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}
    )
    hand_or_base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*Arm.*", ".*base.*"]), "threshold": 1.0},
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # type: ignore


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 1 / 50
        self.sim.render_interval = 1
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = "cuda:0"
        self.sim.enable_scene_query_support = False
        
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
