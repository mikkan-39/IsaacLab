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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise, GaussianNoiseCfg
from isaaclab.utils.modifiers import DelayedObservationCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

GAIT_FREQ = 1.5  # Hz — shared between observation and reward


def gait_phase_obs(env, gait_freq: float = GAIT_FREQ) -> torch.Tensor:
    """Observation: sin/cos of the gait phase clock. Shape (num_envs, 2)."""
    phase = 2.0 * torch.pi * gait_freq * env.episode_length_buf.float() * env.step_dt
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)

##
# Pre-defined configs
##
from isaaclab.terrains.config.minirough import MINI_ROUGH_TERRAINS_CFG  # isort: skip

# controllableJointsRegex = "^(?!.*(Neck|to_Elbow|to_Arm|to_Shoulder|shoulder)).*$"
# controllableJointsRegex = "^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL|Foot)).*$"
controllableJointsRegex = "^(?!.*(Neck|to_Elbow|to_Arm|to_ShoulderR|to_ShoulderL)).*$"

def randomize_actuator_velocity_limit(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    velocity_range: tuple[float, float] = (5.24, 11.1),
):
    """Randomize the velocity_limit on DCMotor-based actuators and recompute derived values."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    for actuator in asset.actuators.values():
        if not hasattr(actuator, "_vel_at_effort_lim"):
            continue
        new_vel = torch.empty(len(env_ids), actuator.num_joints, device=asset.device).uniform_(*velocity_range)
        actuator.velocity_limit[env_ids] = new_vel
        actuator._vel_at_effort_lim[env_ids] = new_vel * (
            1.0 + actuator.effort_limit[env_ids] / actuator._saturation_effort
        )

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
            static_friction=3.0,
            dynamic_friction=3.0,
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
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True,
        force_threshold=1.0  # Restore old behavior (was bugged to 1.0, now defaults to 0.0)
    )
    
    # # Add IMU sensor at robot root/base
    # imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",  # Try articulation root itself
    #     update_period=0.0,
    #     gravity_bias=(0.0, 0.0, 9.81),
    # )
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
            lin_vel_x=(0.0, 0.25), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.3, 0.3)
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
                                        #    clip={".*": (-1.0, 1.0)}
                                           )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

                # Accelerometer with gravity (like real IMU)
        base_lin_acc = ObsTerm(
            func=mdp.base_lin_acc_with_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.2, operation="add"),
            params={"gravity_bias": (0.0, 0.0, 9.81)},
            # modifiers=[
            #     DelayedObservationCfg(
            #         min_lag=0,
            #         max_lag=3,
            #         per_env=True,
            #         hold_prob=0.9,
            #         update_period=1,
            #     )
            # ],
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.2, operation="add"),
            # modifiers=[
            #     DelayedObservationCfg(
            #         min_lag=0,
            #         max_lag=3,
            #         per_env=True,
            #         hold_prob=0.9,
            #         update_period=1,
            #     )
            # ],
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.2, operation="add"),
            # modifiers=[
            #     DelayedObservationCfg(
            #         min_lag=0,
            #         max_lag=3,
            #         per_env=True,
            #         hold_prob=0.9,
            #         update_period=1,
            #     )
            # ],
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        gait_phase = ObsTerm(
            func=gait_phase_obs,
            params={"gait_freq": GAIT_FREQ},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"), 
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[controllableJointsRegex]
            )},
            # modifiers=[
            #     DelayedObservationCfg(
            #         min_lag=0,
            #         max_lag=3,
            #         per_env=True,
            #         hold_prob=0.9,
            #         update_period=1,
            #     )
            # ],
        )
        joint_pos_t1 = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"), 
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[controllableJointsRegex]
            )},
            modifiers=[DelayedObservationCfg(
                min_lag=1, 
                max_lag=1, 
                per_env=False,
                update_period=0)],
        )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel, 
        #     noise=GaussianNoiseCfg(mean=0.0, std=0.2, operation="add"), 
        #     params={"asset_cfg": SceneEntityCfg(
        #         "robot", joint_names=[controllableJointsRegex]
        #     )}
        # )
        actions = ObsTerm(func=mdp.last_action)
        action_t1 = ObsTerm(
            func=mdp.last_action,
            modifiers=[DelayedObservationCfg(
                min_lag=1, 
                max_lag=1, 
                per_env=False,
                update_period=0)],
        )
        action_t2 = ObsTerm(
            func=mdp.last_action,
            modifiers=[DelayedObservationCfg(
                min_lag=2, 
                max_lag=2, 
                per_env=False,
                update_period=0)],
        )
        action_t3 = ObsTerm(
            func=mdp.last_action,
            modifiers=[DelayedObservationCfg(
                min_lag=3, 
                max_lag=3, 
                per_env=False,
                update_period=0)],
        )

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
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),"yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, -0.0),
                "pitch": (-0.0, -0.0),
                "yaw": (-0.0, -0.0),
            },
        },
    )

    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 10.0),
    #     params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
        
    # )

    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
    #         "stiffness_distribution_params": (0.75, 1.25),
    #         "damping_distribution_params": (0.75, 1.25),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # robot_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
    #         "friction_distribution_params": (0.1, 0.3),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )


    robot_velocity_limit = EventTerm(
        func=randomize_actuator_velocity_limit,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
            "velocity_range": (5.24, 11.1),
        },
    )



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



def adaptive_reward_ramp(
    env,
    env_ids,
    term_names: list[str],
    command_name: str = "base_velocity",
    min_scale: float = 0.1,
    max_scale: float = 1.0,
    increase_threshold: float = 0.12,
    decrease_threshold: float = 0.25,
    step_size: float = 0.02,
    smoothing: float = 0.99,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, float]:
    """Curriculum that adapts reward weights based on velocity tracking quality.

    Tracks an EMA of XY velocity error. When error drops below
    ``increase_threshold`` (good tracking), scale increases. When it rises
    above ``decrease_threshold`` (struggling), scale decreases. This lets the
    policy first learn to walk, then get progressively pressured toward
    better step quality.

    Base weights are captured on first call; scaling is always relative
    to those original values.
    """
    import torch
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    if not hasattr(env, "_adapt_base_weights"):
        base = {}
        for name, cfg in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs):
            if name in term_names:
                base[name] = cfg.weight
        env._adapt_base_weights = base
        env._adapt_scale = min_scale
        env._adapt_vel_error_ema = 0.3

    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    vel_error = torch.norm(cmd[:, :2] - vel_yaw[:, :2], dim=1).mean().item()

    env._adapt_vel_error_ema = smoothing * env._adapt_vel_error_ema + (1 - smoothing) * vel_error

    ema = env._adapt_vel_error_ema
    if ema < increase_threshold:
        env._adapt_scale = min(env._adapt_scale + step_size, max_scale)
    elif ema > decrease_threshold:
        env._adapt_scale = max(env._adapt_scale - step_size, min_scale)

    scale = env._adapt_scale
    for name, cfg in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs):
        if name in env._adapt_base_weights:
            cfg.weight = env._adapt_base_weights[name] * scale

    return {"step_reward_scale": scale, "vel_error_ema": ema}


def gait_metrics(
    env,
    env_ids,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
) -> dict[str, float]:
    """Passive gait metrics: per-foot step counts, frequency, symmetry, and swing time.

    body_ids[0] = right foot, body_ids[1] = left foot (matching body_names order).
    """
    import torch
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    if not hasattr(env, "_gait_steps_right"):
        env._gait_steps_right = torch.zeros(env.num_envs, device=env.device)
        env._gait_steps_left = torch.zeros(env.num_envs, device=env.device)
        env._gait_elapsed = torch.zeros(env.num_envs, device=env.device)
        env._gait_prev_air_time = torch.zeros(env.num_envs, 2, device=env.device)
        env._gait_swing_sum = torch.zeros(env.num_envs, device=env.device)
        env._gait_swing_count = torch.zeros(env.num_envs, device=env.device)

    env._gait_steps_right += first_contact[:, 0].float()
    env._gait_steps_left += first_contact[:, 1].float()
    env._gait_elapsed += env.step_dt

    elapsed = env._gait_elapsed.clamp(min=0.1)
    total_steps = env._gait_steps_right + env._gait_steps_left
    freq = total_steps / elapsed

    # Symmetry: min/max of per-foot counts (1.0 = perfect, 0.0 = one-legged)
    max_steps = torch.max(env._gait_steps_right, env._gait_steps_left).clamp(min=1.0)
    min_steps = torch.min(env._gait_steps_right, env._gait_steps_left)
    symmetry = min_steps / max_steps

    # Swing duration: use PREVIOUS step's air_time (before first_contact resets it to 0)
    for foot_idx in range(2):
        landed = first_contact[:, foot_idx]
        env._gait_swing_sum += torch.where(landed, env._gait_prev_air_time[:, foot_idx], torch.zeros_like(env._gait_swing_sum))
        env._gait_swing_count += landed.float()

    mean_swing = env._gait_swing_sum / env._gait_swing_count.clamp(min=1.0)

    # Cache current air_time for next step (read AFTER using the previous cache)
    env._gait_prev_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids].clone()

    # Reset counters for terminated envs
    env._gait_steps_right[env_ids] = 0.0
    env._gait_steps_left[env_ids] = 0.0
    env._gait_elapsed[env_ids] = 0.0
    env._gait_swing_sum[env_ids] = 0.0
    env._gait_swing_count[env_ids] = 0.0

    return {
        "step_freq_hz": freq.mean().item(),
        "symmetry": symmetry.mean().item(),
        "mean_swing_s": mean_swing.mean().item(),
        "steps_right": env._gait_steps_right.mean().item(),
        "steps_left": env._gait_steps_left.mean().item(),
    }


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # type: ignore

    gait_monitor = CurrTerm(
        func=gait_metrics,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"]),
        },
    )

    # step_reward_ramp = CurrTerm(
    #     func=adaptive_reward_ramp,
    #     params={
    #         "term_names": ["step_distance", "step_symmetry"],
    #         "command_name": "base_velocity",
    #         "min_scale": 1.0,
    #         "max_scale": 10.0,
    #         "increase_threshold": 0.12,
    #         "decrease_threshold": 0.25,
    #         "step_size": 0.02,
    #         "smoothing": 0.99,
    #     },
    # )


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
        self.decimation = 4
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = "cuda:0"
        self.sim.enable_scene_query_support = False

        self.sim.physx.enable_stabilization = True
        
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.viewer.eye = (0.3, 1.4, 0.2)  # Camera position (x, y, z)
        self.viewer.env_index = 14
        # self.viewer.origin_type = "robot" 
        self.viewer.origin_type = "asset_root"  # Track the robot's root
        self.viewer.asset_name = "robot"  # Asset to track

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
