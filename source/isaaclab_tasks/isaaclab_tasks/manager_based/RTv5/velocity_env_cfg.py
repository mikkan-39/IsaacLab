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
from isaaclab_tasks.manager_based.RTv5.delayed_backlash_action import DelayedBacklashJointPositionActionCfg
import torch

GAIT_FREQ_RANGE = (1.0, 1.5)  # Hz — per-env random frequency range


def _get_gait_freq(env) -> torch.Tensor:
    """Return per-env gait frequency tensor, creating and sampling it if needed."""
    if not hasattr(env, "_gait_freq"):
        env._gait_freq = torch.empty(env.num_envs, device=env.device).uniform_(*GAIT_FREQ_RANGE)
    return env._gait_freq


def _resample_gait_freq(env, env_ids):
    """Resample gait frequency for given env_ids. Called from gait_metrics on reset."""
    freq = _get_gait_freq(env)
    n = len(env_ids) if not isinstance(env_ids, slice) else env.num_envs
    freq[env_ids] = torch.empty(n, device=env.device).uniform_(*GAIT_FREQ_RANGE)


def gait_phase_obs(env) -> torch.Tensor:
    """Observation: sin/cos of the per-env gait phase clock. Shape (num_envs, 2)."""
    freq = _get_gait_freq(env)
    phase = 2.0 * torch.pi * freq * env.episode_length_buf.float() * env.step_dt
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


def randomize_actuator_effort_limit(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_range: tuple[float, float] = (1.5, 1.96),
):
    """Randomize the effort_limit on DCMotor-based actuators and recompute derived values.

    Tier-1 #2: real bus servos lose peak torque as battery voltage drops. The
    ST3215 spec'd at 1.96 Nm@12V loses ~25% of peak torque by the time the pack
    sits at 10V. Without exposure to this in sim, the policy plans assuming
    full torque authority every step — and falls the first time the hips need
    to push against a low-SoC battery. Cycling effort_limit per reset across
    a plausible voltage envelope teaches torque-saturation robustness.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    for actuator in asset.actuators.values():
        if not hasattr(actuator, "_vel_at_effort_lim"):
            continue
        new_eff = torch.empty(len(env_ids), actuator.num_joints, device=asset.device).uniform_(*effort_range)
        actuator.effort_limit[env_ids] = new_eff
        # `_vel_at_effort_lim` is the velocity at which the motor model crosses
        # over from torque-limited to velocity-limited. Depends on both vel and
        # effort limits, so recompute when either changes.
        actuator._vel_at_effort_lim[env_ids] = actuator.velocity_limit[env_ids] * (
            1.0 + new_eff / actuator._saturation_effort
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
            # Tier-1 #1: previously friction_combine_mode="multiply" with
            # static_friction=dynamic_friction=3.0 effectively welded the feet
            # to the floor and prevented any slipping in sim. Switch to "min"
            # (effective μ becomes the smaller of foot vs ground, which is the
            # realistic worst case) and use physical friction values. Per-env
            # randomization on top is handled by `robot_foot_material` event.
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.05, 0.35), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_speed = mdp.JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=[controllableJointsRegex],
    #     scale=5.0,
    #     use_default_offset=True,
    #     preserve_order=True,
    #     clip={".*": (-1.0, 1.0)},
    # )

    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[controllableJointsRegex],
    #     scale=1.0,
    #     use_default_offset=True,
    #     preserve_order=True,
    #     clip={".*": (-2.0, 2.0)},
    # )

    joint_integrated_pos = DelayedBacklashJointPositionActionCfg(
        asset_name="robot",
        joint_names=[controllableJointsRegex],
        # `scale` is unused in delta mode; per-step magnitude is `delta_scale`.
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
        # Tier-2 #4: delta-integrated targets instead of absolute targets.
        # ~0.2 rad/step at 50 Hz caps slew rate at ~572 deg/s under unit action,
        # matching what real ST3215-HS-class servos can track without saturating.
        delta_scale=0.3,
        # Tier-3 #11: stochastically perturb action history at reset so the
        # policy learns to recover from non-default startup states (handed
        # control from stand-up routine, hot restarts on hardware, etc.).
        reset_history_jitter_std=0.00,
        reset_history_jitter_prob=0.00,
        min_delay_steps=0,
        max_delay_steps=0,
        backlash_deg=0.0,
        # Tier-2 #5: enable servo position jitter. ~0.007 rad ≈ 0.4° matches
        # bus-servo step quantization (~0.087°/count) plus mechanical jitter.
        action_noise_std=0.000,
        action_lpf_alpha=1.0,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Tier-3 #9 note: these observations read the abstract base orientation,
        # not a physically mounted IMU. If the real IMU sits on a sub-link (head,
        # torso plate) and samples async, switch these to mdp.imu_* by adding an
        # ImuCfg to MySceneCfg with the correct prim_path/offset and update_period.
        # Leaving placement to whoever knows the actual hardware mount geometry.

        # Accelerometer with gravity (like real IMU)
        # base_lin_acc = ObsTerm(
        #     func=mdp.base_lin_acc_with_gravity,
        #     noise=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        #     params={"gravity_bias": (0.0, 0.0, 9.81)},
        # )
        # base_ang_vel = ObsTerm(
        #     func=mdp.base_ang_vel,
        #     noise=GaussianNoiseCfg(mean=0.0, std=0.04, operation="add"),
        # )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.025, operation="add"),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        gait_phase = ObsTerm(func=gait_phase_obs)
        # Tier-2 #6: joint position readback from a serial-bus servo is laggy
        # *and* quantized, not the clean float you get from `joint_pos_rel`.
        # Add Gaussian noise to approximate quantization (~0.0015 rad = 1 count
        # on a 12-bit servo, plus mechanical wobble), and use the existing
        # DelayedObservationCfg modifier to introduce variable readback lag.
        # Keeping joint position observation (rather than dropping it like Bimo)
        # may be worth considering,
        # because RT has more DoF than Bimo and benefits from proprioception,
        # provided the *modelled* readback matches what the real bus provides.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.003, operation="add"),
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[controllableJointsRegex]
            )},
            modifiers=[
                DelayedObservationCfg(
                    min_lag=0,
                    max_lag=2,
                    per_env=True,
                    hold_prob=0.9,
                    update_period=1,
                )
            ],
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Tier-2 #6 (related): `enable_corruption=False` silently disables
            # every NoiseCfg attached to every ObsTerm. This was the root cause
            # of the entire observation pipeline being noise-free in sim while
            # the real robot's observations are heavily noisy. Enable.
            self.enable_corruption = True
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
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                # Tier-3 #13: prior config had roll/pitch = (-0.1, -0.1), i.e.
                # both bounds equal, which is a constant initial spin rather
                # than randomization. Fixed to a symmetric range so episodes
                # don't all start with the same biased tipping.
                "x": (0.2, 0.2),
                "y": (-0.05, 0.05),
                "z": (-0.0, 0.0),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # Tier-2 #7: replace single base-only one-shot mass randomization with two
    # complementary events:
    #   (a) base ±25% on startup — accounts for battery presence, top-plate
    #       payload, cabling. Once per env at simulation startup.
    #   (b) every link ±5% on reset, with inertia recomputed — accounts for
    #       3D-print density variation, mass distribution along limbs, and
    #       trains the policy against per-episode dynamics shifts.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
            "mass_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_all_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # Tier-3 (COM): real robots have COM uncertainty from battery placement,
    # cabling routing, and assembly tolerances. ±15 mm on base translates to
    # noticeable balance shifts that a sim policy must tolerate.
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
            "com_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.03, 0.03),
            },
        },
    )

    # Tier-1 #1 (companion to terrain physics fix): per-env randomization of
    # foot friction and restitution. The terrain's μ is now physical (0.7-0.8),
    # but real floors vary wildly between sessions (polished concrete, rubber,
    # ceramic tile, lab linoleum). Randomizing the robot's foot material in
    # buckets exposes the policy to the full range.
    robot_foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 10.0),
    #     params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    # )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # robot_velocity_limit = EventTerm(
    #     func=randomize_actuator_velocity_limit,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
    #         "velocity_range": (11.1 * 0.75, 11.1 * 1.5),
    #     },
    # )

    # Tier-1 #2: battery-voltage-droop model. ST3215 spec is 1.96 Nm @ 12V; a
    # 10V pack delivers roughly 1.5 Nm. Sampling per reset across that range
    # forces the policy to handle torque-limited stance and pushoff, which is
    # the most common cause of policy collapse on a partially discharged pack.
    # robot_effort_limit = EventTerm(
    #     func=randomize_actuator_effort_limit,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[controllableJointsRegex]),
    #         "effort_range": (1.5, 1.96),
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
    min_air_time: float = 0.04,
) -> dict[str, float]:
    """Passive gait metrics: per-foot step counts, frequency, symmetry, and swing time.

    Only counts a touchdown as a "step" if the foot was airborne for at least
    ``min_air_time`` seconds before landing. This filters out physics contact
    bouncing that inflates step counts.

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

    # Only count as a real step if preceding air time exceeded min_air_time
    real_step_right = first_contact[:, 0] & (env._gait_prev_air_time[:, 0] > min_air_time)
    real_step_left = first_contact[:, 1] & (env._gait_prev_air_time[:, 1] > min_air_time)

    env._gait_steps_right += real_step_right.float()
    env._gait_steps_left += real_step_left.float()
    env._gait_elapsed += env.step_dt

    elapsed = env._gait_elapsed.clamp(min=0.1)
    total_steps = env._gait_steps_right + env._gait_steps_left
    freq = total_steps / elapsed

    # Symmetry: min/max of per-foot counts (1.0 = perfect, 0.0 = one-legged)
    max_steps = torch.max(env._gait_steps_right, env._gait_steps_left).clamp(min=1.0)
    min_steps = torch.min(env._gait_steps_right, env._gait_steps_left)
    symmetry = min_steps / max_steps

    # Swing duration: only from real steps (filtered)
    for foot_idx, real_step in enumerate([real_step_right, real_step_left]):
        env._gait_swing_sum += torch.where(real_step, env._gait_prev_air_time[:, foot_idx], torch.zeros_like(env._gait_swing_sum))
        env._gait_swing_count += real_step.float()

    mean_swing = env._gait_swing_sum / env._gait_swing_count.clamp(min=1.0)

    # Cache current air_time for next step (read AFTER using the previous cache)
    env._gait_prev_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids].clone()

    # Reset counters and resample gait frequency for terminated envs
    env._gait_steps_right[env_ids] = 0.0
    env._gait_steps_left[env_ids] = 0.0
    env._gait_elapsed[env_ids] = 0.0
    env._gait_swing_sum[env_ids] = 0.0
    env._gait_swing_count[env_ids] = 0.0
    _resample_gait_freq(env, env_ids)

    gait_freq = _get_gait_freq(env)

    return {
        "step_freq_hz": freq.mean().item(),
        "symmetry": symmetry.mean().item(),
        "mean_swing_s": mean_swing.mean().item(),
        "steps_right": env._gait_steps_right.mean().item(),
        "steps_left": env._gait_steps_left.mean().item(),
        "gait_clock_hz": gait_freq.mean().item(),
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
        # general settings.
        # Tier-1 #3 (verify on hardware): decimation=4 at sim_dt=1/200 gives 50 Hz
        # control. Bimo runs at 20 Hz on hardware (their decimation=10) because
        # serial-bus servo round-trip at 8+ joints often can't sustain 50 Hz on
        # real hardware. Measure your actual control loop period on the robot;
        # if it's >25 ms, switch to decimation=8 (25 Hz) or 10 (20 Hz) and retrain.
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
