# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import use_stage

from .rtv6_action import RTv6DelayedJointPositionController
from .rtv6_constants import CONTROLLABLE_JOINTS_REGEX
from .rtv6_observations import (
    gait_metrics,
    gait_phase_sincos,
    joint_pos_rel_subset,
    projected_gravity_b,
)
from .rtv6_rewards import compute_rtv6_rough_reward
from .rtv6_rough_env_cfg import RTv6RoughEnvCfg


class _ManagerEnvShim:
    """Minimal surface expected by :class:`UniformVelocityCommand` / command terms."""

    __slots__ = ("_env",)

    def __init__(self, env: "RTv6RoughEnv"):
        self._env = env

    @property
    def scene(self):
        return self._env.scene

    @property
    def sim(self):
        return self._env.sim

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def step_dt(self) -> float:
        return self._env.step_dt


class _CommandFacade:
    __slots__ = ("_env",)

    def __init__(self, env: "RTv6RoughEnv"):
        self._env = env

    def get_command(self, name: str) -> torch.Tensor:
        if name != "base_velocity":
            raise KeyError(name)
        return self._env._cmd_base_velocity.command


class _TerminationFacade:
    __slots__ = ("_env",)

    def __init__(self, env: "RTv6RoughEnv"):
        self._env = env

    @property
    def terminated(self) -> torch.Tensor:
        return self._env._rtv6_bad_term_buf

    @property
    def time_outs(self) -> torch.Tensor:
        return self._env._rtv6_time_out_buf


class _DirectActionManagerShim:
    """``ActionManager``-compatible buffers + term dict for locomotion reward / obs helpers."""

    __slots__ = ("_env", "_action", "_prev_action", "_terms")

    def __init__(self, env: "RTv6RoughEnv", joint_term: RTv6DelayedJointPositionController) -> None:
        self._env = env
        self._terms = {"joint_pos": joint_term}
        d = joint_term.action_dim
        self._action = torch.zeros(env.num_envs, d, device=env.device)
        self._prev_action = torch.zeros_like(self._action)

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action

    @property
    def device(self):
        return self._env.device

    @property
    def total_action_dim(self) -> int:
        """Matches :class:`~isaaclab.managers.ActionManager` (used by ``RslRlVecEnvWrapper``)."""
        return self._terms["joint_pos"].action_dim

    def process_action(self, action: torch.Tensor) -> None:
        term = self._terms["joint_pos"]
        if action.shape[1] != term.action_dim:
            raise ValueError(f"Invalid action shape, expected: {term.action_dim}, received: {action.shape[1]}.")
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)
        term.process_actions(self._action)

    def apply_action(self) -> None:
        self._terms["joint_pos"].apply_to_sim()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self._terms["joint_pos"].reset(env_ids)
        return {}


class RTv6RoughEnv(DirectRLEnv):
    cfg: RTv6RoughEnvCfg

    def __init__(self, cfg: RTv6RoughEnvCfg, render_mode: str | None = None, **kwargs):
        self._cmd_base_velocity: UniformVelocityCommand | None = None
        self._joint_ctrl: RTv6DelayedJointPositionController | None = None
        super().__init__(cfg, render_mode, **kwargs)

        self.command_manager = _CommandFacade(self)
        self.termination_manager = _TerminationFacade(self)

        self._rtv6_bad_term_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rtv6_time_out_buf = torch.zeros_like(self._rtv6_bad_term_buf)

    def _setup_scene(self) -> None:
        """Scene comes from :class:`InteractiveSceneCfg`; robot is spawned by the scene."""
        pass

    def _init_rtv6_after_physics(self) -> None:
        """Joint counts, velocity command, direct joint controller, and :class:`SceneEntityCfg` resolution."""
        if self._joint_ctrl is not None:
            return

        robot = self.scene["robot"]
        if self._cmd_base_velocity is None:
            shim = _ManagerEnvShim(self)
            self._cmd_base_velocity = UniformVelocityCommand(self.cfg.commands.base_velocity, shim)  # type: ignore[arg-type]

        patterns = self.cfg.joint_action.joint_names
        if patterns is None:
            patterns = [CONTROLLABLE_JOINTS_REGEX]
        self._policy_joint_name_patterns = patterns

        self._joint_ctrl = RTv6DelayedJointPositionController(
            cfg=self.cfg.joint_action,
            robot=robot,
            joint_name_patterns=patterns,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.action_manager = _DirectActionManagerShim(self, self._joint_ctrl)
        self._n_policy_joints = self._joint_ctrl.action_dim

        self._proj_gravity_delay = torch.zeros(self.num_envs, 3, device=self.device)

        self._r6_gait_contact_sensor = SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"])
        self._r6_lin_vel_z_asset = SceneEntityCfg("robot", body_names=".*base.*")
        self._r6_base_flat_asset = SceneEntityCfg("robot", body_names=".*base.*")
        self._r6_base_ang_asset = SceneEntityCfg("robot", body_names=".*base.*")
        self._r6_speed_asset = SceneEntityCfg("robot", joint_names=patterns)
        self._r6_torque_asset = SceneEntityCfg("robot", joint_names=patterns)
        self._r6_torque_feet_asset = SceneEntityCfg("robot", joint_names=[".*FootJoint.*"])
        self._r6_dof_limits_asset = SceneEntityCfg("robot", joint_names=patterns)
        self._r6_feet_slide_sensor = SceneEntityCfg("contact_forces", body_names=".*Foot")
        self._r6_feet_slide_robot = SceneEntityCfg("robot", body_names=".*Foot")
        self._r6_hip_spread = SceneEntityCfg("robot", joint_names=[".*HipBracket_to_HipBulk.*"])
        self._r6_hip_rotate = SceneEntityCfg("robot", joint_names=[".*HipBracket_revolute"])
        self._r6_feet_main = SceneEntityCfg("robot", joint_names=[".*to_FootJoint.*"])
        self._r6_feet_secondary = SceneEntityCfg("robot", joint_names=["FootJoint.*"])
        self._r6_arms = SceneEntityCfg("robot", joint_names=["base_link_to_shoulder.*"])

        self._term_bad_orient_cfg = SceneEntityCfg("robot", body_names=".*base.*")
        self._term_illegal_sensor = SceneEntityCfg("contact_forces", body_names=[".*Arm.*", ".*base.*"])
        self._obs_joint_cfg = SceneEntityCfg("robot", joint_names=patterns)
        self._gait_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["RightFoot", "LeftFoot"])

        for c in (
            self._r6_gait_contact_sensor,
            self._r6_lin_vel_z_asset,
            self._r6_base_flat_asset,
            self._r6_base_ang_asset,
            self._r6_speed_asset,
            self._r6_torque_asset,
            self._r6_torque_feet_asset,
            self._r6_dof_limits_asset,
            self._r6_feet_slide_sensor,
            self._r6_feet_slide_robot,
            self._r6_hip_spread,
            self._r6_hip_rotate,
            self._r6_feet_main,
            self._r6_feet_secondary,
            self._r6_arms,
            self._term_bad_orient_cfg,
            self._term_illegal_sensor,
            self._obs_joint_cfg,
            self._gait_sensor_cfg,
        ):
            c.resolve(self.scene)

    def _configure_gym_env_spaces(self) -> None:
        # DirectRLEnv may skip sim.reset() when launched embedded; articulation views need a played sim.
        robot = self.scene["robot"]
        if getattr(robot, "_root_physx_view", None) is None:
            with use_stage(self.sim.get_initial_stage()):
                self.sim.reset()
            self.scene.update(dt=self.physics_dt)

        self._init_rtv6_after_physics()
        obs_dim = 6 + 3 + 2 + 2 * self._n_policy_joints
        self.cfg.observation_space = obs_dim
        self.cfg.action_space = self._n_policy_joints
        super()._configure_gym_env_spaces()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.action_manager.process_action(actions)

    def _apply_action(self) -> None:
        self.action_manager.apply_action()

    def _get_observations(self) -> dict:
        robot = self.scene["robot"]
        g0 = projected_gravity_b(robot)
        g1 = self._proj_gravity_delay
        self._proj_gravity_delay = g0.clone()
        cmd = self.command_manager.get_command("base_velocity")
        gphase = gait_phase_sincos(self)
        jpos = joint_pos_rel_subset(robot, self._obs_joint_cfg.joint_ids)
        lact = self.action_manager.action
        obs = torch.cat((g0, g1, cmd, gphase, jpos, lact), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Match :class:`RewardManager.compute`: each term is ``weight * f(env) * dt`` (see reward_manager.py).
        # ``compute_rtv6_rough_reward`` is the ``weight * f`` sum only.
        return compute_rtv6_rough_reward(self) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        to = mdp.time_out(self)
        bo = mdp.bad_orientation(self, limit_angle=1.0, asset_cfg=self._term_bad_orient_cfg)
        ic = mdp.illegal_contact(self, threshold=1.0, sensor_cfg=self._term_illegal_sensor)
        bad = bo | ic
        self._rtv6_bad_term_buf[:] = bad
        self._rtv6_time_out_buf[:] = to
        return bad, to

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if self.cfg.enable_terrain_curriculum and self.scene.terrain.cfg.terrain_generator is not None:
            mdp.terrain_levels_vel(self, env_ids_t)
        if self.cfg.enable_gait_curriculum:
            gait_metrics(self, env_ids_t, sensor_cfg=self._gait_sensor_cfg)

        self.scene.reset(env_ids_t)

        if self.cfg.events and "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids_t, global_env_step_count=env_step_count)

        self.action_manager.reset(env_ids_t)
        self._cmd_base_velocity.reset(env_ids_t)

        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids_t)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids_t)

        self.episode_length_buf[env_ids_t] = 0

        g0 = projected_gravity_b(self.scene["robot"])
        self._proj_gravity_delay[env_ids_t] = g0[env_ids_t].clone()

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        self._cmd_base_velocity.compute(self.step_dt)

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
