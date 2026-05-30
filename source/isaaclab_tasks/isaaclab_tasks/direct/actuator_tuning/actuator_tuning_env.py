# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct environment that replays a recorded servo trajectory to tune DCMotor parameters."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from . import dc_motor_params as dcp
from . import metrics
from . import pd_params as pdp
from .actuator_tuning_env_cfg import ActuatorTuningEnvCfg
from .csv_replay import load_servo_trajectory


class ActuatorTuningEnv(DirectRLEnv):
    """Replays ``target_rad`` on one fixed-base joint; scores sim tracking vs. the real servo.

    Each parallel environment may hold a different DCMotor parameter vector (set via
    :meth:`set_param_batch`). All environments replay the same trajectory synchronously; when the
    replay is exhausted the per-environment scores are computed and stored in :attr:`last_scores`.
    """

    cfg: ActuatorTuningEnvCfg

    def __init__(self, cfg: ActuatorTuningEnvCfg, render_mode: str | None = None, **kwargs):
        # load + resample the recording before the sim spins up (pure numpy)
        self.trajectory = load_servo_trajectory(
            cfg.csv_path, cfg.control_hz, max_duration_s=cfg.max_duration_s, ref_lag_steps=cfg.ref_lag_steps
        )
        self._num_steps = self.trajectory.num_steps

        # split the recording into the fixed-duration excitation segments (steps/sines/sawtooths)
        seg_len_steps = max(1, int(round(cfg.segment_len_s / self.trajectory.step_dt)))
        self._segment_ids = (np.arange(self._num_steps) // seg_len_steps).astype(np.int64)

        # position target the joint will follow. For ideal_pd + an MLP checkpoint this is the MLP's
        # closed-loop (delay-free) prediction; otherwise it is the raw command target.
        self._joint_target_np = self.trajectory.target.astype(np.float32)
        if cfg.actuator_model == "ideal_pd" and cfg.mlp_checkpoint:
            from . import mlp_actuator

            mlp = mlp_actuator.load_mlp(cfg.mlp_checkpoint, device="cpu")
            self._joint_target_np = mlp_actuator.rollout_targets(
                mlp,
                self.trajectory.target,
                self.trajectory.ref_pos,
                self.trajectory.ref_vel,
                self._segment_ids,
                self.trajectory.step_dt,
            )
            print(f"[env] driving IdealPD with MLP target from checkpoint: {cfg.mlp_checkpoint}")

        # per-step error weights + reversal mask (shared across envs)
        wspec = metrics.WeightSpec(
            move_eps=cfg.weight_move_eps,
            transient_window_s=cfg.weight_transient_window_s,
            transient_weight=cfg.weight_transient,
            reversal_weight=cfg.weight_reversal,
            accel_thresh=cfg.weight_accel_thresh,
        )
        self._weights_np, self._reversal_np = metrics.compute_weights(
            self.trajectory.target, self.trajectory.step_dt, wspec
        )

        super().__init__(cfg, render_mode, **kwargs)

        # resolve the joint being tuned
        joint_ids, joint_names = self._robot.find_joints(cfg.joint_name)
        if len(joint_ids) != 1:
            raise ValueError(
                f"joint_name '{cfg.joint_name}' resolved to {len(joint_ids)} joints ({joint_names}); "
                "exactly one is required."
            )
        self._joint_idx = int(joint_ids[0])

        # widen the (tight) USD joint position limits so replayed targets are never clipped
        if cfg.override_joint_pos_limits is not None:
            lo, hi = cfg.override_joint_pos_limits
            limits = torch.empty(self.num_envs, self._robot.num_joints, 2, device=self.device)
            limits[..., 0] = lo
            limits[..., 1] = hi
            self._robot.write_joint_position_limit_to_sim(limits, warn_limit_violation=False)

        # device-side trajectory tensors
        self._target = torch.as_tensor(self.trajectory.target, dtype=torch.float32, device=self.device)
        self._joint_target = torch.as_tensor(self._joint_target_np, dtype=torch.float32, device=self.device)
        self._ref_pos = torch.as_tensor(self.trajectory.ref_pos, dtype=torch.float32, device=self.device)
        self._ref_vel = torch.as_tensor(self.trajectory.ref_vel, dtype=torch.float32, device=self.device)

        # recorded simulated trajectory (per env), and a persistent copy of the last completed run
        self._sim_pos = torch.zeros(self.num_envs, self._num_steps, device=self.device)
        self._sim_vel = torch.zeros(self.num_envs, self._num_steps, device=self.device)
        self._last_sim_pos = torch.zeros_like(self._sim_pos)
        self._step_idx = 0

        # snapshot the tunable parameters as per-(env, joint) tensors (model-dependent set)
        actuator = self._robot.actuators[cfg.actuator_name]
        j = self._joint_idx
        self._ideal_pd = cfg.actuator_model == "ideal_pd"
        if self._ideal_pd:
            self._param_values: dict[str, torch.Tensor] = {
                "stiffness": actuator.stiffness[:, j].clone(),
                "damping": actuator.damping[:, j].clone(),
                "effort_limit": actuator.effort_limit[:, j].clone(),
                "armature": actuator.armature[:, j].clone(),
                "friction": actuator.friction[:, j].clone(),
                "dynamic_friction": actuator.dynamic_friction[:, j].clone(),
                "viscous_friction": actuator.viscous_friction[:, j].clone(),
            }
        else:
            dcp.ensure_tensor_saturation(actuator)
            self._param_values = {
                "stiffness": actuator.stiffness[:, j].clone(),
                "damping": actuator.damping[:, j].clone(),
                "velocity_limit": actuator.velocity_limit[:, j].clone(),
                "effort_limit": actuator.effort_limit[:, j].clone(),
                "saturation_effort": actuator._saturation_effort[:, j].clone(),
                "armature": actuator.armature[:, j].clone(),
                "friction": actuator.friction[:, j].clone(),
                "dynamic_friction": actuator.dynamic_friction[:, j].clone(),
                "viscous_friction": actuator.viscous_friction[:, j].clone(),
            }

        # Decoupled solver velocity cap: hold velocity_limit_sim at a fixed high value so the tuned
        # `velocity_limit` only shapes the DCMotor torque-speed curve, never a hard brick-wall cap.
        cap = float(cfg.solver_velocity_limit)
        actuator.velocity_limit_sim[:, j] = cap
        self._robot.write_joint_velocity_limit_to_sim(
            torch.full((self.num_envs, 1), cap, device=self.device), joint_ids=[j]
        )

        # persistent results from the most recently completed replay (not cleared by reset)
        self.last_scores: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # parameter batch API (used by the tuning scripts)
    # ------------------------------------------------------------------
    def set_param_batch(self, params: dict[str, np.ndarray | torch.Tensor], env_ids: torch.Tensor | None = None):
        """Set DCMotor parameter values per environment.

        Args:
            params: Mapping of parameter name (subset of :data:`dc_motor_params.ALL_PARAMS`) to a
                value per environment (shape ``(num_envs,)`` or ``(len(env_ids),)``).
            env_ids: Environments to update. Defaults to all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for name, values in params.items():
            if name not in self._param_values:
                raise KeyError(f"Unknown parameter '{name}'. Expected one of {tuple(self._param_values)}.")
            t = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1)
            self._param_values[name][env_ids] = t

    def _apply_param_values(self, env_ids: torch.Tensor):
        sub = {name: vals[env_ids] for name, vals in self._param_values.items()}
        mod = pdp if self._ideal_pd else dcp
        mod.apply_params(self._robot, self.cfg.actuator_name, self._joint_idx, env_ids, sub)

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device).reshape(-1)

        # replay restarts from the beginning for every (synchronous) reset
        self._step_idx = 0
        self._sim_pos[env_ids_t] = 0.0
        self._sim_vel[env_ids_t] = 0.0

        # initial joint state: active joint at first measured position, others at default
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids_t].clone()
        joint_pos[:, self._joint_idx] = self._ref_pos[0]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # hold all joints at their default target, then apply the tuned DCMotor params
        self._robot.set_joint_position_target(
            self._robot.data.default_joint_pos[env_ids_t], env_ids=env_ids_t
        )
        self._apply_param_values(env_ids_t)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        k = min(self._step_idx, self._num_steps - 1)
        # record the state at the start of this control step (result of the previous target)
        self._sim_pos[:, k] = self._robot.data.joint_pos[:, self._joint_idx]
        self._sim_vel[:, k] = self._robot.data.joint_vel[:, self._joint_idx]
        # drive the joint with the MLP-predicted target (ideal_pd) or the raw command (dc_motor)
        self._cur_target = self._joint_target[k]

    def _apply_action(self) -> None:
        target = self._cur_target.reshape(1, 1).expand(self.num_envs, 1)
        self._robot.set_joint_position_target(target, joint_ids=[self._joint_idx])

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        # per-step reward is unused; the objective is computed from the full trajectory at the end
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._step_idx += 1
        done = self._step_idx >= self._num_steps
        if done:
            self._finalize_scores()
        time_out = torch.full((self.num_envs,), bool(done), dtype=torch.bool, device=self.device)
        terminated = torch.zeros_like(time_out)
        # report as time-out (truncation), not failure termination
        return terminated, time_out

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _finalize_scores(self):
        self._last_sim_pos[:] = self._sim_pos
        sim_pos = self._sim_pos.detach().cpu().numpy()
        sim_vel = self._sim_vel.detach().cpu().numpy()
        self.last_scores = metrics.compute_scores(
            sim_pos,
            sim_vel,
            self.trajectory.ref_pos.astype(np.float32),
            self.trajectory.ref_vel.astype(np.float32),
            self._weights_np,
            self._reversal_np,
            score_mode=self.cfg.score_mode,
            spike_percentile=self.cfg.spike_percentile,
            segment_ids=self._segment_ids,
            step_dt=self.trajectory.step_dt,
            lag_weight=self.cfg.lag_weight,
            lag_max_s=self.cfg.lag_max_s,
        )

    def get_last_sim_pos(self) -> np.ndarray:
        """Return the simulated positions from the most recently completed replay. Shape ``(E, N)``."""
        return self._last_sim_pos.detach().cpu().numpy()

    def get_joint_target(self) -> np.ndarray:
        """Return the per-step position target the joint is driven with (MLP rollout or command)."""
        return self._joint_target_np

    @property
    def uses_mlp_target(self) -> bool:
        return bool(self.cfg.actuator_model == "ideal_pd" and self.cfg.mlp_checkpoint)

    @property
    def num_ctrl_steps(self) -> int:
        return self._num_steps
