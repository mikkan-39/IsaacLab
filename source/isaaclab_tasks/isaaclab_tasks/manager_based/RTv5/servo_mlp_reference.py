from __future__ import annotations

"""Stateful, batched runtime for the learned servo MLP (online reference generator).

The MLP is trained offline by ``scripts/tools/actuator_tuning/train_servo_mlp.py`` as an
actuator-net-style one-step predictor: from a window of the servo's recent tracking **error** and
**velocity** it predicts the next position *increment*. Integrating the increment yields a
(delay-free) position trajectory.

At training/tuning time the rollout was precomputed over a recording. At RL runtime we run the same
rollout **incrementally and vectorized** over ``(num_envs, num_joints)``: each control step takes the
commanded joint target, advances every joint's MLP one step, and returns the predicted position used
as the IdealPD position setpoint. The actuator's PD lag re-introduces the (delay-free) servo lag, so
no extra command delay is needed.

The feature layout (error history then velocity history, most-recent-first) **must match** the
training script's ``build_training_pairs`` / ``rollout_segment`` exactly; it is duplicated here
(rather than imported) because the trainer is intentionally Isaac-Lab-free.
"""

import numpy as np
import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden, out_dim: int = 1) -> nn.Sequential:
    layers: list = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ServoMlpReference:
    """Online, per-(env, joint) servo-position predictor driven by the commanded target.

    Args:
        checkpoint_path: Path to a ``servo_mlp.pt`` checkpoint from ``train_servo_mlp.py``.
        num_envs: Number of parallel environments.
        num_joints: Number of joints the reference is applied to (the action term's joints).
        device: Torch device.
        control_dt: Control time-step (s) used for velocity derivation. If None, uses the
            checkpoint's training rate (``1 / control_hz``); these should be equal.
    """

    def __init__(self, checkpoint_path: str, num_envs: int, num_joints: int, device, control_dt: float | None = None):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.history: int = int(ckpt["history"])
        self.model = _build_mlp(2 * self.history, list(ckpt["hidden"])).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.velocity_source = str(ckpt.get("velocity_source", "derived"))
        train_dt = 1.0 / float(ckpt.get("control_hz", 50.0))
        self.dt = float(control_dt) if control_dt is not None else train_dt

        # normalization stats as broadcastable tensors over (E, J, 2N) and (E, J)
        self._x_mean = torch.as_tensor(np.asarray(ckpt["x_mean"]), dtype=torch.float32, device=device).view(1, 1, -1)
        self._x_std = torch.as_tensor(np.asarray(ckpt["x_std"]), dtype=torch.float32, device=device).view(1, 1, -1)
        self._y_mean = float(ckpt["y_mean"])
        self._y_std = float(ckpt["y_std"])

        # per-(env, joint) running state
        self._pos = torch.zeros(num_envs, num_joints, device=device)
        self._prev_pos = torch.zeros(num_envs, num_joints, device=device)
        # history buffers, index 0 = most recent (matches training's most-recent-first features)
        self._err_hist = torch.zeros(num_envs, num_joints, self.history, device=device)
        self._vel_hist = torch.zeros(num_envs, num_joints, self.history, device=device)

    @torch.no_grad()
    def reset(self, env_ids, default_pos: torch.Tensor) -> None:
        """Re-seed state for ``env_ids`` to ``default_pos`` (shape ``(len(env_ids), num_joints)``)."""
        self._pos[env_ids] = default_pos
        self._prev_pos[env_ids] = default_pos
        self._err_hist[env_ids] = 0.0
        self._vel_hist[env_ids] = 0.0

    @torch.no_grad()
    def step(self, cmd: torch.Tensor) -> torch.Tensor:
        """Advance every joint's MLP one control step and return the position setpoint.

        Args:
            cmd: Commanded joint target this step. Shape ``(num_envs, num_joints)``.

        Returns:
            Predicted servo position to use as the IdealPD setpoint. Shape ``(num_envs, num_joints)``.
        """
        # setpoint for THIS step is the current prediction (computed last step), matching the offline
        # rollout where the target applied at step k is pos[k].
        setpoint = self._pos.clone()

        err = cmd - self._pos
        vel = (self._pos - self._prev_pos) / self.dt

        # push most-recent-first
        self._err_hist = torch.roll(self._err_hist, shifts=1, dims=-1)
        self._vel_hist = torch.roll(self._vel_hist, shifts=1, dims=-1)
        self._err_hist[..., 0] = err
        self._vel_hist[..., 0] = vel

        feats = torch.cat([self._err_hist, self._vel_hist], dim=-1)  # (E, J, 2N)
        xn = (feats - self._x_mean) / self._x_std
        dn = self.model(xn.reshape(-1, 2 * self.history)).reshape(self.num_envs, self.num_joints)
        dpos = dn * self._y_std + self._y_mean

        self._prev_pos = self._pos
        self._pos = self._pos + dpos
        return setpoint
