# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded servo trajectory once with a single DCMotor parameter set and plot the result.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/actuator_tuning/run_replay.py \
        --joint-name "<exact_usd_joint_name>" \
        --params '{"stiffness": 28.1, "damping": 1.7}'
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a servo recording and plot sim vs. real.")
parser.add_argument("--task", type=str, default="Isaac-Actuator-Tuning-RT-v0", help="Registered task id.")
parser.add_argument("--joint-name", type=str, required=True, help="Exact USD joint name to drive.")
parser.add_argument("--csv", type=str, default=None, help="Path to the recording CSV (defaults to packaged file).")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (1 for a clean GUI replay).")
parser.add_argument("--control-hz", type=float, default=50.0, help="Replay/control rate in Hz.")
parser.add_argument("--decimation", type=int, default=None, help="Physics steps per control step (optional).")
parser.add_argument("--max-duration-s", type=float, default=None, help="Cap replayed duration (seconds).")
parser.add_argument("--score-mode", type=str, default="position_only", help="position_only|position_heavy|balanced.")
parser.add_argument("--params", type=str, default=None, help="JSON dict of DCMotor params to apply.")
parser.add_argument("--out", type=str, default="logs/actuator_tuning/replay_overlay.png", help="Output plot path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  (registers the task)
from isaaclab_tasks.direct.actuator_tuning import plotting
from isaaclab_tasks.direct.actuator_tuning.actuator_tuning_env_cfg import ActuatorTuningEnvCfg


def build_cfg() -> ActuatorTuningEnvCfg:
    cfg = ActuatorTuningEnvCfg()
    cfg.joint_name = args_cli.joint_name
    if args_cli.csv is not None:
        cfg.csv_path = args_cli.csv
    cfg.control_hz = args_cli.control_hz
    if args_cli.decimation is not None:
        cfg.decimation = args_cli.decimation
    cfg.max_duration_s = args_cli.max_duration_s
    cfg.score_mode = args_cli.score_mode
    cfg.scene.num_envs = args_cli.num_envs
    # re-run post-init so sim.dt / robot reflect CLI overrides
    cfg.__post_init__()
    return cfg


def main():
    cfg = build_cfg()
    env = gym.make(args_cli.task, cfg=cfg).unwrapped

    if args_cli.params:
        params = json.loads(args_cli.params)
        batch = {k: np.full(env.num_envs, float(v)) for k, v in params.items()}
        env.set_param_batch(batch)

    env.reset()
    action = torch.zeros(env.num_envs, int(cfg.action_space), device=env.device)
    done = False
    while not done and simulation_app.is_running():
        with torch.inference_mode():
            _obs, _rew, _term, trunc, _info = env.step(action)
            done = bool(trunc[0].item())

    scores = {k: float(np.atleast_1d(v)[0]) for k, v in env.last_scores.items()}
    print("[replay] metrics for env 0:")
    for k, v in scores.items():
        print(f"    {k:18s}: {v:.6g}")

    traj = env.trajectory
    sim_pos = env.get_last_sim_pos()[0]
    out = plotting.save_overlay(
        args_cli.out,
        traj.t_rel,
        traj.target,
        traj.ref_pos,
        sim_pos,
        title=f"{args_cli.joint_name}  ({args_cli.score_mode})",
        metrics=scores,
    )
    print(f"[replay] saved overlay plot to: {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
