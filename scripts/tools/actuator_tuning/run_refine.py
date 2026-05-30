# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Refine the top-k sample-search candidates with derivative-free local optimization (scipy).

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/actuator_tuning/run_refine.py --headless \
        --joint-name "<exact_usd_joint_name>" \
        --results logs/actuator_tuning/run01/sample_results.csv \
        --search-yaml scripts/tools/actuator_tuning/search_spec.example.yaml \
        --output-dir logs/actuator_tuning/run01 --top-k 5
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Refine top-k actuator candidates with scipy.optimize.")
parser.add_argument("--task", type=str, default="Isaac-Actuator-Tuning-RT-v0", help="Registered task id.")
parser.add_argument(
    "--joint-name", type=str, default="base_link_to_Neck_revolute", help="Exact USD joint name to drive."
)
parser.add_argument("--csv", type=str, default=None, help="Path to the recording CSV (defaults to packaged file).")
parser.add_argument("--results", type=str, required=True, help="sample_results.csv from the search pass.")
parser.add_argument(
    "--joint-pos-limit",
    type=float,
    nargs=2,
    default=None,
    metavar=("LOW", "HIGH"),
    help="Override every joint's position limit with this range (rad). Default widens to +/- pi.",
)
parser.add_argument("--keep-usd-limits", action="store_true", help="Keep the USD joint limits (no widening).")
parser.add_argument("--search-yaml", type=str, default=None, help="Search spec YAML (for bounds). Optional.")
parser.add_argument("--control-hz", type=float, default=50.0, help="Replay/control rate in Hz.")
parser.add_argument(
    "--solver-velocity-limit",
    type=float,
    default=20.0,
    help="Fixed PhysX solver speed cap (rad/s), decoupled from the tuned velocity_limit.",
)
parser.add_argument("--decimation", type=int, default=None, help="Physics steps per control step (optional).")
parser.add_argument("--max-duration-s", type=float, default=None, help="Cap replayed duration (seconds).")
parser.add_argument("--score-mode", type=str, default="position_only", help="position_only|position_heavy|balanced.")
parser.add_argument(
    "--spike-percentile", type=float, default=99.0, help="Percentile of |pos error| used as the spike term (default 99)."
)
parser.add_argument(
    "--ref-lag-steps",
    type=float,
    default=1.0,
    help="Advance the real reference by N control steps to compensate the servo transport lag.",
)
parser.add_argument(
    "--lag-weight",
    type=float,
    default=2.0,
    help="Weight of the per-segment sim-vs-real lag term in the score (0 disables).",
)
parser.add_argument("--output-dir", type=str, default="logs/actuator_tuning/run", help="Output directory.")
parser.add_argument("--top-k", type=int, default=5, help="How many seeds to refine (also env batch size).")
parser.add_argument("--max-iter", type=int, default=60, help="Max iterations per scipy.optimize run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np

import gymnasium as gym
from scipy.optimize import minimize

import isaaclab_tasks  # noqa: F401  (registers the task)
from isaaclab_tasks.direct.actuator_tuning import search
from isaaclab_tasks.direct.actuator_tuning.actuator_tuning_env_cfg import ActuatorTuningEnvCfg


def build_cfg(num_envs: int) -> ActuatorTuningEnvCfg:
    cfg = ActuatorTuningEnvCfg()
    cfg.joint_name = args_cli.joint_name
    if args_cli.csv is not None:
        cfg.csv_path = args_cli.csv
    cfg.control_hz = args_cli.control_hz
    cfg.solver_velocity_limit = args_cli.solver_velocity_limit
    if args_cli.decimation is not None:
        cfg.decimation = args_cli.decimation
    cfg.max_duration_s = args_cli.max_duration_s
    cfg.score_mode = args_cli.score_mode
    cfg.spike_percentile = args_cli.spike_percentile
    cfg.ref_lag_steps = args_cli.ref_lag_steps
    cfg.lag_weight = args_cli.lag_weight
    cfg.scene.num_envs = num_envs
    if args_cli.keep_usd_limits:
        cfg.override_joint_pos_limits = None
    elif args_cli.joint_pos_limit is not None:
        cfg.override_joint_pos_limits = tuple(args_cli.joint_pos_limit)
    cfg.__post_init__()
    return cfg


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    param_names, samples, results = search.read_results_csv(args_cli.results)
    k = int(min(args_cli.top_k, samples.shape[0]))
    seeds = samples[:k]  # results CSV is already sorted best-first

    # bounds: from the search YAML if given, else from the sampled data range
    if args_cli.search_yaml is not None:
        spec = search.load_search_spec(args_cli.search_yaml)
        bounds = search.bounds_from_spec(spec, param_names)
    else:
        bounds = [(float(samples[:, i].min()), float(samples[:, i].max())) for i in range(len(param_names))]
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    cfg = build_cfg(num_envs=max(k, 1))
    env = gym.make(args_cli.task, cfg=cfg).unwrapped

    def objective(x: np.ndarray) -> float:
        xc = np.clip(x, lows, highs)
        res = search.evaluate_population(env, simulation_app, param_names, xc[None, :])
        score = float(res["score"][0])
        # soft penalty for proposing points outside the bounds
        penalty = float(np.sum((x - xc) ** 2)) * 1.0e3
        return score + penalty

    refined = []
    for i, seed in enumerate(seeds):
        print(f"[refine] seed {i} start score {results['score'][i]:.6g}  params {seed}")
        res = minimize(
            objective,
            x0=seed,
            method="Nelder-Mead",
            options={"maxiter": args_cli.max_iter, "xatol": 1e-3, "fatol": 1e-5},
        )
        x_best = np.clip(res.x, lows, highs)
        refined.append((float(res.fun), x_best))
        print(f"[refine] seed {i} refined score {res.fun:.6g}")

    refined.sort(key=lambda t: t[0])
    refined_samples = np.array([x for _, x in refined])

    # final metrics for the refined population (single batch) + best json
    final = search.evaluate_population(env, simulation_app, param_names, refined_samples)
    best_metrics = {key: float(final[key][0]) for key in search.METRIC_KEYS}
    best_json = os.path.join(args_cli.output_dir, "best_refined.json")
    search.write_best_json(
        best_json, param_names, refined_samples[0], best_metrics, extra={"phase": "refine", "score_mode": cfg.score_mode}
    )
    print(f"[refine] best refined score {best_metrics['score']:.6g} -> {best_json}")

    plots = search.plot_topk(
        env, simulation_app, param_names, refined_samples, args_cli.output_dir, k=k, title_suffix=f"refined {cfg.score_mode}"
    )
    print(f"[refine] saved {len(plots)} overlay plots to: {os.path.join(args_cli.output_dir, 'plots')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
