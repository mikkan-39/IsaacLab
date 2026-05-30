# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pass 2 (optional): unlock joint friction, seeded from the pass-1 best parameters.

Only runs the friction search if the seed's reversal-segment position error exceeds a threshold
(otherwise the extra search dimensions are not worth the explosion in search space). Use
``--force`` to search regardless.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/actuator_tuning/run_friction_pass.py --headless \
        --joint-name "<exact_usd_joint_name>" \
        --seed-params logs/actuator_tuning/run01/best_refined.json \
        --search-yaml scripts/tools/actuator_tuning/friction_spec.example.yaml \
        --output-dir logs/actuator_tuning/run01_friction --reversal-mse-threshold 0.01
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Optional friction search seeded from pass-1 best params.")
parser.add_argument("--task", type=str, default="Isaac-Actuator-Tuning-RT-v0", help="Registered task id.")
parser.add_argument(
    "--joint-name", type=str, default="base_link_to_Neck_revolute", help="Exact USD joint name to drive."
)
parser.add_argument("--csv", type=str, default=None, help="Path to the recording CSV (defaults to packaged file).")
parser.add_argument("--seed-params", type=str, required=True, help="best_grid.json / best_refined.json from pass 1.")
parser.add_argument(
    "--joint-pos-limit",
    type=float,
    nargs=2,
    default=None,
    metavar=("LOW", "HIGH"),
    help="Override every joint's position limit with this range (rad). Default widens to +/- pi.",
)
parser.add_argument("--keep-usd-limits", action="store_true", help="Keep the USD joint limits (no widening).")
parser.add_argument("--search-yaml", type=str, required=True, help="Friction search spec YAML.")
parser.add_argument("--num-envs", type=int, default=64, help="Population batch size (parallel envs).")
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
    default=0.0,
    help="Advance the real reference by N control steps (delay compensation). Default 0 (none).",
)
parser.add_argument(
    "--lag-weight",
    type=float,
    default=2.0,
    help="Weight of the per-segment sim-vs-real lag term in the score (0 disables).",
)
parser.add_argument("--output-dir", type=str, default="logs/actuator_tuning/run_friction", help="Output directory.")
parser.add_argument("--top-k", type=int, default=10, help="How many top candidates to plot.")
parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
parser.add_argument(
    "--reversal-mse-threshold",
    type=float,
    default=0.0,
    help="Skip the friction search if the seed's reversal_pos_mse is below this (0 = always run).",
)
parser.add_argument("--force", action="store_true", help="Run the friction search regardless of the threshold.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import numpy as np

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  (registers the task)
from isaaclab_tasks.direct.actuator_tuning import search
from isaaclab_tasks.direct.actuator_tuning.actuator_tuning_env_cfg import ActuatorTuningEnvCfg


def build_cfg() -> ActuatorTuningEnvCfg:
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
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.keep_usd_limits:
        cfg.override_joint_pos_limits = None
    elif args_cli.joint_pos_limit is not None:
        cfg.override_joint_pos_limits = tuple(args_cli.joint_pos_limit)
    cfg.__post_init__()
    return cfg


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    with open(args_cli.seed_params, "r") as f:
        seed = json.load(f)
    seed_params = seed.get("params", seed)
    print(f"[friction] seeding fixed params from {args_cli.seed_params}: {seed_params}")

    cfg = build_cfg()
    env = gym.make(args_cli.task, cfg=cfg).unwrapped

    # seed the fixed (pass-1) params for all envs; this persists across resets
    env.set_param_batch({k: np.full(env.num_envs, float(v)) for k, v in seed_params.items()})

    # evaluate the seed (friction still 0) to decide whether pass 2 is warranted
    seed_metrics = search.evaluate_population(env, simulation_app, [], np.zeros((1, 0)))
    rev = float(seed_metrics["reversal_pos_mse"][0])
    print(f"[friction] seed reversal_pos_mse = {rev:.6g} (threshold {args_cli.reversal_mse_threshold:.6g})")
    if not args_cli.force and rev < args_cli.reversal_mse_threshold:
        print("[friction] reversal error below threshold; skipping friction search. Use --force to override.")
        env.close()
        return

    spec = search.load_search_spec(args_cli.search_yaml)
    param_names, samples = search.sample_params(spec, seed=args_cli.seed)
    print(f"[friction] {samples.shape[0]} samples over params: {param_names}")

    results = search.evaluate_population(env, simulation_app, param_names, samples)
    results_csv = os.path.join(args_cli.output_dir, "friction_results.csv")
    search.write_results_csv(results_csv, param_names, samples, results)
    print(f"[friction] wrote ranked results to: {results_csv}")

    order = np.argsort(results["score"])
    sorted_samples = samples[order]
    best_idx = int(order[0])
    best_metrics = {k: float(results[k][best_idx]) for k in search.METRIC_KEYS}
    # record the full parameter set (seed + friction) for the best candidate
    full_params = dict(seed_params)
    full_params.update({name: float(samples[best_idx, i]) for i, name in enumerate(param_names)})
    best_json = os.path.join(args_cli.output_dir, "best_friction.json")
    with open(best_json, "w") as f:
        json.dump(
            {"params": full_params, "metrics": best_metrics, "phase": "friction", "score_mode": cfg.score_mode},
            f,
            indent=2,
        )
    print(f"[friction] best score {best_metrics['score']:.6g} -> {best_json}")

    plots = search.plot_topk(
        env, simulation_app, param_names, sorted_samples, args_cli.output_dir, k=args_cli.top_k, title_suffix=f"friction {cfg.score_mode}"
    )
    print(f"[friction] saved {len(plots)} overlay plots to: {os.path.join(args_cli.output_dir, 'plots')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
