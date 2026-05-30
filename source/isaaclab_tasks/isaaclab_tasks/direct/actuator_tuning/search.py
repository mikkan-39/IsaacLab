# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampling, evaluation, and result I/O helpers for actuator parameter search.

Kept free of Isaac/torch imports at module load so it can be used both before and after the
simulation app is launched (torch is imported lazily inside :func:`evaluate_population`).
"""

from __future__ import annotations

import csv
import json
import numpy as np

METRIC_KEYS = (
    "score",
    "tracking",
    "weighted_pos_mse",
    "weighted_vel_mse",
    "spike_pos_err",
    "max_abs_pos",
    "reversal_pos_mse",
)


def load_search_spec(path: str) -> dict:
    """Load a YAML search spec (n_samples, method, params: {name: {low, high, log}})."""
    import yaml

    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    if "params" not in spec or not spec["params"]:
        raise ValueError(f"Search spec '{path}' has no 'params'.")
    return spec


def sample_params(spec: dict, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Draw samples for the parameters described by ``spec``.

    Returns ``(param_names, samples)`` where ``samples`` has shape ``(n_samples, len(param_names))``.
    Supports ``method: latin_hypercube`` (default) or ``random``; per-param ``log: true`` samples
    in log-space.
    """
    params: dict = spec["params"]
    names = list(params.keys())
    d = len(names)
    n = int(spec.get("n_samples", 200))
    method = str(spec.get("method", "latin_hypercube")).lower()

    if method in ("latin_hypercube", "lhs"):
        try:
            from scipy.stats import qmc

            unit = qmc.LatinHypercube(d=d, seed=seed).random(n)
        except Exception:
            rng = np.random.default_rng(seed)
            unit = rng.random((n, d))
    elif method == "random":
        rng = np.random.default_rng(seed)
        unit = rng.random((n, d))
    else:
        raise ValueError(f"Unknown sampling method '{method}'.")

    samples = np.zeros((n, d), dtype=np.float64)
    for i, name in enumerate(names):
        p = params[name]
        low, high = float(p["low"]), float(p["high"])
        if p.get("log", False):
            if low <= 0:
                raise ValueError(f"Parameter '{name}' uses log sampling but low={low} <= 0.")
            samples[:, i] = np.exp(np.log(low) + unit[:, i] * (np.log(high) - np.log(low)))
        else:
            samples[:, i] = low + unit[:, i] * (high - low)
    _enforce_friction_constraint(names, samples)
    return names, samples


def _enforce_friction_constraint(names: list[str], samples: np.ndarray) -> None:
    """Clamp ``dynamic_friction`` <= ``friction`` in-place.

    PhysX requires static friction >= dynamic friction; the sampler draws them independently, so we
    project invalid pairs onto the boundary here (matching the guard applied in the env) so the
    recorded samples are exactly what gets simulated.
    """
    if "friction" in names and "dynamic_friction" in names:
        fi, di = names.index("friction"), names.index("dynamic_friction")
        np.minimum(samples[:, di], samples[:, fi], out=samples[:, di])


def bounds_from_spec(spec: dict, names: list[str]) -> list[tuple[float, float]]:
    """Return (low, high) bounds in natural (non-log) space for the given parameter names."""
    params = spec["params"]
    return [(float(params[n]["low"]), float(params[n]["high"])) for n in names]


def evaluate_population(env, simulation_app, param_names: list[str], samples: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate a population of parameter vectors in batches of ``env.num_envs``.

    Parameters not in ``param_names`` keep whatever value is currently stored in the env (defaults
    or a previously seeded value), which persists across resets.

    Returns a dict of metric name -> array of shape ``(n_samples,)``.
    """
    import torch

    n = samples.shape[0]
    e = env.num_envs
    out = {k: np.full(n, np.nan) for k in METRIC_KEYS}
    action_dim = int(getattr(env.cfg, "action_space", 1))
    zero_action = torch.zeros(e, action_dim, device=env.device)

    for start in range(0, n, e):
        batch = samples[start : start + e]
        b = batch.shape[0]
        # pad short final batch by repeating the last row; extra envs are ignored
        batch_full = batch if b == e else np.concatenate([batch, np.repeat(batch[-1:], e - b, axis=0)], axis=0)
        params = {name: batch_full[:, i] for i, name in enumerate(param_names)}

        env.set_param_batch(params)
        env.reset()
        done = False
        while not done and simulation_app.is_running():
            _obs, _rew, _term, trunc, _info = env.step(zero_action)
            done = bool(trunc[0].item())

        scores = env.last_scores
        for k in METRIC_KEYS:
            out[k][start : start + b] = np.atleast_1d(scores[k])[:b]
        print(f"[search] evaluated {min(start + b, n)}/{n}  best score so far: {np.nanmin(out['score']):.5g}")
    return out


def write_results_csv(path: str, param_names: list[str], samples: np.ndarray, results: dict[str, np.ndarray]):
    """Write a results CSV sorted by ascending score (best first)."""
    order = np.argsort(results["score"])
    header = list(param_names) + list(METRIC_KEYS)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx in order:
            row = [f"{samples[idx, i]:.10g}" for i in range(len(param_names))]
            row += [f"{results[k][idx]:.10g}" for k in METRIC_KEYS]
            w.writerow(row)


def read_results_csv(path: str) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Read a results CSV written by :func:`write_results_csv`."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader if row]
    arr = np.asarray(rows, dtype=np.float64)
    param_names = [h for h in header if h not in METRIC_KEYS]
    n_params = len(param_names)
    samples = arr[:, :n_params]
    results = {k: arr[:, header.index(k)] for k in METRIC_KEYS}
    return param_names, samples, results


def plot_topk(
    env,
    simulation_app,
    param_names: list[str],
    sorted_samples: np.ndarray,
    out_dir: str,
    k: int = 10,
    title_suffix: str = "",
) -> list[str]:
    """Re-evaluate the best ``k`` parameter vectors in one batch and save overlay plots.

    Args:
        env: The tuning environment (unwrapped).
        simulation_app: The running simulation app.
        param_names: Names of the searched parameters (columns of ``sorted_samples``).
        sorted_samples: Samples sorted best-first. Shape ``(>=k, len(param_names))``.
        out_dir: Output directory; plots go to ``out_dir/plots``.
        k: Number of top candidates to plot (capped by population and ``env.num_envs``).
        title_suffix: Extra text appended to each plot title.

    Returns:
        List of written plot paths.
    """
    import os

    from . import plotting

    k = int(min(k, sorted_samples.shape[0], env.num_envs))
    top = sorted_samples[:k]
    res = evaluate_population(env, simulation_app, param_names, top)
    sim_pos = env.get_last_sim_pos()
    traj = env.trajectory

    paths: list[str] = []
    for i in range(k):
        metrics_row = {key: float(res[key][i]) for key in METRIC_KEYS}
        path = os.path.join(out_dir, "plots", f"rank_{i:02d}_overlay.png")
        plotting.save_overlay(
            path,
            traj.t_rel,
            traj.target,
            traj.ref_pos,
            sim_pos[i],
            title=f"rank {i}  {title_suffix}".strip(),
            metrics=metrics_row,
        )
        paths.append(path)

    # dump trajectories for offline interactive (plotly) plotting
    metrics_arr = np.stack([res[key][:k] for key in METRIC_KEYS], axis=1)
    plotting.save_trajectories_npz(
        os.path.join(out_dir, "trajectories.npz"),
        traj.t_rel,
        traj.target,
        traj.ref_pos,
        sim_pos[:k],
        param_names=param_names,
        params=top,
        metric_names=list(METRIC_KEYS),
        metrics=metrics_arr,
    )
    return paths


def write_best_json(path: str, param_names: list[str], sample: np.ndarray, metrics_row: dict[str, float], extra: dict | None = None):
    """Write a single best-parameter record to JSON."""
    record = {"params": {name: float(sample[i]) for i, name in enumerate(param_names)}}
    record["metrics"] = {k: float(v) for k, v in metrics_row.items()}
    if extra:
        record.update(extra)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
