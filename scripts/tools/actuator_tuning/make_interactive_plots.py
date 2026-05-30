# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build an interactive (zoom/pan) Plotly HTML from a tuning run's ``trajectories.npz``.

This is standalone: it only needs ``numpy`` and ``plotly`` (NOT Isaac Sim). Run it with your
normal Python (e.g. the one that has plotly installed), not ``isaaclab.bat``:

.. code-block:: bash

    python scripts/tools/actuator_tuning/make_interactive_plots.py --run-dir logs/actuator_tuning/run01

Open the resulting ``interactive.html`` in a browser; use the rank dropdown to switch candidates
and the range slider / box-zoom to inspect the fast-motion bursts.
"""

from __future__ import annotations

import argparse
import os
import webbrowser

import numpy as np


def _load(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out


def build_html(npz_path: str, out_html: str | None = None, open_browser: bool = False) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "plotly is required for interactive plots. Install it with `pip install plotly` "
            "in the Python you are running this script with."
        ) from e

    d = _load(npz_path)
    t = d["t_rel"]
    target = d["target"]
    ref = d["ref_pos"]
    sim = np.atleast_2d(d["sim_pos"])
    k = sim.shape[0]

    param_names = list(d["param_names"]) if "param_names" in d else []
    params = d["params"] if "params" in d else None
    metric_names = list(d["metric_names"]) if "metric_names" in d else []
    metrics = d["metrics"] if "metrics" in d else None

    def label(i: int) -> str:
        bits = [f"rank {i}"]
        if metrics is not None and "score" in metric_names:
            bits.append(f"score={metrics[i, metric_names.index('score')]:.4g}")
        return "  ".join(bits)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("position (rad)", "position error: sim - real (rad)"),
    )

    # always-on reference traces
    fig.add_trace(go.Scattergl(x=t, y=target, name="target (cmd)", line=dict(color="gray", dash="dash", width=1)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=t, y=ref, name="real", line=dict(color="royalblue", width=1.5)), row=1, col=1)

    # per-rank sim + error traces (rank 0 visible by default)
    sim_trace_idx = []
    err_trace_idx = []
    for i in range(k):
        vis = i == 0
        fig.add_trace(
            go.Scattergl(x=t, y=sim[i], name=f"sim {label(i)}", visible=vis, line=dict(color="darkorange", width=1.5)),
            row=1,
            col=1,
        )
        sim_trace_idx.append(len(fig.data) - 1)
        fig.add_trace(
            go.Scattergl(x=t, y=sim[i] - ref, name=f"err {label(i)}", visible=vis, line=dict(color="crimson", width=1), showlegend=False),
            row=2,
            col=1,
        )
        err_trace_idx.append(len(fig.data) - 1)

    n_traces = len(fig.data)

    def visibility(selected: int | None) -> list[bool]:
        vis = [False] * n_traces
        vis[0] = True  # target
        vis[1] = True  # real
        for i in range(k):
            on = (selected is None) or (i == selected)
            vis[sim_trace_idx[i]] = on
            vis[err_trace_idx[i]] = on
        return vis

    buttons = []
    for i in range(k):
        title = label(i)
        if params is not None and len(param_names):
            title += "<br>" + ", ".join(f"{n}={params[i, j]:.4g}" for j, n in enumerate(param_names))
        buttons.append(dict(label=f"rank {i}", method="update", args=[{"visible": visibility(i)}, {"title.text": title}]))
    buttons.append(dict(label="all", method="update", args=[{"visible": visibility(None)}, {"title.text": "all ranks"}]))

    init_title = label(0)
    if params is not None and len(param_names):
        init_title += "<br>" + ", ".join(f"{n}={params[0, j]:.4g}" for j, n in enumerate(param_names))

    fig.update_layout(
        title=dict(text=init_title, x=0.0, font=dict(size=13)),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
        margin=dict(l=60, r=20, t=90, b=40),
        updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.18, yanchor="top", showactive=True)],
        template="plotly_white",
    )
    # range slider on the bottom shared axis for fast scrubbing into bursts
    fig.update_xaxes(rangeslider=dict(visible=True), row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="black", row=2, col=1)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(os.path.abspath(npz_path)), "interactive.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[interactive] wrote {out_html}  ({k} ranks, {t.shape[0]} samples)")
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out_html))
    return out_html


def main():
    parser = argparse.ArgumentParser(description="Build interactive Plotly HTML from trajectories.npz.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", type=str, default=None, help="Path to a trajectories.npz file.")
    g.add_argument("--run-dir", type=str, default=None, help="Run directory containing trajectories.npz.")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (defaults next to the npz).")
    parser.add_argument("--open", action="store_true", help="Open the HTML in a browser when done.")
    args = parser.parse_args()

    npz = args.npz if args.npz is not None else os.path.join(args.run_dir, "trajectories.npz")
    if not os.path.isfile(npz):
        raise SystemExit(f"trajectories.npz not found at: {npz}")
    build_html(npz, args.out, open_browser=args.open)


if __name__ == "__main__":
    main()
