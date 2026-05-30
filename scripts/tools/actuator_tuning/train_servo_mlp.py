# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a standalone MLP servo model on recorded servo data (no Isaac Lab required).

The MLP is an actuator-net-style one-step predictor: from a window of the servo's recent
**tracking error** and **velocity** it predicts the next position *increment*. Integrating the
increment yields the position trajectory. Predicting the increment (rather than absolute position)
is well-posed -- absolute position is not recoverable from ``error`` + ``velocity`` alone.

Pipeline
--------
1. Load the CSV, resample to ``--control-hz`` (default 50) with a zero-order hold.
2. Advance the real position/velocity by ``--ref-lag-steps`` to remove the servo's transport delay
   (the MLP learns the delay-free dynamics; the delay is re-added downstream, e.g. by a PD follower).
3. Build per-step features ``[e[k..k-N+1], v[k..k-N+1]]`` -> label ``pos[k+1]-pos[k]``.
4. Split train/val by whole 10 s segments (validation = motions the MLP never saw).
5. Train (one-step, teacher-forced), then evaluate with a **closed-loop autoregressive rollout**
   per segment (warm-start with N real samples, then free-run on its own predictions).
6. Plot target / real / MLP position (static PNG + interactive Plotly HTML if available).

Example
-------
.. code-block:: bash

    python scripts/tools/actuator_tuning/train_servo_mlp.py \
        --csv source/isaaclab_tasks/isaaclab_tasks/direct/actuator_tuning/data/servo_recording.csv \
        --history 16 --epochs 300 --out-dir logs/actuator_tuning/mlp01
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# torch is imported lazily inside the training path so the `--plot-npz` render mode can run in a
# plotly-only environment that has no torch (the Isaac bundle has torch but not plotly, and vice versa).

try:
    import plotly.graph_objects as go  # noqa: E402

    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


# ----------------------------------------------------------------------------------------------
# data loading (self-contained; mirrors csv_replay but without any Isaac dependency)
# ----------------------------------------------------------------------------------------------
def load_servo(csv_path: str, control_hz: float, ref_lag_steps: float, max_duration_s: float | None):
    """Load the recording, zero-order-hold downsample to control_hz, and advance ref by the delay.

    Returns ``(t, target, pos, vel_recorded, step_dt, num_steps)`` (all 1-D, length num_steps).
    """
    cols = {"timestamp": [], "target_rad": [], "position_rad": [], "speed_rad_s": []}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        need = list(cols.keys())
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV '{csv_path}' missing columns {missing}. Found {reader.fieldnames}.")
        for row in reader:
            if not row.get("timestamp"):
                continue
            for c in cols:
                cols[c].append(float(row[c]))
    ts = np.asarray(cols["timestamp"], dtype=np.float64)
    if ts.size < 2:
        raise ValueError(f"CSV '{csv_path}' has too few rows ({ts.size}).")

    t_rel_raw = ts - ts[0]
    total = float(t_rel_raw[-1])
    if max_duration_s is not None:
        total = min(total, float(max_duration_s))
    step_dt = 1.0 / float(control_hz)
    num_steps = int(np.floor(total / step_dt)) + 1
    t_ctrl = np.arange(num_steps, dtype=np.float64) * step_dt

    idx = np.clip(np.searchsorted(t_rel_raw, t_ctrl, side="right") - 1, 0, t_rel_raw.size - 1)
    target = np.asarray(cols["target_rad"], dtype=np.float64)[idx]
    pos = np.asarray(cols["position_rad"], dtype=np.float64)[idx]
    vel = np.asarray(cols["speed_rad_s"], dtype=np.float64)[idx]

    if ref_lag_steps:
        src = np.clip(np.arange(num_steps, dtype=np.float64) + float(ref_lag_steps), 0.0, num_steps - 1)
        lo = np.floor(src).astype(int)
        hi = np.clip(lo + 1, 0, num_steps - 1)
        frac = src - lo
        pos = pos[lo] * (1.0 - frac) + pos[hi] * frac
        vel = vel[lo] * (1.0 - frac) + vel[hi] * frac

    return t_ctrl, target, pos, vel, step_dt, num_steps


def derived_velocity(pos: np.ndarray, step_dt: float) -> np.ndarray:
    """Causal backward-difference velocity (v[0] = 0)."""
    v = np.zeros_like(pos)
    v[1:] = (pos[1:] - pos[:-1]) / step_dt
    return v


# ----------------------------------------------------------------------------------------------
# feature construction
# ----------------------------------------------------------------------------------------------
def build_training_pairs(target, pos, vel, seg_ids, history: int):
    """Build one-step (feature, label) pairs that stay within a single segment.

    Feature at step k: ``[e[k], e[k-1], ..., e[k-N+1], v[k], ..., v[k-N+1]]`` (2N values).
    Label: ``pos[k+1] - pos[k]``.
    Only k with a full in-segment window and an in-segment k+1 are used.
    """
    e = target - pos
    feats, labels = [], []
    n = pos.shape[0]
    for k in range(history - 1, n - 1):
        if seg_ids[k - history + 1] != seg_ids[k] or seg_ids[k + 1] != seg_ids[k]:
            continue  # window or label crosses a segment boundary
        win = slice(k - history + 1, k + 1)
        feats.append(np.concatenate([e[win][::-1], vel[win][::-1]]))  # most-recent-first
        labels.append(pos[k + 1] - pos[k])
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.float64).reshape(-1, 1)


def build_mlp(in_dim: int, hidden: list[int], out_dim: int = 1):
    import torch.nn as nn

    layers: list = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------------------------
# closed-loop rollout
# ----------------------------------------------------------------------------------------------
def rollout_segment(model, target, pos_real, vel_real, step_dt, history, x_mean, x_std, y_mean, y_std, device,
                    velocity_source):
    """Closed-loop autoregressive rollout over one segment.

    Warm-starts the first ``history`` steps with the real position/velocity, then free-runs on its
    own predictions. Returns the predicted position trajectory for the segment.
    """
    import torch

    n = pos_real.shape[0]
    pos_pred = pos_real.copy()
    vel_pred = vel_real.copy()
    for k in range(history - 1, n - 1):
        win = slice(k - history + 1, k + 1)
        e = (target[win] - pos_pred[win])[::-1]
        v = vel_pred[win][::-1]
        feat = np.concatenate([e, v])
        xn = (feat - x_mean) / x_std
        with torch.no_grad():
            dn = model(torch.as_tensor(xn, dtype=torch.float32, device=device).unsqueeze(0)).item()
        dpos = dn * y_std + y_mean
        pos_pred[k + 1] = pos_pred[k] + dpos
        if velocity_source == "recorded":
            # no recorded velocity for a predicted step; fall back to derived for the free-run part
            vel_pred[k + 1] = (pos_pred[k + 1] - pos_pred[k]) / step_dt
        else:
            vel_pred[k + 1] = (pos_pred[k + 1] - pos_pred[k]) / step_dt
    return pos_pred


def main():
    p = argparse.ArgumentParser(description="Train a standalone MLP servo model on recorded data.")
    p.add_argument("--csv", type=str, default=None, help="Path to the recording CSV (required for training).")
    p.add_argument("--control-hz", type=float, default=50.0, help="Resample/training rate (Hz).")
    p.add_argument("--ref-lag-steps", type=float, default=1.0,
                   help="Advance real pos/vel by N control steps to remove the servo transport delay.")
    p.add_argument("--max-duration-s", type=float, default=None, help="Cap recording duration (s).")
    p.add_argument("--segment-len-s", type=float, default=10.0, help="Length of each excitation segment (s).")
    p.add_argument("--history", type=int, default=16, help="N: window length for error & velocity history.")
    p.add_argument("--velocity-source", choices=["derived", "recorded"], default="derived",
                   help="'derived' = backward-difference of position (consistent in rollout); "
                        "'recorded' = the CSV speed_rad_s column (rollout still derives).")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128], help="Hidden layer sizes.")
    p.add_argument("--epochs", type=int, default=300, help="Max training epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=512, help="Minibatch size.")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of segments held out for validation.")
    p.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default: cuda if available).")
    p.add_argument("--out-dir", type=str, default="logs/actuator_tuning/mlp01", help="Output directory.")
    p.add_argument("--plot-npz", type=str, default=None,
                   help="Render-only mode: load a rollout.npz (from a prior run) and write the interactive "
                        "HTML next to it. Needs plotly but not torch.")
    args = p.parse_args()

    # render-only mode: build the interactive plot from a saved rollout (torch not required).
    if args.plot_npz:
        if not _HAVE_PLOTLY:
            raise RuntimeError("--plot-npz needs plotly; run this in an env where 'import plotly' works.")
        d = np.load(args.plot_npz)
        out_html = os.path.splitext(args.plot_npz)[0] + ".html"
        _save_interactive_plot(out_html, d["t"], d["target"], d["real"], d["pred"], set(d["val_segs"].tolist()),
                               set(d["val_segs"].tolist()), int(d["seg_len_steps"]), float(d["step_dt"]))
        print(f"[mlp] wrote interactive plot to {out_html}")
        return

    if not args.csv:
        p.error("--csv is required for training (omit it only with --plot-npz).")

    import torch  # noqa: F401  (needed for training; imported lazily so plot-only mode is torch-free)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- data ---
    t, target, pos, vel_rec, step_dt, n = load_servo(args.csv, args.control_hz, args.ref_lag_steps, args.max_duration_s)
    vel = vel_rec if args.velocity_source == "recorded" else derived_velocity(pos, step_dt)
    seg_len_steps = max(1, int(round(args.segment_len_s / step_dt)))
    seg_ids = (np.arange(n) // seg_len_steps).astype(np.int64)
    segments = np.unique(seg_ids)
    print(f"[mlp] {n} steps @ {args.control_hz:g} Hz, {len(segments)} segments of ~{seg_len_steps} steps")

    # --- train/val segment split (evenly spaced val segments cover diverse motions) ---
    stride = max(2, int(round(1.0 / max(args.val_frac, 1e-6))))
    val_segs = set(segments[::stride].tolist())
    train_segs = [s for s in segments if s not in val_segs]
    print(f"[mlp] train segments: {len(train_segs)}, val segments: {len(val_segs)}")

    train_mask = np.isin(seg_ids, list(train_segs))
    val_mask = np.isin(seg_ids, list(val_segs))

    # build pairs (restricting windows to train- vs val-only segments via masked seg_ids copies)
    seg_train = np.where(train_mask, seg_ids, -1)
    seg_val = np.where(val_mask, seg_ids, -2)
    Xtr, Ytr = build_training_pairs(target, pos, vel, seg_train, args.history)
    Xva, Yva = build_training_pairs(target, pos, vel, seg_val, args.history)
    if Xtr.size == 0:
        raise RuntimeError("No training pairs built; reduce --history or check segmentation.")
    print(f"[mlp] training pairs: {Xtr.shape[0]}, val pairs: {Xva.shape[0]}")

    # --- normalization (from training set only) ---
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    y_mean, y_std = float(Ytr.mean()), float(Ytr.std()) + 1e-8

    def to_norm(X, Y):
        xn = torch.as_tensor((X - x_mean) / x_std, dtype=torch.float32, device=device)
        yn = torch.as_tensor((Y - y_mean) / y_std, dtype=torch.float32, device=device)
        return xn, yn

    Xtr_n, Ytr_n = to_norm(Xtr, Ytr)
    Xva_n, Yva_n = (to_norm(Xva, Yva) if Xva.size else (None, None))

    # --- model ---
    import torch.nn as nn

    model = build_mlp(in_dim=2 * args.history, hidden=list(args.hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    ntr = Xtr_n.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(ntr, device=device)
        tot = 0.0
        for i in range(0, ntr, args.batch_size):
            bi = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_n[bi]), Ytr_n[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * bi.shape[0]
        tr_loss = tot / ntr

        if Xva_n is not None:
            model.eval()
            with torch.no_grad():
                va_loss = loss_fn(model(Xva_n), Yva_n).item()
        else:
            va_loss = tr_loss

        improved = va_loss < best_val - 1e-9
        if improved:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or improved:
            print(f"[mlp] epoch {epoch:4d}  train {tr_loss:.3e}  val {va_loss:.3e}"
                  + ("  *" if improved else ""))
        if bad >= args.patience:
            print(f"[mlp] early stop at epoch {epoch} (no val improvement for {args.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- closed-loop rollout over every segment ---
    model.eval()
    pos_pred = pos.copy()
    per_seg = []
    for s in segments:
        sl = np.flatnonzero(seg_ids == s)
        a, b = sl[0], sl[-1] + 1
        if b - a <= args.history:
            continue
        pp = rollout_segment(model, target[a:b], pos[a:b], vel[a:b], step_dt, args.history,
                             x_mean, x_std, y_mean, y_std, device, args.velocity_source)
        pos_pred[a:b] = pp
        rmse = float(np.sqrt(np.mean((pp[args.history:] - pos[a:b][args.history:]) ** 2)))
        per_seg.append((int(s), rmse, s in val_segs))

    tr_rmse = np.mean([r for _, r, v in per_seg if not v]) if any(not v for _, _, v in per_seg) else float("nan")
    va_rmse = np.mean([r for _, r, v in per_seg if v]) if any(v for _, _, v in per_seg) else float("nan")
    print(f"[mlp] closed-loop position RMSE (rad)  train {tr_rmse:.4g}  val {va_rmse:.4g}")
    for sid, rmse, isval in per_seg:
        print(f"        segment {sid:3d} [{'val ' if isval else 'train'}]  rmse {rmse:.4g}")

    # --- save checkpoint + config ---
    ckpt = os.path.join(args.out_dir, "servo_mlp.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "history": args.history,
            "hidden": list(args.hidden),
            "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
            "control_hz": args.control_hz, "step_dt": step_dt,
            "ref_lag_steps": args.ref_lag_steps, "velocity_source": args.velocity_source,
        },
        ckpt,
    )
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "csv": args.csv, "control_hz": args.control_hz, "ref_lag_steps": args.ref_lag_steps,
                "history": args.history, "hidden": list(args.hidden), "velocity_source": args.velocity_source,
                "val_frac": args.val_frac, "best_val_loss": best_val,
                "train_rmse": tr_rmse, "val_rmse": va_rmse,
                "val_segments": sorted(int(s) for s in val_segs),
            },
            f, indent=2,
        )
    print(f"[mlp] saved model to {ckpt}")

    # --- dump rollout arrays (lets you render the interactive plot from any env with plotly) ---
    npz_path = os.path.join(args.out_dir, "rollout.npz")
    np.savez(
        npz_path, t=t, target=target, real=pos, pred=pos_pred, seg_ids=seg_ids,
        val_segs=np.asarray(sorted(int(s) for s in val_segs)), seg_len_steps=seg_len_steps, step_dt=step_dt,
    )

    # --- plots: target / real(delay-comp) / MLP ---
    _save_static_plot(os.path.join(args.out_dir, "mlp_fit.png"), t, target, pos, pos_pred, seg_ids, val_segs)
    if _HAVE_PLOTLY:
        _save_interactive_plot(os.path.join(args.out_dir, "mlp_fit.html"), t, target, pos, pos_pred,
                               seg_ids, val_segs, seg_len_steps, step_dt)
        print(f"[mlp] saved interactive plot to {os.path.join(args.out_dir, 'mlp_fit.html')}")
    else:
        print(f"[mlp] plotly not in this env; static PNG saved and arrays dumped to {npz_path}.")
        print(f"[mlp] render the interactive HTML from a plotly env with:\n"
              f"      python {os.path.relpath(__file__)} --plot-npz {npz_path}")


def _save_static_plot(path, t, target, real, pred, seg_ids, val_segs):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[3, 1])
    ax0.plot(t, target, label="target (cmd)", color="tab:gray", lw=1.0, ls="--")
    ax0.plot(t, real, label="real (delay-comp)", color="tab:blue", lw=1.5)
    ax0.plot(t, pred, label="MLP", color="tab:orange", lw=1.3, alpha=0.9)
    ax0.set_ylabel("position (rad)")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("MLP servo model: closed-loop rollout vs real")
    ax1.plot(t, pred - real, color="tab:red", lw=0.9)
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("MLP - real (rad)")
    ax1.set_xlabel("time (s)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_interactive_plot(path, t, target, real, pred, seg_ids, val_segs, seg_len_steps, step_dt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=target, name="target (cmd)", line=dict(color="gray", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=t, y=real, name="real (delay-comp)", line=dict(color="royalblue", width=2)))
    fig.add_trace(go.Scatter(x=t, y=pred, name="MLP", line=dict(color="darkorange", width=2)))
    # shade validation segments
    seg_dt = seg_len_steps * step_dt
    for s in sorted(val_segs):
        fig.add_vrect(x0=s * seg_dt, x1=(s + 1) * seg_dt, fillcolor="LightSalmon", opacity=0.15, line_width=0)
    fig.update_layout(
        title="MLP servo model: closed-loop rollout vs real (shaded = held-out val segments)",
        xaxis_title="time (s)", yaxis_title="position (rad)", hovermode="x unified", template="plotly_white",
    )
    fig.write_html(path)


if __name__ == "__main__":
    main()
