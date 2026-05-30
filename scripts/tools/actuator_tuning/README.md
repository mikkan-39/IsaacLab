# Actuator parameter tuning

Replay a recorded real-servo trajectory on a **fixed-base** RT robot (same robot as RTv5,
`RT_CFG`) and fit the simulated `DCMotor` parameters so the virtual joint tracks the real one.

- Environment: `Isaac-Actuator-Tuning-RT-v0`
  (`source/isaaclab_tasks/isaaclab_tasks/direct/actuator_tuning/`)
- Recording shipped at: `.../direct/actuator_tuning/data/servo_recording.csv`
  (columns: `timestamp,target_rad,position_rad,error_rad,speed_rad_s`)

## How it works

- The robot base link is fully fixed (`fix_root_link=True`). One joint (`--joint-name`,
  exact USD name, default `base_link_to_Neck_revolute`) replays the recorded `target_rad` at a
  configurable control rate (default 50 Hz via `decimation`); all other joints are frozen at
  their defaults.
- The RT USD ships tight angular limits on its joints. By default every joint's position limit
  is widened to +/- pi so replayed targets are never clipped by the solver. Use
  `--joint-pos-limit LOW HIGH` to set a custom range, or `--keep-usd-limits` to keep the USD
  limits.
- The recording (50-1000 Hz) is downsampled to the control rate with a zero-order hold
  (no interpolation). Replay starts at the first measured `position_rad`.
- Each parallel environment holds a different `DCMotor` parameter vector but replays the
  **same** trajectory, so a whole population is evaluated in one headless sim.

## Scoring (position-first)

```
weighted_pos_mse = sum(w * (sim_pos - real_pos)^2) / sum(w)
tracking         = position_only: weighted_pos_mse
                   position_heavy: 0.9*pos + 0.1*vel
                   balanced:       0.7*pos + 0.3*vel
spike_pos_err    = p99(|sim_pos - real_pos|)                     # high percentile, not raw max
score            = 0.8 * tracking + 0.2 * spike_pos_err          # lower is better
```

The spike term is a high percentile of the absolute position error (default p99, set with
`--spike-percentile`) rather than the raw `max`, so a single unavoidable step-instant doesn't
dominate the score. `max_abs_pos` is still logged for reference.

Per-step weights `w` emphasize the first ~300 ms after movement starts (2x) and direction
reversals / sudden accelerations (2x). Velocity MSE is always logged but de-emphasized by
default (`--score-mode position_only`), because differentiated encoder velocity is noisy.

> Windows / PowerShell notes
>
> - Use `.\isaaclab.bat -p .\scripts\tools\...` instead of `./isaaclab.sh -p scripts/tools/...`.
> - PowerShell mangles inline JSON (it strips the inner `"`), so for `run_replay.py` prefer the
>   quote-free `--set NAME=VALUE` form, or pass a `.json` file to `--params`.
> - Line continuation in PowerShell is a backtick `` ` `` (not `\`). The examples below are
>   shown on one line so they paste cleanly.

## Workflow

1. Visual sanity check (single param set, GUI + overlay plot).

PowerShell (recommended, quote-free):

```powershell
.\isaaclab.bat -p .\scripts\tools\actuator_tuning\run_replay.py --set stiffness=28.1 --set damping=1.7 --set effort_limit=1.96 --set velocity_limit=11.1 --set saturation_effort=1.96 --set armature=0.01
```

Or pass a JSON file (create `params.json` with the dict, then):

```powershell
.\isaaclab.bat -p .\scripts\tools\actuator_tuning\run_replay.py --params .\params.json
```

bash / Linux:

```bash
./isaaclab.sh -p scripts/tools/actuator_tuning/run_replay.py \
    --params '{"stiffness": 28.1, "damping": 1.7, "effort_limit": 1.96, "velocity_limit": 11.1, "saturation_effort": 1.96, "armature": 0.01}'
```

2. Pass 1 - LHS/random search (friction locked at 0), headless, parallel:

```powershell
.\isaaclab.bat -p .\scripts\tools\actuator_tuning\run_sample_search.py --headless --num-envs 64 --search-yaml .\scripts\tools\actuator_tuning\search_spec.example.yaml --output-dir logs\actuator_tuning\run01
```

Outputs: `sample_results.csv` (ranked), `best_grid.json`, `plots/rank_*.png`.

3. Refine the top-k with local optimization (scipy Nelder-Mead):

```powershell
.\isaaclab.bat -p .\scripts\tools\actuator_tuning\run_refine.py --headless --results logs\actuator_tuning\run01\sample_results.csv --search-yaml .\scripts\tools\actuator_tuning\search_spec.example.yaml --output-dir logs\actuator_tuning\run01 --top-k 5
```

Outputs: `best_refined.json`, refreshed `plots/rank_*.png`.

4. (Optional) Pass 2 - unlock friction, only if reversals still mismatch:

```powershell
.\isaaclab.bat -p .\scripts\tools\actuator_tuning\run_friction_pass.py --headless --seed-params logs\actuator_tuning\run01\best_refined.json --search-yaml .\scripts\tools\actuator_tuning\friction_spec.example.yaml --output-dir logs\actuator_tuning\run01_friction --reversal-mse-threshold 0.01
```

(All scripts default `--joint-name base_link_to_Neck_revolute`; pass `--joint-name <name>` to change it.)

## Interactive plots (zoom/pan into fast bursts)

Static PNGs are hard to read for a 3-minute recording. Every search/refine/replay run also
writes a `trajectories.npz` (top-k sim curves + target/real + params/metrics). Turn it into a
zoomable Plotly HTML with the standalone builder.

Important: run this with your **normal Python that has plotly** (e.g. anaconda), NOT
`isaaclab.bat` (the Isaac Python does not ship plotly):

```powershell
python .\scripts\tools\actuator_tuning\make_interactive_plots.py --run-dir logs\actuator_tuning\run01 --open
```

This writes `logs\actuator_tuning\run01\interactive.html`. In the browser:

- Box-zoom / pan / range-slider to inspect the fast-motion bursts at full resolution.
- Use the rank dropdown (top-right) to switch candidates; target and real stay pinned, the
  selected rank's sim + error curves toggle. Pick `all` to overlay every rank's sim.
- The title shows the selected rank's parameter values and score.

(If you prefer it inline during the run, `pip install plotly` into the Isaac Python; otherwise
the npz + this script is the intended path.)

Skips automatically if the seed's `reversal_pos_mse` is below the threshold (use `--force`
to run regardless).

## Tunable parameters

Tunable (any can go in a single search YAML): `stiffness`, `damping`, `velocity_limit`,
`effort_limit`, `saturation_effort`, `armature`, `friction`, `dynamic_friction`,
`viscous_friction`.

Coupling / decoupling:

- `effort_limit == effort_limit_sim` (coupled).
- `velocity_limit` shapes the **DCMotor torque-speed curve only**. The hard PhysX solver speed cap
  (`velocity_limit_sim`) is **decoupled** and held fixed at `--solver-velocity-limit` (default 20
  rad/s, set above your measured peak), so the search cannot fake reversal lag with a low cap.

The physically-correct levers for the real servo's reversal lag are `friction` /
`dynamic_friction` (resist starting/reversing) and `armature` (rotor inertia) -- include those in
the search rather than letting `velocity_limit` / `damping` compensate non-physically. The
separate `run_friction_pass.py` (seeded from a best JSON) remains available if you prefer a staged
friction-only pass.

Edit the `*_spec.example.yaml` files to change bounds, `n_samples`, or sampling `method`
(`latin_hypercube` or `random`); per-param `log: true` samples in log-space.

## Tips

- Use `--max-duration-s` to fit on a short slice first (much faster iteration).
- The full recording is ~150 s; at 50 Hz that is ~7500 control steps per replay.
- Always eyeball `plots/rank_*.png` (target vs real vs sim): a good score can still hide a
  wrong-shape trajectory near startup/reversals.
