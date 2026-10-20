# Calibrating output initialization

`compute_initialization_gains` in `src/training_profiling_utils.py` estimates
Xavier gains across several **fresh, untrained** `LatentStateModel` instances. Run it manually before
training, using representative training slices with the same frame interval,
units, warm-up, and architecture as the intended run. It does not load data or
change the supplied model, observation history, or random-number state.

The model retains 0.01 for dynamics and 0.1 for the decoder as compatibility
defaults. These are configurable starting points, not universal recommendations.

The command accepts only the action name. It collects a training split using the
current defaults from `train.py`, selects CUDA when available and CPU otherwise,
calibrates the current default model across five seeds, and prints the complete
report as JSON. Change the training defaults before running it when evaluating a
new scenario or architecture.

## What the targets mean

- **Decoder:** match initial decoded displacement RMS to observed displacement
  RMS. `decoder_target_ratio=1.0` means equal RMS; this controls initial scale,
  not prediction accuracy or direction. Only consecutive observed frames after
  warm-up contribute, and padding is excluded.
- **Dynamics:** keep maximum RMS latent drift across the autonomous forecast
  within `rollout_drift_budget=0.1` of the starting latent RMS. This is a deliberate
  10% near-identity budget, not a physical constant estimated from the trajectory.
  With a ten-frame horizon, the first estimate targets a 1% residual per step.
  Dividing by the horizon assumes updates may accumulate coherently, since the
  same dynamics weights are used at each step. The function then measures the
  repeated nonlinear rollout and halves the gain until it fits the budget.

For fixed inputs and zero output biases, scaling a final weight matrix scales
that branch's output by the same factor. The dynamics gain is multiplied by
`(budget / horizon) / measured_residual_ratio`, with the rollout check above.
The decoder gain is multiplied by `desired_displacement_RMS / decoded_RMS`,
measured after the dynamics adjustment.

For each seed, calibration holds that model's sampled observation-updated states
fixed. The public recommendation is the median suggested gain across seeds; the
report also includes means, standard deviations, and complete per-seed diagnostics.
The median is used because one unusually quiet or large random output has less
influence on it. Reconstructing a model with the median gains changes its state
path slightly, so the reported after-values are scale diagnostics rather than a
guarantee for every new initialization or scenario.

## When to rerun

Rerun after changing widths, depth, activations, normalization, latent size,
observation features, scene or speed distributions, coordinate units, frame
interval, warm-up, or prediction horizon. Check several representative batches
and several seeds. Increase the default five seeds when the reported standard
deviation is large relative to the median or when the per-seed values span more
than roughly a factor of three.

On a fresh model using the proposed gains, practical warning thresholds are:

- `decoder_ratio_before` outside roughly **1/3 to 3** times the target ratio.
- `rollout_drift_ratio_before` above **twice the drift budget** (20% by default),
  or below **one tenth of it** (1% by default). The lower threshold flags a
  potentially unnecessarily small residual; it is not itself evidence of failure.

These are engineering tolerances, not statistical confidence bounds. The drift
budget and decoder target ratio remain explicit choices. Changing them changes
the initialization prior; it does not infer the true dynamics. The RMS diagnostics
also do not bound worst-case samples or the dynamics Jacobian.

Entirely stationary samples cannot identify a nonzero displacement scale, and
dead branch outputs cannot be repaired by multiplication. The function raises
an explanation for these cases instead of recommending zero weights. Samples
too short to contain a forecast after warm-up are also rejected.
