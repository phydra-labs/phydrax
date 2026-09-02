# Biophysical observation and qualification

`phydrax.observation` provides immutable plans and prepared, fixed-shape JAX
runtimes for turning trajectories, fluorescence traces, photon histograms, and
electrophysiology sweeps into typed scientific results. Configuration is
content-addressed by `plan_id`; `prepare()` constructs reusable indices or
instrument operators once, outside compiled evaluation.

## Correlation observables

The MSD convention is

```text
MSD(k Δt) = mean_t ||x(t + k Δt) - x(t)||².
```

```python
import jax.numpy as jnp
from phydrax.observation import MeanSquareDisplacementPlan

runtime = MeanSquareDisplacementPlan(
    256,
    3,
    64,
    1.0e-3,
    distance_unit="m",
    time_unit="s",
).prepare()
result = runtime.forward(positions_m)  # positions_m.shape == (256, 3)
```

`AutocorrelationPlan` subtracts the trace's sample mean and divides each lag
sum by its available pair count `N-k`, optionally normalizing by the resulting
zero-lag variance. This is the conventional sample-mean-centered finite-record
estimator, not an unbiased autocovariance estimator for an estimated mean.
`FluorescenceCorrelationPlan` uses the FCS convention
`mean(δI(t) δI(t+τ)) / mean(I-background)²`. A constant autocorrelation trace or
a nonpositive background-corrected FCS mean is not identifiable. Normalized
correlations are evaluated after scale normalization, so changing signal units
does not change identifiability.

`PairCorrelationPlan` evaluates every integer lag from `-max_lag` through
`+max_lag`. A positive `peak_lag` means the first input leads the second. Its
`directionality` is the positive-lag correlation minus the corresponding
negative-lag correlation at the inferred lag magnitude. Tied directional peaks
or zero positive/negative asymmetry are unidentifiable and return NaN peak
evidence.

All correlation runtimes require exactly the planned sample shape. This avoids
data-dependent allocation under JIT and makes `pair_counts` explicit at every
lag.

## Diffusion observables

`DiffusionModelPlan(lag_times, dimension, "anomalous")` evaluates

```text
MSD(0) = 0,
MSD(t) = 2 d D_alpha t^alpha + 2 d sigma_localization²  for t > 0,
```

with `0 < alpha <= 2`. The `"confined"` model evaluates the stationary
Ornstein-Uhlenbeck form

```text
MSD(0) = 0,
MSD(t) = 2 d D tau_c (1 - exp(-t/tau_c)) + 2 d sigma_localization²  for t > 0.
```

`forward()` returns the analytic curve. `evaluate()` adds Gaussian residual,
chi-square, and log-probability evidence for observed MSD values and known
standard errors.
The result is identifiable only when the prepared positive-lag design has
full-rank local sensitivities for `D` and the active exponent or confinement
time; for example, `[0, 1]` cannot identify either shape parameter.

`BrightnessConditionedTransportPlan` assigns each active fixed-capacity
displacement to exactly one immutable brightness bin and reports
`D = mean(||Δx||²)/(2 d Δt)`. The final brightness edge is inclusive. An active
brightness outside the recorded edge range fails closed. A bin is identifiable
only when it reaches `minimum_count`; `active` masks unused fixed capacity
without changing shapes.

## Fluorescence lifetime and FRET photons

`FluorescencePhotonPlan` takes uniformly spaced, strictly increasing time-bin
edges beginning at zero and one nonnegative instrument-response mass per bin.
The uniform grid makes the prepared index-based causal convolution a physical
time shift. The response is normalized once. Exponential bin mass is integrated
with a cancellation-safe `expm1` form rather than sampled at bin centers.
FRET uses

```text
tau_DA = tau_D (1 - E),  0 <= E <= 1.
```

At `E = 0`, the donor lifetime is unchanged. At the exact `E = 1` limit, the
intrinsic photon is prompt and the detected distribution equals the normalized
instrument response (subject to the recorded window). That limiting histogram
is finite but does not identify the donor lifetime.

```python
import jax
import jax.numpy as jnp
from phydrax.observation import FluorescencePhotonPlan

runtime = FluorescencePhotonPlan(
    jnp.linspace(0.0, 12.0e-9, 129),
    measured_irf_probability,
    time_unit="s",
).prepare()
expected = runtime.expected(2.4e-9, 20_000.0, fret_efficiency=0.35)
draw = runtime.forward(
    jax.random.key(7), 2.4e-9, 20_000.0, fret_efficiency=0.35
)
fit = runtime.evaluate(
    draw.photon_counts, 2.4e-9, 20_000.0, fret_efficiency=0.35
)
```

The key fully determines the Poisson draw. `evaluate()` uses the exact
independent-Poisson log likelihood, including `log(k!)`, and rejects fractional
or negative observed counts through result evidence.

## Channel dwell times and I-V reversal

`DwellTimeLikelihoodPlan(capacity)` evaluates a right-censored exponential
channel model. `event_observed=True` contributes `log(rate) - rate*time`; a
right-censored dwell contributes only `-rate*time`. Therefore
`maximum_likelihood_rate = event_count / total_exposure`. An all-censored data
set has a finite likelihood at a supplied positive rate but no finite MLE and is
reported as unidentifiable.

`IVReversalPlan` normalizes relative weights, prepares scale-normalized voltage
moments, and fits `I = conductance * (V - reversal_potential)`. Rescaling every
weight cannot alter the fit or its rank evidence. A flat I-V curve, a voltage
grid with zero weighted variation, or a nonfinite inferred reversal returns
`identifiable=False` and a NaN reversal potential instead of silently dividing
by zero.

## Evidence and differentiation

Every typed result exposes `finite`, `identifiable`, and `successful` arrays.
`successful` is fail-closed: it is true only when the numerical output is finite
and the requested inverse quantity is identifiable. Per-bin transport results
also expose `identifiable_bins`. Invalid numerical observations stay in the
compiled path and are reported by evidence; constructor errors are reserved for
invalid static shape, unit, or instrument configuration.

Forward models and likelihoods are differentiable with fixed sample capacity,
lag geometry, bins, active masks, censor indicators, and instrument response.
Gradients through changing event classifications, changing histogram topology,
or a different fixed-capacity layout are not defined.

## Independent SI qualifications

`phydrax.qualification._biophysics` contains independent analytic checks:

- `spherical_membrane_capacitance` evaluates `4 pi r² c_m` in farads;
- `spherical_membrane_ion_count` converts `|C V|` to an ion count using valence;
- `nernst_equilibrium_potential` uses
  `RT/(zF) log(c_out/c_in)` for `psi_in - psi_out`;
- `eyring_rate` uses `kappa k_B T/h exp(-Delta G‡/(RT))` in inverse seconds;
- `antiporter_electrochemical_balance` solves the signed electrochemical cycle
  and marks zero net transported charge as an unidentifiable voltage;
- `recover_brownian_transport` recovers drift and uses the `N-1` finite-sample
  correction for scalar diffusion after estimating drift from the same SI
  position increments; and
- `qualify_censored_dwell_times` independently checks the exponential survival
  likelihood and censor-aware MLE.

Concentrations are in `mol/m³`, temperature in kelvin, molar free energies in
`J/mol`, potential in volts, distances in metres, and time in seconds unless a
plan records another explicit display unit. Unit strings document the numerical
contract; PhydraX does not rescale arrays implicitly.
