# Deterministic reduced climate

`phydrax.applications.climate` couples explicit perturbation gas inventories to a
serial multilayer energy-balance model. It is a deliberately bounded reduced
model, not an atmospheric dynamics solver or an assessed projection system.
The implementation uses native `StrictModule`, fingerprints, `TimeGrid`,
`FixedStepProblem`/`FixedStepRolloutPlan`, `phydrax.linalg` matrix-function actions,
and the native array archive. There is no second rollout, ensemble, calendar, or
checkpoint framework.

## Scientific scope and units

Gas order is **CO2, CH4, N2O**. Each has a finite collection of well-mixed,
linear perturbation reservoirs. Reservoir arrays have shape `(3, boxes)`.
A box stores an **excess inventory**, not the absolute atmospheric stock.
Absolute concentration is its gas's explicit background plus summed excess
inventory divided by the declared inventory-per-concentration conversion.

| Gas | Inventory and cumulative sink | Absolute concentration | Default inventory/concentration |
| --- | --- | --- | --- |
| CO2 | GtC, not GtCO2 | ppm | 2.124 GtC/ppm |
| CH4 | TgCH4 | ppb | 2.78 TgCH4/ppb |
| N2O | TgN2O, not TgN | ppb | 7.8 TgN2O/ppb |

Defaults are explicit, configurable illustrative parameters. Backgrounds are
278.3 ppm, 729.2 ppb and 270.1 ppb, respectively. These are reference choices,
not a time-dependent historical reconstruction. Mass/concentration conversion
constants are approximate and do not recompute dry atmospheric mass.

Internal duration is a **365.25-day model year** (`MODEL_YEAR_SECONDS`).
Emissions use inventory/model-year; gas decay rates use inverse model-years.
No calendar year is silently assumed to have this length. The prepared runtime
converts its bound numerical clock to seconds and then to model-years.
Layer heat capacities use W model-year m^-2 K^-1; feedback and interlayer
conductance use W m^-2 K^-1. Temperatures are K anomalies; forcing uses W m^-2.
Integrated forcing, outgoing energy, and heat content use W model-year m^-2;
multiply by `MODEL_YEAR_SECONDS` to obtain J m^-2.

The default CO2 fractions `(0.2173, 0.2240, 0.2824, 0.2763)` and reservoir
lifetimes `(infinite, 394.4, 36.54, 4.304)` model-years use the widely employed
four-term impulse-response fit associated with
[Joos et al. (2013)](https://doi.org/10.5194/acp-13-2793-2013).
The fixed CH4 and N2O perturbation lifetimes, 9.3 and 121 model-years, are
illustrative configurable choices. They are not a full chemistry model or a
statement of a uniquely calibrated present-day lifetime. No ocean/land sink
partition, methane oxidation source into CO2, permafrost, aerosol chemistry,
spatial feedback, stochastic variability, or carbon-cycle parameter
calibration is inferred from these defaults.

Signed inventories/emissions/net sinks are allowed for removals and
below-reference perturbations. **Absolute concentrations must remain
positive.** The cumulative sink is net atmospheric removal into unrepresented
reservoirs, not a separate ocean or land stock. An immortal CO2 reservoir has
zero decay and retains its supplied perturbation indefinitely.

## Exact subproblems, discretized coupling

For a frozen lifetime multiplier, each reservoir solves
`dB/dt = fraction * emission - decay_rate * B / multiplier` exactly over one
interval with constant emissions. The source gain uses a cancellation-safe
`(1 - exp(-x)) / x` and its analytic zero limit, including finite derivatives.
An immortal reservoir therefore gains exactly `fraction * emission * dt`.
The net sink independently integrates physical decay loss, with a small-rate
source-loss series rather than a budget remainder. It balances interval
supplied inventory minus airborne change; cumulative emissions and sinks
advance only on commit.

A configurable state-dependent lifetime response targets the gas's integrated
impulse response over a declared horizon. Its three coefficient columns
multiply **cumulative net sink, entry surface temperature, entry airborne
inventory**, respectively. Each coefficient carries the corresponding units
needed to make its contribution a model-year. The baseline is computed from
the model's own reservoir fractions/rates at multiplier one. All-zero response
coefficients select multiplier one without a nonlinear solve.

The integrated response is monotone in a positive multiplier whenever a
positive-fraction decaying box exists. A fixed-iteration log-space bisection
solves inside explicit positive bounds, then certifies the target bracket,
positive derivative, and scaled residual. Unattainable targets, insufficient
iterations, invalid concentrations, nonfinite state, or failed budget evidence
return `successful=False` and **retain every incoming state leaf unchanged**.
There is no target clipping, silent root fallback, or partial budget commit.
Derivatives at certified smooth roots use the implicit response derivative,
not derivatives of discrete bisection comparisons. Failed roots have no
physical sensitivity contract.

For serial thermal layers, exchange appears with opposite signs in the two
adjacent layer energy equations. Only the surface receives forcing and
radiates linear-feedback energy to space. The thermal subproblem is advanced
with native exponential and phi1 actions. An augmented integral coordinate
tracks the surface-temperature integral independently, allowing heat storage
to be checked against supplied minus radiated energy. Singular generators,
including zero surface feedback, do not require inversion. A tiny singleton
dense operator batch selects the existing native dense matrix-function path;
this is a numerical implementation detail, not a model member axis.

**These exact linear maps do not make the whole nonlinear climate evolution
exact.** Lifetimes are frozen at interval entry, and the constant thermal
forcing is the mean of the interval's incoming/outgoing gas forcing. External
and explicitly prescribed gas forcing are interval-constant. The full
coupling is generally first order with state-dependent lifetimes; timestep
refinement remains necessary. With fixed lifetimes the gas reservoirs are
exact for piecewise-constant emissions, but nonlinear concentration-to-forcing
and thermal coupling are still discretized.

## Named forcing and provenance

`Myhre1998Forcing` implements the simplified adjusted radiative forcing
relationships of [Myhre et al. (1998)](https://doi.org/10.1029/98GL01908), as
used in IPCC TAR's separate-gas approximation:

- CO2: `5.35 * log(C/C0)` with C in ppm.
- CH4 direct: `0.036 * (sqrt(M) - sqrt(M0))` with M in ppb.
- N2O direct: `0.12 * (sqrt(N) - sqrt(N0))` with N in ppb.
- Overlap: `f(M,N) = 0.47 * log(1 + 2.01e-5*(M*N)^0.75 + 5.31e-15*M*(M*N)^1.52)`.
- CH4 overlap: `-[f(M,N0) - f(M0,N0)]`.
- N2O overlap: `-[f(M0,N) - f(M0,N0)]`.

The other gas is deliberately held at its **background** in each published
separate-gas overlap term. Replacing this with an unqualified simultaneous-gas
formula would define a different approximation. All five terms are returned
individually with physical names, followed by user-named external channels.
`source` and `forcing_id` carry source and formula convention; a forcing-driven
gas uses a `*_prescribed` name instead of claiming a calculated logarithmic or
square-root contribution.

This is **not** effective radiative forcing, an updated CO2/CH4/N2O fit, or a
line-by-line radiative transfer calculation. It omits methane shortwave,
oxidation/ozone/stratospheric-water effects and rapid adjustments. Named
external channels can supply declared approximations to omitted effects;
the package does not invent them. Positive concentration admission is a
mathematical domain check, not certification of arbitrary extreme
concentrations against the original fit. There are no external model parity
or assessed projection claims.

## Driver roles and a native run

Each gas has exactly one static role:

- `emissions`: interval-constant inventory/model-year drives its reservoirs.
- `concentration`: an absolute **endpoint** concentration determines the unique
  interval-constant emission using the same frozen-lifetime source gain.
  Returned `inferred_emissions` can reproduce the forward reservoir map.
- `forcing`: reservoirs and gas sink/emission accounting remain unchanged;
  the prescribed interval forcing replaces both its direct and overlap terms.

Inactive driver entries are ignored. External channels are additive interval
forcing, never implicit gas emissions. A driver trajectory has a leading step
axis, and concentration values refer to interval ends, not interval starts.

```python
import jax.numpy as jnp
from phydrax.applications.climate import (
    ClimateDrivers, MODEL_YEAR_SECONDS, Myhre1998Forcing, ReducedClimatePlan,
)
from phydrax.dynamics import TimeGrid

plan = ReducedClimatePlan(forcing=Myhre1998Forcing(("solar",)))
runtime = plan.prepare(
    TimeGrid(jnp.arange(11.), time_id="ten-model-years"),
    seconds_per_time_unit=MODEL_YEAR_SECONDS,
)
initial = runtime.initial_state()
drivers = ClimateDrivers(
    jnp.broadcast_to(jnp.array([10., 20., 3.]), (10, 3)),
    jnp.broadcast_to(plan.gases.background, (10, 3)),
    jnp.zeros((10, 3)),
    jnp.zeros((10, 1)),
)
result = runtime.rollout(initial, drivers, retention="trajectory")
if not bool(result.successful):
    raise RuntimeError("Climate interval rejected; inspect native residuals/validity")
print(result.final_state.temperature)
```

Alternatively pass the shared `GeophysicalTimeSpec` as `time_spec` when
preparing. The shared host encoder supplies numerical times; climate neither
implements a calendar nor resamples calendar dates. Exactly one of
`time_spec` and `seconds_per_time_unit` is required. Fixed rollout admits only
a uniform numerical grid; calendar years/months need not produce one.

`runtime.step` exposes detailed root, gas, and energy evidence. The native
rollout exposes success/validity, accepted states and solver residuals. Once
an interval fails, its final state remains the last accepted state and later
saved states are invalid. Default numerical tolerances also support float32;
qualification and close analytical comparisons use float64.

## Scenarios, calibration, and restart

Build the immutable plan, prepared runtime and native problem **outside** JAX
transforms. Numerical leaves can then be mapped through the native problem
with `jax.vmap`/`eqx.filter_vmap` and differentiated with JAX. Do not mutate a
prepared object's coefficients and assume its static content identity changes;
construct a new plan/prepare for a persistent scientific configuration.
Checkpoint contracts additionally fingerprint actual numerical coefficients,
so changed leaves cannot restore a checkpoint from a different realization.
The enabled lifetime-response mask is a static scientific choice made when
constructing `GasBoxModel`: a configuration that introduces a previously
disabled gas response must construct a new model rather than replacing an
inactive coefficient leaf.

`examples/reduced_climate_scenarios.py` runs scenario × configuration × member
axes using nested native JAX mapping, reports member moments, and calibrates
surface feedback against a clearly synthetic observation with the rollout's
JAX gradient. It is not an observational climate constraint or an assessed
uncertainty distribution.

`write_reduced_climate_checkpoint(path, runtime, accepted_state, drivers)` uses
the existing atomic, checksum-protected, pickle-free array archive.
`read_reduced_climate_checkpoint(path, runtime, initial_template, drivers)`
requires the exact numerical model, clock, and complete driver trajectory.
A restart uses `runtime.problem(restored, drivers, start_step=k)` or the same
window arguments to `rollout`; `k` must equal the accepted state's step index.
No interpolation, driver replacement, or rebasing of cumulative accounting
occurs. First windows can use `stop_step=k`. The full driver trajectory remains
the single scenario input for both windows.

Run the scenario/calibration example and scientific qualification with:

```sh
python examples/reduced_climate_scenarios.py
python tools/reduced_climate_qualification.py
```

The qualification command measures analytical zero/near-zero/finite decay,
a one-layer thermal analytical limit, concentration/emission inverse error,
conserved inventory and energy, whole-model state-dependent timestep
refinement, cold/hot runtime, and bitwise native restart parity. It exits with
failure when its measured acceptance criteria fail. Focused regressions in
`tests/unit/test_reduced_climate.py` additionally defend equilibrium, singular
thermal dynamics, implicit lifetime sensitivity, rejected-step atomicity,
clock conversion, forcing overlap and changed-scenario checkpoint rejection.
These are internal qualification cases, not external calibration or parity.
