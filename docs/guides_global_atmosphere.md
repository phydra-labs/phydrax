# Global hydrostatic reference atmosphere

`GlobalPrimitiveEquationPlan` implements a rotating spherical hydrostatic primitive-equation PDE. It is a small-resolution reference, not an operational forecast model. Horizontal derivatives and Helmholtz wind inversion use native spin-weighted spherical operators. Layers interact through mass continuity, vertical transport, hydrostatic pressure gradients and pressure work; they are not independently advanced columns.

## Running

From the repository root with the project environment active:

```sh
JAX_ENABLE_X64=1 python examples/global_atmosphere.py --steps 2
JAX_ENABLE_X64=1 python examples/global_atmosphere.py --moist --steps 2
JAX_ENABLE_X64=1 python tools/global_atmosphere_qualification.py --cases rest wave moist --steps 2
JAX_ENABLE_X64=1 python tools/global_atmosphere_qualification.py --cases wave --steps 4 --refine
JAX_ENABLE_X64=1 python tools/global_atmosphere_qualification.py --cases solid_rotation baroclinic held_suarez aquaplanet --steps 4
```

These commands are evidence-producing recipes, not precomputed passing claims. The qualification script emits actual acceptance, pressure-mass/water/closed-energy budgets, explicit filter work, linear solve residual, thermal/wind ranges, restart equality, and optional equal-duration time refinement. It exits unsuccessfully on rejected steps, including restart and refinement continuations. Use different `--bandlimit` and `--levels` values to investigate spatial convergence; no resolution-independent accuracy is implied by a short stable trajectory.

## State, geometry and numerical method

The retained state contains spherical-harmonic coefficients of relative radial vorticity, horizontal divergence, absolute temperature and surface pressure. Each of vapor, liquid and ice is stored as a **mass inventory per horizontal area**, not a separately conserved mixing ratio. The vertical axis is last: `(ell, m, level)` in coefficients and `(theta, phi, level)` in physical fields. East is positive longitude; north points toward decreasing colatitude. Positive curl is radially outward.

Physical diagnostics and initialization use `model.work_space`, the padded evaluation grid. Scalars and one-dimensional vertical profiles broadcast. The retained `plan.space` and padded `work_space` are intentionally different. `model.project` and `model.reconstruct` connect retained modes to that grid. `terrain` is the exception: its input belongs to the retained space's grid and is projected at preparation.

Interfaces follow `HybridPressureCoordinate`: pressure = A + B surface_pressure, increasing from top to bottom. The top is a fixed **positive pressure**, B_top = 0; the bottom equals surface pressure, A_bottom = 0 and B_bottom = 1. A zero-pressure logarithm is rejected, not floored. Layer mass is exactly interface-pressure difference divided by gravity, including moisture. Evaporation and precipitation change atmospheric mass and surface pressure.

Let M be layer mass, D the horizontal divergence of M times wind, S a physical mass source, and F the relative interface mass flux (pressure per time). Continuity is

```text
surface_pressure_t = gravity × sum(S − D)
F_top = 0
F_bottom_of_layer = F_top_of_layer − gravity × (D + M_t − S)
M_t = difference(B) × surface_pressure_t / gravity
```

The lower boundary flux vanishes by the same column identity, up to floating-point accumulation. `model.continuity` exposes surface pressure tendency, every interface flux and the corresponding pressure-coordinate omega. Centered vertical finite-volume fluxes transport temperature, water and momentum. Horizontal momentum uses the vector-invariant absolute-vorticity force, kinetic-energy gradient and pressure force. Water uses a conservative horizontal/vertical inventory divergence. By default, a negative stage or endpoint is rejected; there is no moisture floor, saturation clip, pressure floor, mass rescale or hidden conservation repair.

For resolved moist integrations,
`GlobalPrimitiveEquationPlan(..., water_limiter="conservative")` enables a
joint phase-composition admission operator. It alternates between:

1. the native pointwise simplex for nonnegative vapor/liquid/ice at the
   **original local total-water inventory**, and
2. the retained spherical space with the **original total-water coefficients**.

Individual phase means may change because phase mass is not a conserved
inventory. The represented total-water field cannot move between columns. Four
Dykstra iterations are followed by the smallest contraction toward the
equal-phase feasible field needed to close finite-iteration negativity; the
phase-repartition threshold prevents that closure from hiding unresolved
composition.

Temperature then changes at fixed composition to preserve local moist enthalpy,
followed by one uniform roundoff correction to close global energy. Evidence
separates total-water redistribution, phase-repartition mass, activation, and
energy roundoff. A step rejects when phase repartition exceeds
`maximum_water_phase_repartition_fraction`, when represented total water changes
beyond roundoff, or when the original total-water field itself is infeasible.
Unresolved input is never repaired during initialization. The qualification
compares against independent phasewise contraction with compensating vapor;
the joint projection must reduce both moved phase mass and weighted-L2 change
while preserving total water.

Hydrostatic layer temperature is piecewise constant in log-pressure integration. The pressure force is paired with the negative mass-weighted adjoint of that same logarithmic hydrostatic map in the thermodynamic equation. This is a finite-layer pressure-work approximation converging to `(R/cp) T omega/p`, **not** midpoint omega divided by pressure combined with a different logarithmic pressure force. The pairing is important: an inconsistent combination creates artificial pressure-work energy even at small horizontal truncation error.

Nonlinear products are evaluated on a grid with at least 1.5 times the retained bandlimit and then projected. This reduces polynomial product aliasing; it does not make reciprocals, pressure logarithms, moist closures or all higher-order products exactly alias-free. No dense full-globe dynamical matrix is assembled. Existing spherical precision and precompute/operator resource limits propagate to the padded space. Native transforms require float64/complex128 and enabled JAX x64; small resolutions are the default deliberately.

The fixed-step IMEX midpoint method treats a **genuine coupled dry hydrostatic gravity-wave operator** implicitly: divergence, every layer's temperature and surface pressure are solved together by degree using native batched LU blocks. The operator is the exact linearization of the discrete dry equations about uniform isothermal rest. Nonlinear advection, rotation, moisture feedback, forcing and departures from that reference are explicit. Rotation is not falsely included in a degree-diagonal gravity-wave claim. The residual of every implicit solve is checked. Fast-wave implicitness does not remove the explicit advection, mixing or phase-process step restrictions.

An optional exponential high-degree filter acts on vorticity, divergence and temperature. Its energy and kinetic-energy changes are measured separately. No water inventory is filtered. A timestep is accepted only when initial/stage/final admissibility, phase closure, linear solve, Courant bound and independent mass/water/energy budgets pass. Default tolerances are admission thresholds, not accuracy certifications. Rejection leaves physical state, time, held forcing and accumulated ledgers unchanged; only the rejection count changes.

## Terrain and the upper boundary

Terrain is a supplied surface geopotential in square metres per square second. The pressure force is evaluated as the gradient of `surface_geopotential + R_reference T_reference log(surface_pressure/reference_pressure)`, plus hydrostatic thermal anomalies and their remaining pressure metric. Thus uniform-temperature hydrostatic cancellation is discretely paired rather than subtracting large unrelated gradients.

Preparation actually projects the isothermal rest pressure `reference_pressure × exp(−terrain/(R_reference T_reference))`, evaluates its residual acceleration, and rejects terrain failing `terrain_rest_tolerance`. This is a concrete **isothermal-rest admission gate**. It is not a general well-balanced terrain guarantee, does not cover moist stratified resting atmospheres, and cannot rescue under-resolved topography.

The positive-pressure top is an impermeable moving isobaric lid, not a rigid fixed-height roof. Its pressure-work reservoir is included analytically: adding `p_top × (geopotential_top − surface_geopotential)/gravity` to atmospheric internal plus vertically integrated gravitational energy produces the column moist-enthalpy plus surface-geopotential energy used by `model.inventories`. This accounts for mechanical exchange with the constant-pressure exterior without an unreported energy boundary source.

## Moist processes and budgets

`GlobalAtmosphereProcesses(thermodynamics=MoistThermodynamicPlan(), ...)` activates the real constant-caloric ideal-mixture liquid/ice closure. Phase fractions are per total moist mass including condensate loading. The mixture gas constant and heat capacity enter pressure work and hydrostatics. Isobaric equilibrium conserves moist enthalpy; the finite relaxation tendency uses phase enthalpy differences, rather than independently relaxing temperature and moisture.

The composition provides:

- prescribed sensible heat and evaporation at the bottom layer;
- grey Newtonian radiative cooling with an equal-and-opposite environment-energy transfer;
- closed-boundary conservative vertical exchanges of phase inventories, enthalpy and momentum;
- liquid/ice precipitation tendencies removing actual mass, with corresponding surface-water and energy deposits;
- optional standard-form Held–Suarez thermal relaxation and boundary Rayleigh drag.

Surface water and surface/environment energy are actual continuation-state arrays. Evaporation draws water and vapor enthalpy from the surface reservoir, and precipitation returns phase enthalpy. Transported geopotential and kinetic energy are paired as well: falling condensate releases this energy into the surface reservoir, and prescribed co-moving vapor injection draws the same mechanical terms. These are explicit co-moving injection and instantaneous-fall approximations, not rain terminal-speed or cloud microphysics models. Without optional interactive surface physics, surface energy is an unrestricted signed reference-energy reservoir, not a prognostic sea-surface temperature. With that component, `initialize(surface_temperature=...)` constructs physical slab energy from its wet-slab closure; initial, stage and candidate slab temperature and water bounds participate in the same atomic admission gate.

Process tendencies are evaluated at **both numerical stages**. Only forcing fields (equilibrium temperature, cooling/drag rates, prescribed flux fields, and the distinct diffuse TOA `solar_down` field) are held on an accepted-step cadence. They, their age, both step counters, time, every physical reservoir and the cumulative ledger are in `GlobalAtmosphereContinuation` and in native checkpoints. Solar flux is never encoded in a temperature target.

`model.inventories` returns total atmosphere-plus-surface mass, total water, total closed energy including the upper-lid work reservoir, and atmospheric kinetic energy. Total closed energy additionally includes the surface and radiation/drag environment reservoirs. The independent external power of this composition is zero. The energy residual therefore subtracts **only the separately measured optional filter work**, not a computed process energy derivative. The ledger's `process_energy` is a diagnostic JVP of actual closed-energy change under the process tendency; a nonzero value exposes spatial/process inconsistency and is never used to define it away. Spectral projection, finite vertical transport and time discretization leave measurable residuals; no exact nonlinear energy conservation or long-time climate stability is claimed.

Native `write_global_atmosphere_checkpoint(path, model, continuation)` and `read_global_atmosphere_checkpoint(path, model, template)` use the repository array archive and strict model identity. A change to timestep, forcing, thermodynamic constants, vertical coordinate, terrain, transform or filter policy invalidates the checkpoint identity. There is no alternate restart schema or lossy restart path.

Checkpoint manifests additionally bind the **current numerical process leaves**
by their content-sensitive fingerprint, not only the static prepared identity.
Changing trainable physical parameters via a native pytree update therefore
invalidates restart compatibility even when an earlier static identity survives.

## Qualification boundaries

- **Rest:** analytic dry uniform isothermal rest, with actual dynamical step evidence.
- **Wave:** a weak hydrostatic gravity wave compared with an independent matrix exponential of the qualified linear operator; temporal refinement separates integration error from finite-amplitude effects.
- **Solid rotation:** analytic isothermal gradient-wind equilibrium, reporting retained pressure-projection error rather than promising exact representation of an exponential field.
- **Baroclinic:** globally coupled, vertically sheared thermal perturbation evolution. The supplied initial condition is intentionally described as unbalanced; it is not a named published balanced benchmark.
- **Held–Suarez:** actual idealized cooling and drag. A short executable case does not qualify equilibrated zonal climatology, eddy statistics or climate drift.
- **Controlled moist:** phase adjustment, condensation/evaporation, precipitation, cooling and paired reservoirs with the full global dynamics active.
- **Aquaplanet:** prescribed lower-boundary fluxes and zonal heating idealization. There is no coupled ocean, parameterized deep convection or real-world forecast skill claim.

Focused regressions defend rest, continuity/boundary closure, actual fast-operator linearization and vertical coupling, atomic rejection, moist reservoir conservation, checkpoint continuation across a forcing refresh, and rejection of unresolved terrain. Longer climate integrations, spatial convergence and application-specific tolerances remain measurements for the intended configuration, not inferred release qualifications.

## Analytic dry gradient/thermal-wind reference

`DryGradientWindReference` provides an independently derived, continuously
balanced sheared family on **flat ground**, distinct from the original
`global_atmosphere_qualification.py --cases baroclinic` unbalanced case:

```python
from phydrax.applications.atmosphere._balanced import DryGradientWindReference

reference = DryGradientWindReference(speed=20.0, shear=-10.0)
initial = reference.initialize(model)  # a prepared, unforced dry global owner
residual = reference.diagnostics(model, initial)
```

For latitude φ, x = log(p/p_ref), q = sin²φ, U = U₀ + s x:

```text
u = U cos φ,  v = 0
Φ = −R T₀ x − (Ω a U + U²/2) q
T = T₀ + s (Ω a + U) q / R
```

Differentiating gives ∂Φ/∂log p = −R T, and
`(1/a) ∂Φ/∂φ + 2 Ω sinφ u + u² tanφ/a = 0`.
Differentiating the gradient-wind equation in log-pressure gives
`(R/a) ∂T/∂φ = (2 Ω sinφ + 2 u tanφ/a) ∂u/∂log p`.
The implementation evaluates equivalent regular expressions in `sinφ cosφ`,
so neither zero shear, the equator nor the poles require singular divisions.
The unperturbed zonal fields are longitude-independent and nondivergent, with
zero pressure-coordinate vertical velocity; thus hydrostatic and horizontal
momentum balance also give an adiabatic steady primitive-equation solution.

The flat-ground pressure satisfies Φ(p_s, φ) = 0. With
`A = s²q/2`, `B = R T₀ + s(Ωa+U₀)q` and
`C = (ΩaU₀+U₀²/2)q`, the equator-connected root is
`log(p_s/p_ref) = −2C / (B + sqrt(B²−4AC))`.
This rationalized quadratic formula remains regular when A or C vanishes.
The reference checks positive-temperature root existence at every latitude,
temperature admission and static stability over its entire declared pressure
interval, and requires the model interfaces to lie inside that interval.
Its immutable parameter identity is not a trainable collection of secretly
changing reference data. `fields` reports unsuccessful evaluation outside the
declared interval or latitude domain; it never clips inputs into validity.

Initialization uses native hybrid-pressure midpoints, wind curl/divergence
analysis and retained scalar projection. It rejects terrain, active processes,
nonzero filtering, inconsistent physical constants and bandlimits below three. It does **not**
alter the pressure force or remove the model's initial residual. Continuous
balance does not imply exact modal representation of surface pressure or exact
piecewise-constant finite-layer hydrostatics. Diagnostics separately expose:

- surface-pressure, temperature and wind projection errors in Pa, K and m/s;
- finite-layer reconstructed geopotential error in m²/s²;
- mass-weighted RMS and maximum actual native RHS acceleration in m/s²;
- actual native temperature and surface-pressure tendencies in K/s and Pa/s.

### Independent evidence and qualification

```sh
JAX_ENABLE_X64=1 python tools/balanced_atmosphere_qualification.py --output balanced-results.json
JAX_ENABLE_X64=1 python tools/balanced_atmosphere_qualification.py \
  --bandlimits 4 6 8 --levels 2 4 8 --dt 120 --steps 8 \
  --wave-dt 300 --wave-steps 12
```

The script emits **measured** results and explicit dimensional criteria; these
recipes are not precomputed passing claims. Its campaign includes independent
centered-difference checks of all three continuous identities, nonzero
thermal-wind signal, separate horizontal and vertical sweeps, integrated
spurious wind, and finite-amplitude perturbation evolution after subtracting
a separately integrated unperturbed trajectory. The latter reports thermal
mode phase/amplitude and perturbation kinetic energy; it does not claim an
independently validated baroclinic growth rate for this derived family.

Independent wave evidence comes from a nonrotating, one-layer weak gravity
wave. For top-pressure ratio r, `h = R log(2/(1+r))` and spherical mode
`k² = ell(ell+1)/a²`, direct scalar linearization gives
`ω² = k²[T₀ h²/cp + R T₀(1−r)]`. The qualification compares the full native
trajectory with that oscillator's exact phase and unit amplitude; it does
not construct its reference from the owner's fast matrix. This qualifies a
semidiscrete one-layer wave, **not** a continuum vertical-mode spectrum.

The differentiable observable is the time integral of the resolved
mass-weighted enthalpy projection onto a degree-two longitudinal temperature
mode, in J s/m². Its AD action with respect to a Kelvin-valued initial thermal
perturbation is compared with two centered finite-difference step sizes and
equal-duration dt, dt/2, dt/4 integrations. Both the trajectory and AD action
are compared across time refinement. Only accepted trajectories support a
derivative claim.

The campaign additionally measures admitted isothermal terrain rest and an
analytically stratified resting profile's **nonzero discrete** terrain residual.
Smooth, strictly positive, dilute water vapor is compared with independent
solid-rotation advection over the reported angle. This is the existing active
water/thermodynamic path in its dilute limit, not a new passive tracer or a
claim of monotone sharp-front transport. Unresolved or nonphysical input
boundaries are reported as explicit rejections.

The primary published comparison motivating this separation is
[Jablonowski and Williamson (2006), *A baroclinic instability test case for
atmospheric model dynamical cores*, doi:10.1256/qj.06.12](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/qj_jablonowski_williamson_2006.pdf),
especially sections 2 and 4–5. That paper prescribes different jets,
temperature, terrain and perturbations, and numerical high-resolution
reference trajectories. **This log-pressure family is not that benchmark.**
The paper's multi-day growth and intermodel convergence results are not
reference data for these short runs; no such trajectories are fabricated.

### Physically scaled energy and axial torque

`model.step_energy_flux(evidence)` reports signed unclosed energy per sphere
area and attempted timestep in **W/m²**, after the separately measured filter
work. The qualification also reports cumulative residual divided by actual
elapsed time and area. These quantities supplement the owner's existing
relative-energy admission threshold; a small ratio to a huge reference energy
alone does not establish physically small heating or climate drift.

`model.angular_momentum(state)` is the atmosphere's absolute axial inventory:

```text
L = integral sum_layers[M a cosφ (u + Ω a cosφ)] dA
```

`model.budget_rates(state, held)` computes the unfiltered native RHS derivative
of L and compares it with independently defined mountain torque
`−integral[(p_s/g) ∂Φ_surface/∂longitude] dA`, plus prescribed zonal-force
torque and co-moving mass-source angular momentum. The constant-pressure lid
has zero integrated axial torque. With the default
`angular_momentum_projection="none"`, the raw residual remains unchanged.

`angular_momentum_projection="energy-neutral"` applies an explicit constrained
tangent correction. A represented solid-body zonal-acceleration basis cancels
the raw residual; a uniform thermal tangent cancels its instantaneous kinetic
power. Both responses are evaluated through the actual discrete inventories,
not assumed from continuum formulas. Evidence retains raw residual, applied
torque, corrected residual, thermal power, maximum acceleration, and correction
fraction. Excess correction or failed roundoff closure rejects atomically. This
is a measured structure-preserving numerical projection, not a physical torque
or permission to hide a poor discretization.

This is a shallow-sphere atmospheric inventory. The existing surface and
environment reservoirs do not supply a resolved velocity or moment of inertia,
so their momentum exchange is an **external atmospheric torque**, not a claimed
conserved atmosphere-plus-ocean angular momentum. Filters are absent from this
instantaneous RHS diagnostic. Native `GlobalStepEvidence.filter_angular_momentum`
measures postfilter minus prefilter angular momentum directly in kg m²/s; the
long-run tool accumulates **accepted-step** values and reports mean filter torque
in N m separately from physical torques. Neither these budgets nor a short stable trajectory establish
weather skill, climate skill or equilibrated climate statistics.

## Interactive atmosphere–wet-slab boundary

`GlobalAtmosphereProcesses(thermodynamics=thermo, surface_physics=GlobalSurfacePhysics(slab, exchange, radiation))` replaces the prescribed radiative/heat/water boundary with one explicit physical owner. `slab` is a `WetSlabPlan`, `exchange` a `BulkSurfaceExchangePlan`, and `radiation` a `ColumnRadiationPlan` with explicitly supplied `ColumnOpticalProperties`. The air and slab thermodynamic references must match. Combining this owner with Held–Suarez, Newtonian cooling, prescribed sensible heat, or prescribed evaporation is rejected rather than double-counting fluxes.

The **only** slab authority is still `GlobalAtmosphereState.surface_water` and `.surface_energy`, on the padded physical grid. `model.initialize(surface_temperature=290., surface_water=1000.)` initializes these values through the slab plan. Every stage derives

```text
T_surface = T_reference + E_surface / (C_dry + M_surface c_liquid).
```

There is no second slab state or independently advanced SST. The non-water capacity is J/m²/K, water is kg/m², and surface energy is J/m² in the same constant-caloric liquid reference as the atmosphere. Ice precipitation carries ice enthalpy: melting therefore draws real slab energy. Slab freezing is outside this liquid-only owner and rejects a proposal; it does not clamp SST or silently introduce sea ice. Negative donor water, invalid gas states, excessive saturation pressure/boiling, and invalid stage/candidate temperatures reject the entire native atmospheric step, preserving all inventories, held forcing and ledgers except the rejection counter.

The held TOA flux is genuinely latitude dependent:

```text
solar_down(latitude) = solar_constant / 4 × [1 + solar_p2 P2(sin(latitude))]
P2(x) = (3 x² − 1) / 2.
```

Its continuous spherical mean, and degree-two-exact native quadrature mean, are `solar_constant/4`. It is nonnegative for the admitted `solar_p2` range [−1, 2]. Defaults 1361 W/m² and −0.48 are **declared idealized experiment parameters**, not a fitted orbital insolation product. The radiation owner interprets this as diffuse hemispheric downward TOA flux, not beam-normal solar irradiance. There is no diurnal cycle, seasonal orbit, direct-beam zenith correction, or hidden conversion. Distribution and solar parameters participate in forcing/model identity. Current numeric optical scales, slab capacity and bulk coefficients are additionally checked at native host checkpoint I/O; they are never hashed as tracers in a compiled stage.

Radiation uses top-to-bottom layer masses and current vapor/liquid/ice inventories. Net-upward interface flux gives layer heating by bottom-minus-top divergence; the slab gains minus bottom net-upward flux and the environment gains top net-upward flux. Their column sum telescopes to zero before spatial projection. Longwave emission depends on **current** atmospheric and slab temperatures: greenhouse opacity and heat capacity changes feed back, rather than changing a prescribed thermal target.

Bulk exchange uses the lowest model cell as its shallow boundary-air approximation. Gas-only humidity is `q_v/(1−q_l−q_i)` and gas-only density excludes condensate loading. No unreported reconstruction makes a coarse lowest layer into a resolved surface layer. Resolved wind combines in quadrature with explicitly prescribed subgrid RMS ventilation (default 5 m/s); this is not a speed floor or a modeled natural-convection closure. Measurement height defaults to 10 m. Neutral or the bulk owner's declared Richardson closure may be used. Generic coefficient defaults are idealized transfer parameters, not evidence of calibration at global grid scale.

### Energy and water source derivation

Let `E` denote signed surface-to-air water mass flux, `H_donor` the bulk operator's **total** vapor donor-enthalpy flux, and `Q` sensible heat. `H_donor` already includes the thermodynamic latent reference: adding another latent term would double count it. The native pressure-coordinate source changes atmospheric mass, layer masses, pressure, compression and vertical redistribution. Its existing caloric contribution carries `E h_v(T_air)`. Thus the additional direct bottom-layer heating is

```text
M cp T_t|donor = Q + H_donor − E h_v(T_air).
M_t|water = E at the bottom; (M q_v)_t|water = E.
```

Co-moving injection puts that water at the bottom cell's geopotential and resolved horizontal velocity. It is not free mechanical energy:

```text
slab energy_t|upward exchange = −Q − H_donor − E (Phi_bottom + kinetic_bottom).
slab water_t|upward exchange = −E.
```

Positive evaporation takes donor vapor enthalpy at SST; negative evaporation/dew takes it at air temperature. Falling condensate returns liquid/ice enthalpy **plus** its originating geopotential and kinetic energy, and returns actual mass. These are explicit work terms, not residual repair. The native paired moving-coordinate pressure work and upper-lid work inventory remain authoritative. Mixing/drag momentum dissipation continues to enter the explicit environment reservoir; optional numerical filtering has independently measured work. Neither is concealed as adjusted SST.

The global owner still uses three water inventories with finite relaxation toward isobaric equilibrium and bulk instantaneous-fall precipitation. It does **not** become `InteractiveMoistColumnPlan`, whose finite-rate cloud/rain/snow species, fall speeds, rain evaporation and turbulence transport resolve richer column mechanisms. There is no resolved ocean transport, sea ice, deep convection, surface stress, precipitation fall trajectory, or observational climate skill assertion here.

## Native long-run circulation qualification

The experiment tool uses `PreparedGlobalAtmosphere.advance` inside compiled `jax.lax.scan`, sampled in bounded native chunks. It does not introduce a second simulation/training framework or universal state:

```sh
JAX_ENABLE_X64=1 python examples/global_atmosphere.py --interactive --steps 2

# Two steps exercise execution, never climate qualification.
JAX_ENABLE_X64=1 python tools/global_feedback_qualification.py \
  --smoke --scenarios baseline --bandlimit 3 --levels 2

# Joint phase projection: locality, exact total water and enthalpy evidence.
JAX_ENABLE_X64=1 python -m tools.global_water_projection_qualification

# 420 days, excluding 180 spinup days, leaving eight 30-day blocks.
JAX_ENABLE_X64=1 python tools/global_feedback_qualification.py \
  --days 420 --spinup-days 180 --sample-days 1 --block-days 30 \
  --scenarios baseline greenhouse solar surface_capacity \
  --sensitivity --checkpoint-dir /tmp/global-feedback-checkpoints \
  --output /tmp/global-feedback.json

# Continue one exact model after inspecting a prior segment. Statistics start
# afresh; prior samples are never spliced into the new confidence interval.
JAX_ENABLE_X64=1 python tools/global_feedback_qualification.py \
  --scenarios baseline --resume-checkpoint /tmp/global-feedback-checkpoints/baseline-L6-dt300-filter2.npz \
  --days 420 --spinup-days 60 --checkpoint-dir /tmp/global-feedback-checkpoints

# Or continue sequentially until the full gate passes or three segments finish.
# No "best" segment is selected.
JAX_ENABLE_X64=1 python tools/global_feedback_qualification.py \
  --scenarios baseline --days 420 --spinup-days 180 \
  --maximum-segments 3 --checkpoint-dir /tmp/global-feedback-checkpoints
```

The longer configuration is a **runnable qualification attempt, not precomputed successful equilibration**. Defaults are bandlimit 6, four pressure layers and 300 s timesteps. By default, the baseline first solves two bounded scalar controls—a uniform air-temperature offset retaining lapse rate/anomalies, and a uniform wet-slab-temperature offset—until actual global-mean TOA and surface flux residuals are within 0.1 W/m². This instantaneous preconditioner reports its Jacobian condition, iterations, offsets, energy change and before/after residuals. It does **not** zero local heating, balance horizontal dynamics, or establish stationarity. `--initialization raw` preserves the declared unmodified state.

Every intervention begins from the baseline-preconditioned physical temperature, pressure, wind, water and SST fields. The intervention's own flux residual is deliberately not preconditioned away; changed slab capacity receives the energy consistent with the shared SST. This prevents forcing information from leaking into separate scenario-specific initial states. `--resume-checkpoint` continues exactly one scenario as a new statistical segment. Its spinup clock is relative to the resumed state, counters remain cumulative, and prior samples are not silently reused. Segment intervention rates restart for reporting, but lifetime limiter mass and absolute projection impulse remain in the native ledger and gate every resumed segment; restart cannot launder earlier numerical intervention.

Optical coefficients in the tool are explicitly tagged synthetic: SW absorption `(1e−5, .002, .01, .01)`, SW scattering `(0, 0, .1, .1)`, LW absorption `(1e−4, .03, .1, .1)` m²/kg in dry/vapor/liquid/ice order. Baseline albedo is .25; dry slab capacity is 2×10⁷ J/m²/K plus capacity of 1000 kg/m² initial water. Greenhouse multiplies all LW scales by 1.2 (not a CO₂ doubling claim); solar multiplies irradiance by 1.02; surface-capacity doubles dry capacity.

Every result separates:

1. **Execution success:** requested native attempts accepted; rejection exits nonzero. Native checkpoint round-trip and subsequent continuation are compared bitwise. Repeated rejected states never count as stationary samples.
2. **Numerical accounting adequacy:** closed-energy drift and peak step drift in W/m² (default bound .1), total mass/water change, radiative telescoping residual, raw and corrected angular-momentum residuals, projection power/impulse, total-water redistribution, and phase-repartition burden. Corrected torque uses a 10⁻¹⁰ relative gate; raw/projected torque uses the declared 10⁻⁵ per-step fraction and 1% lifetime impulse gates. Radiation closure uses 10⁻⁹ W/m². Total-water redistribution is limited to 2×10⁻¹² per step and 10⁻⁸ over the checkpoint lifetime. Phase repartition is limited to 2×10⁻⁴ of atmospheric water per step and 10% of the actual integrated physical phase conversion over the midpoint construction and accepted update. Both raw phase mass and that process-relative ratio remain visible. These are transparent experiment gates, not universal accuracy constants. `numerical_accounting_adequacy` still does not imply small filter bias or certify a physical response.
3. **Sampling adequacy:** exclude spinup, discard incomplete final blocks, report nonoverlapping block means, Student 95% intervals, standard errors and adjacent-block correlation. At least eight complete blocks and absolute lag-one correlation below .3 are required. Insufficient blocks explicitly fail; no two-step climate label.
4. **Statistical stationarity:** first/second-half block means must agree within the declared two-halfwidth diagnostic on global SST, atmospheric temperature, EKE, precipitation and TOA balance. This may fail independently of numerical adequacy.
5. **Stationary sample qualification:** absolute mean TOA and slab net flux plus uncertainty must each be ≤1 W/m², production spans at least 180 days, and accounting/sampling/stationarity gates pass. This produces `stationary_sample_qualified`, not an unqualified climate/physical-response label.
6. **Deterministic near-fixed equilibrium:** a separately named gate covers low-variability trajectories for which independent samples do not exist. It still requires numerical and radiative-equilibrium gates, at least 180 production days, endpoint TOA/surface flux within .1 W/m², and half-record changes within .2 K for air/SST, .5 W/m² for TOA/surface flux, .01 m²/s² for EKE, and 10⁻⁷ kg/m²/s for precipitation. High block correlation remains reported and `stationary_sample_qualified` remains false. This gate supplies no confidence interval, variability, climate, or response claim.
7. **Empirical physical-response support:** a response must exclude zero statistically **and** all three actual paired sensitivities must retain qualified stationary samples and a stable nonzero response. Deterministic-equilibrium status cannot substitute. Without sensitivity evidence, `response_claim_supported` is false even when `statistically_resolved_difference` is true. For each variant, `abs(response_variant − response_reference) + combined_95%_halfwidth` must fit both the dimensional tolerance and the declared fraction of the reference response's uncertainty-adjusted lower magnitude. Wide uncertainty therefore cannot excuse unresolved numerical sensitivity.

Deterministic thresholds are configurable through the `--deterministic-*` options; they are fixed before execution and emitted with the evidence. Sequential campaigns stop on either the full stationary-sample gate or this distinct near-fixed-equilibrium gate. They never select a favorable segment retrospectively.

Default dimensional response tolerances are .1 K for SST/air temperature, .5 m²/s² for EKE, 10⁻⁷ kg/m²/s for precipitation, and .1 W/m² for TOA balance. They are configurable with `--response-temperature-tolerance-k`, `--response-eke-tolerance-m2-s2`, `--response-precipitation-tolerance-kg-m2-s`, and `--response-toa-tolerance-w-m2`. The simultaneous signal-relative limit defaults to .25 via `--maximum-response-sensitivity-fraction`. These are declared experiment tolerances, not calibrated universal scientific thresholds.

The physical-response gate also budgets filter work independently. For accepted **post-spinup** steps it records signed filter work and the sum of absolute kinetic and remaining thermal filter work before time averaging; signed or channel cancellation cannot erase this budget. Both members of the reference and each sensitivity pair must fit `min(--maximum-filter-work-w-m2, --target-signal-w-m2 × --maximum-filter-signal-fraction)`, and their signed filter-work contrast must fit it too. Defaults are .1 W/m², a declared 1 W/m² energetic target, and .1 respectively. This target is explicit—not a radiative signal inferred from a temperature response. Filter-work checks remain empirical safeguards and are **not** rigorous bias bounds; paired filter-rate response sensitivity is still required.

Resolved diagnostics include zonal winds and hemisphere jet maxima, EKE and eddy momentum flux, mean/eddy meridional moist-enthalpy and water transports, mass overturning, temperatures/SST, precipitation/evaporation, donor enthalpy and mechanical work, TOA/surface radiation, rain energy return, atmospheric angular momentum and physical torques. Spherical kinetic-energy spectra use native vorticity/divergence coefficients and inverse-Laplacian eigenvalues; they are per-layer **specific** kinetic energy, not mass-weighted spectra or evidence of an inertial range. Transports are longitude-circumference integrals. Eddies are departures from instantaneous zonal means, not a separately qualified transient/stationary decomposition.

Daily point samples can alias weather; adjacent-block checks are necessary but insufficient to prove independence. A liquid-only polar slab may freeze, phase repartition can exceed its explicit per-step or lifetime bounds, spinup can be insufficient, and block correlation can stay large. These are measured failed gates—not invitations to move total water, reset lifetime ledgers, hide phase adjustment, repair budgets, relax tolerances silently, or label synthetic evidence real-Earth climate skill.
