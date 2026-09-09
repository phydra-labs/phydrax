# Interactive finite-rate moist column

`InteractiveMoistColumnPlan` is a different physical owner from the equilibrium
`MoistColumnPlan`. The equilibrium API remains unchanged. The interactive model
retains cloud and falling hydrometeors, diagnoses temperature at their **current
composition**, and couples one finite wet slab to actual radiative interface
fluxes and bulk sensible/water exchange.

## Scope and ownership

The column has unit horizontal area and fixed cell volumes, ordered **top to
bottom**; the surface is index `-1`. Thus a layer volume in m³/m² is also its
thickness in metres. State inventories are:

- `dry_mass`, `vapor_mass`, `cloud_liquid_mass`, `cloud_ice_mass`, `rain_mass`,
  and `snow_mass`, each in kg/m²;
- `internal_energy`, J/m², and immutable `layer_volume`, m³/m²;
- `slab: WetSlabState(water_mass, energy)`, the sole lower water, energy and
  surface-temperature owner;
- signed `environment_energy` and cumulative `external_energy`, J/m²;
- cumulative precipitation mass/energy, signed surface evaporation, diagnostic
  terminal-fall potential-energy loss and temperature-variance destruction;
- accepted `time` and `step_count`.

There is no duplicate prescribed sea-surface temperature, rain reservoir,
held forcing, hidden equilibrium projection, or private integration clock.
`plan.slab.temperature(state.slab, plan.thermodynamics)` derives surface
 temperature from the slab's total energy and liquid-water caloric capacity.

This is a **forced fixed-volume column**, not a compressible, anelastic, or
hydrostatic dynamical column. Pressure is diagnosed from the gas masses,
volume and temperature. It need not satisfy hydrostatic balance after heating
or mixing. Horizontal momentum, resolved vertical velocity, pressure adjustment,
kinetic energy and gravitational potential energy are not prognostic. Specified
ventilation/shear represent imposed unresolved stirring. The conservative
energy statement below is a caloric control-volume statement, not conservation
of a dynamically closed atmosphere's total mechanical energy.

## Construction and numerical parameters

```python
import jax.numpy as jnp
from phydrax.applications.atmosphere import (
    BulkSurfaceExchangePlan,
    ColumnOpticalProperties,
    ColumnRadiationPlan,
    InteractiveMoistColumnPlan,
)

# Illustrative grey coefficients, not measured/calibrated optical data.
optics = ColumnOpticalProperties(
    shortwave_absorption=(1e-5, 0.002, 0.04, 0.03),
    shortwave_scattering=(0.0, 0.0, 60.0, 30.0),
    longwave_absorption=(1e-4, 0.08, 50.0, 25.0),
    shortwave_asymmetry=(0.0, 0.0, 0.85, 0.7),
    reference_id="illustrative-grey-column",
)
plan = InteractiveMoistColumnPlan(
    radiation=ColumnRadiationPlan(optics),
    surface_exchange=BulkSurfaceExchangePlan(),
    background_diffusivity=0.2,
    mixing_length=50.0,
)
state = plan.initialize(
    dry_mass=jnp.asarray([100.0, 110.0]),
    vapor_mass=jnp.asarray([0.2, 0.4]),
    temperature=jnp.asarray([285.0, 290.0]),
    layer_volume=100.0,
    surface_temperature=295.0,
    surface_water_mass=1000.0,
)
result = plan.step(state, 1.0, solar_down=340.0, wind_speed=5.0,
                   ventilation=0.05, shear=0.01)
```

Process timescales (seconds), cloud threshold (kg water/kg dry), terminal fall
speeds (m/s), diffusivity (m²/s), mixing length (m), and critical Richardson
number are numerical scalar JAX leaves. Optical scale arrays, slab heat capacity
and bulk exchange coefficients likewise remain numerical interventions; they
are not hidden in Python control flow. `thermodynamics` retains the existing
native caloric constants and domain.

The initialized atmospheric arrays declare the numerical state dtype. Slab
initialization/diagnosis, numerical physics leaves and exchanged rates are bound
to that same dtype through differentiable casts. Explicit float32 inventories
therefore remain float32 through multi-step/native scans even with global x64
enabled and float64 optical or bulk coefficients; float64 inventories remain
float64. This does not change global JAX precision, detach parameter derivatives,
or change the original numeric parameters recorded in checkpoints.

Radiation and bulk exchange are disabled by `radiation=None` and
`surface_exchange=None`, respectively. Disabling radiation does not inject
Newtonian relaxation or silently redirect heating to another reservoir.

## Current-composition thermodynamics

For each cell the native phase energies obey

\[
 U = m_d e_d(T)+m_v e_v(T)+(m_c+m_r)e_l(T)+(m_i+m_s)e_i(T).
\]

They are affine in temperature. The diagnosis inverts this equation with the
**current** six species inventories and its exact combined heat capacity.
Falling rain and snow contribute their liquid/ice calorics but are never
repartitioned by `thermodynamics.adjust`. Consequently supersaturation,
undersaturated rain and supercooled liquid are possible transient states.
Native liquid/ice Clausius–Clapeyron pressures determine kinetic source rates.

## Finite-rate microphysics

One accepted first-order split step performs:

1. Conservative balanced-volume mixing and any supplied conservative interior
   fluxes, radiative divergence, explicit prescribed heating and surface rates.
2. Thermal cloud/rain/snow freezing or melting. Conversion is limited by both
   its specified timescale and the latent-energy amount required to reach the
   reference freezing temperature. Existing internal energy is retained.
3. Finite-rate condensation of supersaturated vapor into the thermally stable
   cloud phase. Its source is the excess vapor mass divided by
   `condensation_timescale`; it is not an algebraic saturation adjustment.
4. Cloud evaporation followed by below-cloud rain evaporation/snow sublimation,
   using each phase's own native saturation law. Available donor mass bounds
   each transfer. Cloud vapor-deficit relaxation uses the condensation timescale;
   falling hydrometeors use `rain_evaporation_timescale`.
5. Autoconversion of cloud mass above `cloud_threshold * dry_mass` into rain/snow
   on `autoconversion_timescale`, preserving liquid/ice identity and energy.
6. Adjacent-layer upwind sedimentation with constant specified rain/snow fall
   speeds, carrying the donor phase enthalpy. Only the bottom flux reaches the
   wet slab. Arriving snow enters a liquid slab with its ice enthalpy retained;
   a slab temperature below its liquid-model domain rejects the transaction.

This is an explicit bulk mass model, not a droplet-size distribution, ice
nucleation model, collision/coalescence kernel, or calibrated precipitation
scheme. Effective timescales must be justified for the experiment. It does not
remove every cloud condensate instantaneously, use equilibrium humidity as
precipitation, or teleport an upper cloud directly into the surface.

For a layer thickness Δz, sedimentation requires `dt * fall_speed / Δz <= 1`.
A rain pulse moves at most one cell per step. At unit Courant number it traverses
three equal cells in exactly three steps; at smaller Courant number the
first-order upwind scheme has the expected dispersive arrival distribution.
Refinement must compare equal physical elapsed time, not equal step counts.

Latent heating increases the stiffness of vapor relaxation. With saturated vapor
mass `m_sat(T)` and cell heat capacity `C`, the source bound includes

\[
 \Delta t \le \frac{\tau}{2\,[1+\max(0,m'_{sat}(T)\,\Delta e/C)]}.
\]

This bound applies on active condensation/evaporation branches, using native
latent energies and saturation derivatives. Phase conversion and autoconversion
also require their explicit source timescales. There is no hidden substepping,
unconditional clipping, or automatic change of the physical timestep.

## Stability-aware interior transport

Virtual potential temperature includes vapor buoyancy and condensate loading.
Adjacent values diagnose a bulk `N²`; positive `N²` is stable. Prescribed
`ventilation` (nonnegative m/s) and `shear` (s⁻¹) broadcast to the `n-1` interior
interfaces. A mixing-length closure combines their stirring frequency with
unstable buoyancy, and suppresses the added diffusivity under stable Richardson
number. `background_diffusivity` remains an explicit independently imposed
floor; setting it and `mixing_length` to zero disables mixing.

The diffusivity divided by adjacent-centre distance gives two equal opposite
parcel-volume exchanges. These exchange **every species**, including dry mass,
and donor total enthalpy; their paired cell increments telescope exactly. The
species donor-volume CFL is strengthened by the largest gas `cp/cv` ratio.
The step additionally checks actual intermediate/final composition and caloric
temperature instead of assuming a CFL alone guarantees admissibility.

`mixing_temperature_variance_dissipation` reports
`2 * sum(K / distance * (T_upper-T_lower)**2)`, in K² m/s. Its cumulative state
ledger is in K² m. This is a closure variance-destruction diagnostic, **not** a
measured entropy-production rate or an energy source. Likewise
`fall_potential_power` reports the gravitational energy implied by terminal
mass transfer over cell-centre distances (a half-cell distance at the surface).
That mechanical energy is explicitly outside the modeled caloric budget. It is
not silently reintroduced as drag heat or claimed to close mechanical energy.

Optional `water_flux` and `energy_flux` use the same closed interior divergence.
They broadcast to `n-1`, are positive downward, and have units kg/(m² s) and
W/m². Water enters vapor; the supplied energy flux is **already total transported
energy**, so no additional latent enthalpy is appended. Gross vapor donor
availability and thermodynamic domain checks apply. A learned closure can use
this interface without another state owner or integration framework.

## Surface and radiation bookkeeping

The bulk boundary receives **gas-only** density and specific humidity, excluding
suspended/falling condensate. Positive surface rates transfer slab-to-air water,
sensible heat and total water enthalpy; negative rates reverse the exchange
with the native donor convention. Requested evaporation cannot borrow from rain
that arrives later in the same split step: exhaustion of the initial slab water
rejects the whole transaction.

Radiative heating uses the actual interface flux divergence. Cloud liquid and
ice contribute cloud optical properties; rain and snow are passed separately
under the radiation model's explicit transparent-precipitation approximation,
not miscounted as dry gas or small cloud droplets. For net upward interface
flux `N = upward - downward`, top to bottom,

\[
 H_k=N_{k+1}-N_k,\qquad H_{slab}=-N_n,\qquad H_{environment}=N_0.
\]

Thus the signed environmental reservoir includes incoming solar as well as
escaped longwave. There is no duplicate external-solar ledger. For every
accepted step:

\[
 \Delta\left[\sum_k m_{water,k}+m_{slab}\right]=0,
\]

\[
 \Delta\left[\sum_k U_k+E_{slab}+E_{environment}\right]
   = \Delta E_{external},
\]

where `external_energy` accumulates only prescribed `heating_rate`. Cumulative
precipitation and evaporation are diagnostic integrals of the **same accepted
fluxes**, not extra reservoirs counted in total water or energy.

`InteractiveColumnStepResult` exposes accepted surface/radiative/precipitation
rates and increments, `stable_step`, residuals, acceptance and derivative
validity. Rejected attempted residuals remain visible, but all accepted-flux
fields are zero and the entire continuation—including slab, ledgers and
clock—is unchanged.

## Native execution, restart and differentiation

`advance(state, dt, steps, **forcing)` is the existing style of fixed-step scan;
a rejection freezes subsequent continuation. For native retention/replay and
lifecycle composition, use the method adapter rather than inventing a new
column-specific driver:

```python
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver import FixedStepProblem, FixedStepRolloutPlan

problem = FixedStepProblem(
    plan.fixed_step_method(), state,
    t0=float(state.time), t1=float(state.time) + 20.0, step_size=1.0,
    args={"solar_down": 340.0, "wind_speed": 5.0},
    state_geometry=EuclideanStateGeometry(),
)
rollout = FixedStepRolloutPlan(retention="final").rollout(problem)
```

`save_checkpoint(path, state)` and `load_checkpoint(path)` use the native bounded,
checksummed array archive. They preserve all numeric inventories and accepted
clock, bind structural physics identities and compare the exact current numeric
parameter leaves. Changing a coefficient through an Equinox tree intervention
cannot silently load a checkpoint of another numeric model. For deliberate
physical interventions, load with the original plan, then evaluate its state
with a structurally compatible numerically varied plan. Forcing is supplied
explicitly at each continuation call and is not hidden checkpoint memory.

Diagnosis validity only certifies a state. **A trajectory derivative requires
all accepted step `derivative_valid` flags**, not just a regular endpoint.
These flags account for thermal phase boundaries, saturation onset, active
availability minima, autoconversion thresholds, active stability and mixing-length
floor switches, surface donor/Richardson/wind branches, donor-CFL boundaries and
slab exhaustion. Atmospheric and slab temperatures must lie strictly inside
their admissible domains, with precision-scaled margins, throughout the
initial, transport, microphysics and final states used in the certificate.
Ordinary physical admission still includes exact temperature bounds; those
states are accepted when otherwise valid, but cannot certify derivatives
through an accept/rollback boundary.
Inactive zero hydrometeors do not automatically invalidate a smooth dry or
subsaturated neighborhood. No differentiability claim is made across a
rejection or a flagged switch, nor does a short regular derivative certify a
long chaotic or branch-changing rollout.

## Runnable measurements

```sh
PYTHONPATH=. python examples/interactive_moist_column.py --steps 1800 --dt 2
PYTHONPATH=. python examples/interactive_moist_column.py --spinup-steps 43200 --steps 1800 --dt 2 --restart column.phx
PYTHONPATH=. python tools/interactive_moist_column_qualification.py --steps 600 --dt 2
PYTHONPATH=. python tools/interactive_moist_column_qualification.py --spinup-steps 43200 --steps 1800 --dt 2 --solar-change 40 --wind-factor 1.5
```

The example reports physical temperature, pressure, relative humidity, cloud,
falling precipitation, slab water/energy, integrated mm/day water fluxes,
interface radiation and all material/radiative budgets. The qualification adds
same-duration `dt`, `dt/2`, `dt/4` measurements; reproducible solar and wind
interventions from an identical spun-up state; exact restart; the native
fixed-step adapter; rain time-of-flight and evaporative cooling; and a short
observable derivative against an epsilon sweep with actual process regularity.
Its coefficients are explicitly illustrative. Successful execution or longer
spinup is not evidence of equilibrium, observational calibration, parameter
identifiability, or realistic atmospheric circulation. Inspect the measured
rates and drift before making any of those scientific claims.
