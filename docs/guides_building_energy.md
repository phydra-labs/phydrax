# Building energy: explicit reduced physics and reference comparisons

`phydrax.applications.building_energy` composes immutable building descriptions
with native dynamics, linear algebra, optimization, series, and uncertainty
quantification. It is not an EnergyPlus or Radiance reimplementation. External
execution is a host-only evidence-producing operation; imported numerical
operators may subsequently participate in native differentiable calculations.

## Physical source and explicit RC reduction

All native thermal arrays use seconds, Kelvin, watts, joules, and metres.
Heat inputs are **positive into the building node**. Each `Adjacency` reports
positive heat from `left` to `right`; `right=None` uses an explicitly named
environmental `boundary_id` (the single-boundary default is `"outdoor"`).
A single interzone surface belongs to both zones through its explicit adjacency;
do not represent it a second time from the other side.

```python
import jax.numpy as jnp
from phydrax.applications.building_energy import (
    Adjacency, BuildingSource, Zone, compile_building,
)

source = BuildingSource(
    (Zone("room", capacity=10000.0, volume=60.0),),
    adjacencies=(Adjacency("envelope", "room", None, 10.0),),
    source_id="measured-room",
    provenance=("authored-reduced-model",),
)
model = compile_building(source)
step = model.step(jnp.array([300.0]), 280.0, jnp.array([50.0]), 500.0)
# Exact frozen result: 285 + 15 exp(-0.5) K.
```

`Zone.capacity` is an explicit effective J/K capacity; volume does not silently
create air capacitance. Physical parameters remain array leaves and can be
optimized or differentiated. Names, connectivity, and RC order are static.

`BuildingBoundary(boundary_id, kind=...)` declares ordered `ambient`, `ground`,
or `fixed` temperature inputs. Surfaces explicitly select an environmental
`boundary_id`, an `adjacent_zone`, or `adiabatic=True`; these roles are mutually
exclusive. Ground temperature is never silently taken from outdoor air.
Adiabatic surfaces remain in the source/geometry: a massless one contributes no
flux, while a massive one retains its wall capacity and interior connection but
has no exterior heat connection. `VentilationExchange` declares ventilation or
infiltration sensible conductance in W/K and its supply-temperature boundary.
Conductance is an explicit differentiable mass-flow-times-heat-capacity value;
there is no hidden air-change-rate conversion or moisture model.

A `Construction` declares resistance in m² K/W and areal capacity in J/(m² K):

- `massive=False`, zero areal capacity: direct opaque-area/resistance conductance.
- `massive=True`, positive areal capacity: one wall-centre state with two
  half-resistances. Capacity is opaque area times areal capacity.
- Surface area is **gross** area. Apertures subtract from opaque area, have their
  own W/(m² K) U-value, and cannot exceed the parent area.
- Film resistances, effective air/material capacity, and the reduction order
  are caller assumptions. There is no material-name-based inference.
- Aperture solar transmittance is source metadata; solar gains must be computed
  from an explicit radiative/solar model and passed as nodal heat. It is not
  silently equated with normal-incidence transmittance at every angle.

Positive-capacity models contain a native `ContinuousSystem`. Declared
`Zone(..., capacity=0, massless=True)` nodes instead produce a native
`DifferentialAlgebraicSystem` with zero-capacity energy equations. Unanchored
massless components are rejected. `consistent_temperature` solves the algebraic
balance; initial guesses at massless nodes are not stored-energy states.
`step` eliminates the index-one algebraic block and calls
`dynamics.affine_exponential_step`, then lifts temperatures back to all nodes.
This is exact for **frozen** forcing and coefficients, not a weather integration
accuracy claim. Every step retains native matrix-function success and residual
evidence. Algebraic linear-solve failures are errors, not fallback temperatures.

The native Krylov norm uses a zero subgradient for an inactive zero residual at
exact happy breakdown. Its square-root argument is guarded before evaluation,
so discarded residuals cannot inject `0/0` cotangents into an otherwise smooth
affine parameter derivative. Positive norms are unchanged, and nonfinite inputs
remain nonfinite. The scalar RC log-capacity sensitivity is checked against
both its analytic exponential derivative and forward AD. This is not a blanket
gradient guarantee across changing Krylov ranks or a degenerate zero starting
vector; those require their own matrix-function differentiation analysis.

`node_ids` orders zones first, then declared massive-wall states. External
heat arrays contain one entry for every compiled node. `boundary_ids` orders
environmental temperature vectors; scalar temperatures are accepted only when
exactly one boundary exists. Native system input layout is all ordered boundary
temperatures followed by all nodal heat values. `boundary_conductance[node,
boundary]` and `edge_boundary[edge, boundary]` retain the explicit mapping.
`observe` reports directed edge heat, net nodal heat, stored energy relative to
an explicit reference temperature, temperature-bound comfort violations, and
`C * temperature_rate - net_heat`. Supply a measured or independently computed
rate when using balance residuals as a diagnostic. With the default model-derived
rate, the residual checks algebraic consistency, not model adequacy. Comfort is
a temperature-band metric, **not** a claim of PMV/PPD or regulatory compliance.

## Geometry and archetype enrichment

Geometry is optional. Numerical `Surface` areas can be provided directly.
`enrich_building_geometry` consumes the existing `SurfaceModel`, its
`SpatialCoordinateContract`, and explicit `SurfaceRole` mappings from
revision-bound `MeshLabel`s. `surface_tag_labels` converts existing cell tags
to labels without guessing which tag is a wall, roof, window, or thermal zone.

The geometry route:

1. Requires physical Cartesian coordinates and matching coordinate contracts.
2. Converts coordinate length to SI before deriving triangle areas.
3. Rejects degenerate triangles, stale mesh revisions, and overlapping gross
   surface labels.
4. Requires aperture labels to be disjoint subsets of their parent gross label.
5. Binds each enriched surface to model, numerical geometry, revision, spatial
   contract, and selected entities. No volume mesh is required.

`BuildingArchetype` contains caller-provided role/construction assumptions,
source URL, license identity, and assumption provenance. No licensed third-party
archetype dataset is bundled. `archetype.enrich(source, surface_roles, source_id=...)`
performs an explicit source-level replacement.
`retrofit_building` returns a newly identified source with replaced constructions
or zones while leaving the baseline untouched. Recompile the result; never patch
a compiled conductance matrix and continue to label it the original building.

## EPW: calendar, interval meaning, and missingness

`read_epw(path)` and `parse_epw(text)` parse all eight EPW headers and all standard
35-field records. The artifact retains location, source station, standard UTC
offset, complete original headers, record calendar/year, uncertainty strings,
per-quantity missing masks, leap-year declaration, records per hour, and a content
SHA-256. A single continuous declared data period is supported. Nonuniform,
missing, duplicate, or out-of-order records are rejected; no records are fabricated.

- The EPW clock is local **standard time**, not daylight-saving civil time.
- Hour 1/minute 60 ends at 01:00; hour 24/minute 60 ends at next midnight.
- Subhourly records must respect the declared records-per-hour count.
- Dry-bulb and dew-point samples are Kelvin observations at interval endings.
- Radiation is preceding-interval Wh/m² and becomes J/m² by multiplying by 3600.
- Missing sentinels become invalid components, not zero observations. Numerical
  zero is storage under an invalid mask and must not be used as imputed weather.
- TMY source labels or nonchronologically stitched source years identify a typical
  year; an ordinary December/January year rollover does not. Callers can
  explicitly override `typical_year`. A representative 2001 calendar, or 2000
  when February 29 is present, is chosen unless `calendar_year` is supplied.
  Original source years remain in `record_calendar`. Leap-day remapping into a
  non-leap year is rejected.

Each selected weather quantity is a shared `EnergySeries`. TMY clocks use
`time_basis="cyclic"`; observed calendars use `"absolute"`. Both preserve an
explicit offset-aware origin and fixed-standard-time timezone. Temperatures use
node-aligned `instantaneous` samples, solar energy uses edge-aligned
`interval_integral` samples. Use the shared conservative `rebin_energy_series`
for energy intervals. Do not treat instantaneous temperature observations as
interval means without explicitly selecting a reconstruction/holding policy.

## Native HVAC control, replay, identification, and UQ

`replay_building` accepts interval-held boundary temperatures and nodal heat and
returns node temperatures at all time boundaries with numerical success evidence.
Its environmental input shape is `(interval, boundary)`; `(interval,)` is the
single-boundary convenience.
`optimize_hvac` lowers constant-COP and resistance devices into native
`control.LinearQuadraticControlProblem` and `solve_linear_quadratic_control`.
Exact frozen RC transitions, the actual `TimeGrid`, endpoint tracking costs
(including the true terminal state), irregular interval durations, and hard
normalized-power bounds are native control structures, not a second MPC compiler.
Only nonlinear device laws use direct shooting with `optim.ProjectedLBFGS`.
The result explicitly labels `mode="linear-quadratic"` or `"nonlinear-shooting"`.

Controls are bounded **electrical** powers in W. The
`thermofluids.HeatConversionLaw` converts power and source/supply temperatures
into delivered heat. The conversion's source-temperature boundary is explicit
through `source_boundary_id`, or the unique declared ambient boundary.
`heat_distribution[node, device]` must be nonnegative with columns summing to
one. Unsupported operating points fail rather than clip.

The objective integrates squared temperature tracking error and optional
price-weighted electricity use. Electricity prices are per joule. Power bounds
are hard; temperature tracking is a weighted objective, not certified hard
comfort feasibility. Normalize controls with `power_scale` for numerical
conditioning. For receding-horizon operation, apply the first optimized row,
observe the building, and optimize again. For affine devices, `optimization` is
the native `LinearControlQPSolution`, including `ControlTrajectory`, compiled QP,
policy and solver evidence. Native trajectory states are temperature deviations
from `state_reference_temperature`; controls are normalized by `power_scale`.
An independent physical-temperature `replay` is compared with the decoded
native trajectory before success is reported. Nonlinear mode retains the native
minimization result and its physical replay.

Use `thermofluids.HeatPortBridge` at coupling boundaries: Kelvin/W with inward
heat is the building convention. For a component reporting positive outward
heat, explicitly select `OUT_OF_COMPONENT`; Celsius requires the explicit
273.15 offset. Do not negate heat port quantities based on undocumented names.

`calibrate_building(make_source, initial_parameters, training, heldout, ...)`
uses native nonlinear least squares with bounded dense-QR Jacobian solves. The
finite parameter Jacobian is also required for identifiability evidence;
QR avoids nesting an iterative Krylov solve inside thermal automatic differentiation.
`make_source` owns admissible physical
parameterization (for example logarithms of positive capacities). Observation
nodes/scales are explicit. Calibration reports all local sensitivity singular
values, a rank-based identifiability decision, held-out predictions and RMSE,
and native optimizer evidence. Training and held-out experiment identities must
differ. Local full rank is not a guarantee of global identifiability; rank-deficient
fits are not reported as successful identification.

`calibrate_building_prediction_band` returns a native `uq.FunctionalConformal`
model over **exchangeable held-out episodes**, with episode/time/node axes.
Its simultaneous bands are not justified by treating consecutive correlated
samples from one trajectory as independent calibration cases.

## Radiative artifacts and bounded external producers

A `RadiativeBasis` binds ordered sample labels, RGB or caller-declared channels,
measure, measure unit, positive weights, and identity. `RadiativeOperator` maps
source coefficients to target coefficients with explicit input/output units.
Coefficients already include quadrature weights; `from_kernel` is the explicit
route that multiplies a kernel by source weights. `RadiativeComposition` applies
execution-order factors without materializing their dense product. Intermediate
basis, measure, and unit identities must match exactly.

`import_radiance_matrix` consumes real Radiance/Frads ASCII matrix output with
`NROWS`, `NCOLS`, `NCOMP`, and `FORMAT=ascii` headers and validates every payload
value. It does not infer bases or turn RGB into watts, lux, or irradiance.
Angular integration is an explicit numerical factor. `radiative_heat_gains`
converts declared irradiance-band responses into inward nodal W using explicit
spectral weights, SI receiving areas, absorption fractions, and a conservative
receiver-to-node distribution. There is no implicit RGB-to-thermal conversion.
`produce_radiance_matrix` runs a pinned executable through bounded
`interchange.energy_runtime` transport, then imports the actual output.
Provide exact arguments, scene/input bytes, output location, timeout, and
`RAYPATH` where needed. This is also the route for an installed Frads CLI that
emits a compatible matrix; no foreign implementation is copied.

`produce_uniform_sky_reference` runs actual `oconv` and `rtrace` for a unit-radiance
upper hemisphere and an upward irradiance sensor. The analytical per-channel
irradiance is π. Actual measured coefficients and both run records are returned.
The benchmark declares an absolute 0.05 tolerance in advance.

## EnergyPlus references and qualification

`EnergyPlusReference` carries exact IDF/epJSON bytes and provenance. Its `run`
method uses the shared pinned isolated runtime, requires real executable success,
and retains `.err` diagnostics. `parse_energyplus_csv` imports **exactly named**
variables with caller-declared reporting interval, units, quantity meaning,
calendar, and Celsius conversion offset. Missing cells stay invalid. Mixed
frequencies or missing timestamps fail rather than get relabelled.
`compare_energyplus_reference` requires matching physical quantity, meaning,
origin, timezone, and aligned coordinates; all samples must be valid for a pass.
Unit conversion is explicit and tolerances are supplied before comparison.

`energyplus_adiabatic_reference` is an authored closed 60 m³ single zone with six
adiabatic massless surfaces, 100 W all-convective gains, no outdoor-air exchange,
and ideal loads holding 293.15 K. Its native steady counterpart has G=0 and
100 W internal heat balanced by -100 W HVAC heat. The supplied synthetic EPW is
explicitly **not measured weather**; this particular reference is weather
independent. Qualification tolerances are 0.05 K temperature and 1 W sensible
cooling. This demonstrates matching signs, units, and steady energy balance,
not general transient equivalence between the solvers.

Run the authored benchmark tool from the worktree environment:

```sh
python tools/building_energy_benchmarks.py
python tools/building_energy_benchmarks.py --energyplus /absolute/path/energyplus \
  --oconv /absolute/path/oconv --rtrace /absolute/path/rtrace --raypath /absolute/path/radiance/lib
```

Native cases cover one/two-zone analytic flows, a massless DAE equivalent,
finite-horizon control, and identifiable calibration with held-out prediction.
Optional external cases run only when executable paths are supplied; absence
is not reported as external success. The tool reports observed timings and
physical errors without a benchmark schema version. No gradient is claimed
through EnergyPlus, Radiance, Frads, subprocess execution, or parsing.
