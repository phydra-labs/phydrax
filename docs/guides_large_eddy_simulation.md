# Large-eddy simulation

Phydrax treats large-eddy simulation (LES) as a composition of an equation convention,
a resolved-filter identity, parameter provenance, a prepared closure, a numerical
realization, temporal execution, diagnostics, and route-exact qualification. The
[LES equations API](api/equations/les.md) is normative for formulas, signs, trace
handling, filters, identity, Favre/KSGS contracts, and automatic differentiation. This
guide explains how the implemented pieces compose without turning an implemented route
into a released claim.

## Status vocabulary

Use three different statements:

- **Implemented** means the public plan prepares or evaluates the documented route.
- **Candidate** means measured evidence may be assembled for an exact support tuple,
  but the profile remains unreleased.
- **Released** requires the generic signed release boundary for that exact profile and
  all dependencies. No LES profile in this guide should be inferred to be released
  merely because its implementation, documentation, campaign producer, or evidence
  artifact exists.

A candidate LES profile also depends on the corresponding base incompressible-flow
profile; that base profile is an external release dependency, not evidence the LES
campaign can manufacture or waive.

## Choose the numerical owner first

| Objective | Public owner | Implemented boundary |
| --- | --- | --- |
| Static algebraic periodic LES | `compile_periodic_incompressible_flow(..., algebraic_les=...)` | Single-device 3-D full-complex Fourier |
| Dynamic periodic LES | `compile_periodic_incompressible_flow(..., dynamic_les=..., dynamic_test_discretization=...)` | Compiled 3-D Fourier plus transactional ETDRK production |
| Distributed periodic LES | `DistributedPeriodicLESProductionPlan` | Full rotational slab/pencil flow with device-resident ETDRK/SSPRK production; qualification not inherited |
| Wall-resolved spectral channel | `compile_channel_les` | Enforced complete SBDF2 restriction; optional normal-essential/equilibrium-traction owner |
| Static algebraic MAC LES | `compile_mac_incompressible_flow(..., algebraic_les=...)` | 3-D periodic/free-slip/symmetry boundaries; frozen implicit methods |
| Dynamic MAC LES | `compile_mac_incompressible_flow(..., dynamic_les=...)` | Periodic uniform 3-D plus projected explicit method |
| Scalar/Boussinesq/ocean KSGS | `compile_mac_scalar_buoyancy` or `CartesianBoussinesqOceanPlan` | Static/buoyant; dynamic on periodic uniform; low-Re at true no-slip walls |
| Learned stress backends | `PeriodicLearnedStressPlan` / `MACLearnedStressPlan` | Bound 3-D unit-density Fourier or periodic-uniform MAC divergence/projection |
| Conservative unstructured low-Mach LES | `UnstructuredLowMachLESFixedStepMethod` | Fixed tetrahedral transport, pressure projection, restart continuation |
| Compressible Favre transport | `HomogeneousMixtureCompressibleNavierStokesSystem(..., favre_les=...)` | 3-D neglected trace or appended transported SGS energy |
| Athermal LBM Smagorinsky | `SmagorinskyCollisionPlan` | Collision-local SRT closure at unit lattice filter width |

These routes share physics contracts but never numerical or qualification evidence.
Static algebraic and dynamic compiler inputs are mutually exclusive, and learned,
distributed, channel, immersed, and unstructured owners retain their own exact APIs.

## Identity-first algebraic setup

Every deployment starts with filter semantics and coefficient provenance. This
periodic example uses the exact retained Fourier projection and a user-supplied static
Smagorinsky coefficient:

```python
import phydrax as phx

resolved_filter = phx.equations.ResolvedLESFilter(
    "retained-fourier-grid",
    family="sharp-fourier-projection",
    axis_names=("x", "y", "z"),
    topology="tensor-product",
    boundary_class="periodic",
    scale_rule="cutoff-equivalent",
    commutation_status="commuting",
    repeated_filter_semantics="idempotent",
)
provenance = phx.equations.LESParameterProvenance(
    resolved_filter,
    spectral_space.prepared_id,
    "three-dimensional-periodic-unit-density",
    source_kind="user",
    evidence_ids=(),
)
prepared_model = phx.equations.SmagorinskyLESPlan(0.16).prepare(provenance)
closure_method = phx.discretization.PseudospectralMethodPlan(
    dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
)
les_plan = phx.equations.PeriodicAlgebraicLESPlan(
    prepared_model,
    phx.equations.PeriodicFourierGridFilterPlan(resolved_filter),
    closure_method,
)
dynamics = phx.equations.compile_periodic_incompressible_flow(
    problem,
    spectral_space,
    resolved_method,
    algebraic_les=les_plan,
)
```

The resolved `resolved_method` still owns the quadratic rotational term. The separate
`closure_method` owns oversampled nonpolynomial stress evaluation. Neither method ID is
the resolved filter ID. `PeriodicFourierGridFilterPlan` uses directional widths
`domain_length / retained_physical_count` and removes every mode on an even-grid
Nyquist plane.

`dynamics.stage(time, velocity, args)` evaluates the LES stress once and returns named
advective, molecular, SGS, forcing, nonlinear, and total rates. Reuse
`stage.algebraic_les` when asking for a step restriction or statistics; passing evidence
from another prepared action fails identity checks.

WALE, Vreman, and AMD use the same preparation and backend route. Changing the formula,
coefficient, provenance, filter, discretization, oversampling, or energy tolerance
changes a scientific/prepared identity and requires replanning.

## Dynamic periodic and MAC evaluation

Dynamic Smagorinsky is a filter-pair-bound evaluation, not an option hidden inside the
static compiler. Construct distinct resolved/test `ResolvedLESFilter` values,
`DynamicLESProvenance`, explicit averaging, denominator, and backscatter policies, and
then prepare `DynamicSmagorinskyPlan`.

For Fourier evaluation, `PeriodicFourierTestFilterPlan` prepares against a strictly
coarser three-axis Fourier discretization of the identical periodic domain. Its
resolved-to-coarse transfer and embedded retained mask define the test filter exactly.
For MAC evaluation, `MACExplicitTestFilterPlan` is fixed to the separable binomial
kernel and width ratio two; preparation rejects nonuniform, nonperiodic, fewer-than-
three-cell axes, and any physical boundary stage.

Lagrangian averaging requires explicit continuation:

```python
history = prepared_dynamic.initial_state(
    velocity,
    accepted_update_mask=True,
)
stage = prepared_dynamic.evaluate(
    velocity,
    history,
    accepted_update_mask=accepted_step_mask,
)
history = stage.continuation_state
```

Only commit the returned history with the accepted fluid state. Global,
homogeneous-plane, and local-kernel averaging reject a continuation state. The
denominator and backscatter policies are part of the prepared identity; no runtime
regularization or clipping is inferred.

`PeriodicDynamicLESPlan` and `MACDynamicLESPlan` are also compiler inputs, mutually
exclusive with static algebraic LES. Periodic compilation additionally requires the
exact coarser test discretization. `PreparedPeriodicDynamicETDRKMethod` and
`PreparedMACDynamicExplicitMethod` enforce current-state restrictions and commit
Lagrangian history only with accepted velocity. The periodic production assembler
constructs the transactional ETDRK wrapper; periodic-uniform MAC uses the prepared
explicit method directly because wall-plane statistics are scientifically
incompatible with that route.

## Static periodic time integration

The molecular diagonal remains inside ETDRK while static algebraic stress is a
state-dependent nonlinear rate. `LESStabilityGuardedETDRKMethod` evaluates the complete
first equation stage once, reuses its nonlinear rate, and admits a step only when

```text
step_size <= safety_factor * restriction.etdrk_selected
```

with finite, dissipative, energy-consistent LES evidence. It accepts only compiled
periodic algebraic LES. Ordinary prepared ETDRK is refused by LES production, and a
guarded method is refused for no-LES dynamics.

```python
coordinates = spectral_space.real_coordinates(component_shape=(3,))
method = phx.solver.LESStabilityGuardedETDRKMethod(
    phx.solver.ETDRKMethod(4),
    safety_factor=0.8,
).prepare(dynamics, coordinates=coordinates)
```

The live state remains full complex. Hermitian coordinates are a validation/checkpoint
encoding, not a packed runtime or reduced nonlinear-work claim.

## MAC and scalar LES

`MACAlgebraicLESPlan(prepared_model)` prepares against one
`PreparedMACMomentumOperators`. The filter must be the matching 3-D tensor-product
implicit grid-volume filter, the provenance regime must be
`incompressible-unit-density`, and the discretization IDs must agree. Local widths are
factored cell-axis interval widths. Active boundaries are restricted to periodic,
free-slip, and symmetry impermeable sides: no-slip, open, and inflow boundaries are
not accepted by this adapter.

`compile_mac_incompressible_flow(..., algebraic_les=plan)` exposes the pre-projection
LES stage, variational integrated work/boundary power, projected rate, and the
`MACLESStepRestriction` with advective, molecular, SGS, combined limits and explicit
`sgs_supported` status.

For transported cell scalars, closure is deliberately complete and named:

```python
scalar_sgs = phx.discretization.MACScalarSGSPlan(
    (
        phx.discretization.MACScalarSGSField(
            "temperature", turbulent_prandtl_number=0.7
        ),
        phx.discretization.MACScalarSGSField(
            "salinity", turbulent_schmidt_number=0.9
        ),
    )
)
```

Every required scalar declares exactly one positive turbulent Prandtl number, positive
turbulent Schmidt number, or `no_sgs=True`. A prepared set must match the required
field names exactly. Runtime diffusivity is `kinematic_eddy_viscosity / turbulent_number`
or zero for `no_sgs`; there is no default turbulent number. Supported scalar boundary
semantics are periodic, impermeable zero flux, or prescribed total flux.

Algebraic momentum LES and prognostic KSGS are alternatives. If either is active,
`compile_mac_scalar_buoyancy` requires scalar SGS, and scalar SGS cannot be active
without one of those momentum closures.

### Frozen implicit MAC methods

`MACIMEXEulerMethod(dynamics, ...)` and `MACSBDF2Method(dynamics, step_size, ...)`
recognize a stateless prepared algebraic LES with positive coefficient and place
molecular plus frozen eddy viscosity in a native variable-viscosity momentum action.
Only iterative momentum and composite-pressure solves are supported for this profile;
transform/hybrid requests are refused.

IMEX Euler evaluates and freezes the coefficient once from the accepted state for the
attempt. SBDF2 uses backward-Euler startup; later steps project the extrapolated
`2*u[n] - u[n-1]` state at `t[n+1]`, evaluate the coefficient there once, and retain
the complete two-step history. Results report coefficient state/time, frozen
viscosity, LES stage, stage/inverse identities, status, and rollback-safe accepted
state. These frozen implicit methods remain restricted to stateless algebraic LES;
compiled dynamic LES instead uses `PreparedMACDynamicExplicitMethod`. SBDF2 uses an
exact fixed step and does not support adaptive step changes.

## Ocean and prognostic KSGS

`CartesianBoussinesqOceanPlan` now accepts either `algebraic_les` or `ksgs`, together
with the complete `scalar_sgs` plan. Temperature and salinity declarations must exactly
match `LinearSeawaterReference.field_names`. KSGS additionally requires a distinct
`ksgs_field_name` and explicit nonnegative initial `sgs_kinetic_energy`.

The prepared MAC KSGS backend supports static, buoyant, dynamic, and low-Re plans.
It transports `k` with the coupled scalar machinery and uses returned eddy viscosity
in momentum and named scalar diffusivities. Dynamic KSGS requires a periodic-uniform
grid, exact binomial test filter with ratio two, and explicit averaging/update data;
that route is not compatible with a bounded vertical ocean grid. Low-Re KSGS derives
cell-center distance only from true no-slip boundaries and requires at least one;
an ocean preparation may use it only with matching caller-supplied no-slip momentum
boundaries. Free-slip and symmetry sides are never treated as low-Re walls.

Ocean remains a 3-D Cartesian rigid-lid, nonhydrostatic Boussinesq process model with
constant f-plane rotation. LES does not add a free surface, bathymetry/partial cells,
open boundaries, nonlinear seawater EOS, vertically implicit turbulence, or
distributed end-to-end ocean execution. See [Ocean process modeling](guides_ocean.md)
for forcing, budgets, restart, and output.

## Wall-resolved spectral channel

`channel_les_filter(channel.discretization)` creates the only accepted filter identity
for `compile_channel_les(base_channel, prepared_model)`. The route uses retained
Fourier spacings and retained Chebyshev nodal measures interpolated to the nonlinear
evaluation grid. Since the wall-normal width varies, filter and derivative do not
commute. `ChannelLESFilterGeometry.noncommutation_evidence` and
`wall_normal_scale_commutator` expose that effect diagnostically; no commutator model
is applied.

The compiled route evaluates all nine velocity-gradient components and the negative
stress divergence on the existing dealiased grid, restricts it to retained modes, and
lets the Stokes solve impose pressure and incompressibility. By default it remains
wall resolved with prescribed wall velocity.

`VectorEquilibriumWallStressPlan.prepare_channel(...)` creates the sole tangential
owner for both stationary channel walls. It samples velocity off-wall at
`y_lower + distance_lower` and `y_upper - distance_upper` by evaluating the retained
Chebyshev expansion, then zeroes the normal component before applying the wall law.
Normal velocity remains essential while explicit equilibrium traction enters SBDF2.
The route requires zero prescribed pressure gradient, positive density/viscosity,
interior sample coordinates, bounded roughness, and complete wall/energy evidence.

`ChannelLESEnergyLedger` reports molecular and SGS sinks and applied wall power.
`ChannelLESExplicitRestriction` now combines retained-horizontal and Chebyshev
wall-normal rotational-advection bounds with frozen SGS diffusion, uses stability
radius 0.25, and is enforced before every accepted channel SBDF2 step. It is a
complete bound for this declared explicit partition, not a universal bound for
other channel methods.

## Distributed periodic Fourier action

Wrap one already prepared static periodic scientific action:

```python
distributed = phx.discretization.DistributedPeriodicLESPlan(
    dynamics.algebraic_les,
    topology,
    schedule="pencil",
    checkpoint_count=checkpoint_count,
    maximum_bytes=memory_limit,
).prepare()
```

Only `slab` and `pencil` schedules are accepted. Preparation binds canonical/padded
layouts, transposes, global reductions, closure workspace, topology, resource ceiling,
and the scientific prepared ID. Evaluation keeps `NamedSharding` and performs no host
gather. `compile_distributed_periodic_les` adds dealiased rotational advection,
molecular diffusion, forcing, Leray projection, pressure-driving evidence, and the
distributed SGS action.

`DistributedPeriodicLESMethodPlan` selects ETDRK2/4 or SSPRK33/54 and enforces the
globally reduced current-state restriction. `DistributedPeriodicLESStatisticsPlan`
keeps scalar reductions sharded. `DistributedPeriodicLESProductionPlan` binds the
full equation, method, statistics, outputs, moments, triggers, and checkpoint capacity
to a shared runtime configured `device_resident=True`; segment results and restart
state are re-placed on the exact topology/layout. `checkpoint_count>=1` is required.
Qualification remains `backend-specific-not-inherited`; parity still does not imply
scaling or release.

## Unstructured low-Mach Favre LES

`UnstructuredLowMachLESPlan` composes a `PreparedFavreLESModel` with optional
`StaticKSGSPlan`. It prepares only on a fixed, conforming, 3-D tetrahedral
`PreparedUnstructuredCollocatedOperators` whose directional control-volume widths
exactly equal the model widths. The filter is wall-bounded unstructured implicit
volume, with unmodeled commutation and repeated filtering.

`semidiscrete_rate` remains the constitutive transport call and accepts caller-supplied
pressure. `UnstructuredLowMachLESFixedStepMethod` closes execution with a gauged
matrix-free pressure projection and a fixed-step forward-Euler predictor/correction.
The committed restart stores the pressure-corrected face-normal velocity and the
recomputed authoritative mass flux from the corrected rate--never the predictor
flux--together with pressure increment and accepted-step count. Advection, diffusion,
source, positivity, pressure compatibility/gauge/residual, divergence, transition
conservation, shared mass flux, resolved/SGS energy, and admissibility all gate the
atomic commit.

When static KSGS is active, raw production is negative shared deviatoric SGS
face work, equally split between adjacent cells and volume normalized. Negative
raw production fails. The limiter-retained KSGS gain plus the rejected
production exactly equals raw transfer: the rejected amount is thermalized as a
modeled enthalpy-density source, never discarded. Step evidence gates that split
and the total resolved-plus-KSGS-plus-enthalpy balance.

The numerical flux remains piecewise-constant upwind with Rhie–Chow stabilization,
deferred nonorthogonal gradients, one authoritative mass flux, zero molecular bulk
viscosity, and closed impermeable/zero-traction/adiabatic/zero-species-flux
boundaries. Dynamic/low-Re KSGS, 2-D/polyhedral, periodic/open, moving, and coupled
meshes remain refused.

## Compressible Favre transport

`PreparedFavreLESModel` supplies physical SGS transport to
`HomogeneousMixtureCompressibleNavierStokesSystem`. Its species schema, canonical SI
units, filter/provenance, Prandtl/Schmidt numbers, viscosity bound, SGS-energy
dissipation coefficient, and SGS-energy Schmidt number are explicit.

With `isotropic_trace_policy="provided-sgs-kinetic-energy"`, conserved state appends
`rho*k_sgs` after total energy and primitive state appends `k_sgs` after temperature.
Total energy includes SGS energy; isotropic SGS pressure is hyperbolic; deviatoric
work, heat/species transport, and SGS-energy diffusion remain diffusive.
`FavreLESCoupledRate` puts production/dissipation only in the SGS-energy component,
reports its positivity step, and keeps total-energy source exactly zero. The
`neglected` policy retains the smaller state. Both require conserved gradients;
the primitive-gradient-only convenience remains unavailable.

Shock sensing, Riemann flux, limiting, and artificial viscosity remain separate
numerical stabilization. Binding Favre transport does not qualify an application.

## LBM collision-local Smagorinsky

`SmagorinskyCollisionPlan` is an athermal lattice-unit SRT correction, not the filtered
finite-volume/spectral stress action. It derives an effective relaxation time from the
raw nonequilibrium second moment using unit filter width and
`nu = c_s^2 * (tau - 1/2)`. Evidence exposes base/effective relaxation time,
molecular/effective/eddy viscosity, stress norm, coefficient activity, conserved-moment
defect, and support. Density must be positive and the base relaxation rate must lie
strictly in `(0, 2)`. A coefficient of zero recovers the base relaxation exactly.

## Wall stress, inflow, and immersed coupling

`VectorEquilibriumWallStressPlan` is a 2-D/3-D attached, isothermal, Newtonian,
zero-pressure-gradient wall-on-fluid traction law with optional bounded sand-grain
roughness. Besides direct evaluation, `prepare_channel` installs it as the sole
tangential owner of stationary spectral-channel walls. It retains no
adverse-pressure-gradient or separation claim.

`StochasticTurbulentInflowPlan` prepares compact mass-neutral or spectral
surface-divergence-certified modes without covariance repair. Its
`prepare_mac_boundary` route creates `PreparedStochasticTurbulentInflowMACBoundary`,
an accepted-step owner that draws exactly once, emits a concrete velocity-inflow
`MACBoundaryProvider` with exact material rate, carries scalar values, and commits
typed-key/sample-index continuation atomically. It requires covariance, mass, and
represented-divergence-compatible prepared modes; it still makes no temporal-
correlation or variable-density mass claim.

`FixedImmersedMACLESPlan` is the integrated immersed LES route. It binds one static
`MACAlgebraicLESPlan`, `MACImmersedBoundaryProjectionPlan`, stationary
`LagrangianMarkerKinematics`, caller-owned fixed cell fluid fractions, and one
canonical geometry ID. Preparation is restricted to single-device,
three-dimensional, unit-density MAC; fixed marker identities and transfer routes;
stationary active markers; and periodic, free-slip, or symmetry outer boundaries.
Moving, deforming, distributed, open-boundary, truncated-support, and changed-route
requests fail closed.

On active cells the directional width is scaled by the cube root of fluid volume
fraction. Eddy viscosity, deviatoric stress, and energy transfer are then weighted
by that fraction; solid-cell SGS stress is zero. The prepared stage reports SGS and
optional wall rates separately, plus integrated work, boundary power, filter/model/
geometry/marker/boundary/solver identities, and success.

Without `wall_stress`, the immersed projection enforces full marker no-slip. With a
prepared `VectorEquilibriumWallStressPlan`, the marker solve enforces the normal
constraint and the attached equilibrium law supplies tangential traction; active
normal, sample distance, roughness, positive molecular viscosity, wall-law
convergence, and dissipation are all required. This does not create adverse-pressure-
gradient, separation, or moving-wall-model support.

`compile_fixed_immersed_mac_les_flow` installs the prepared action in standard MAC
dynamics. `plan.imex_euler_method(...)` and `plan.sbdf2_method(...)` bind the existing
immersed pressure/marker solvers for full-no-slip or normal-plus-wall-stress
execution. `admission_regime()` reuses the prescribed-marker DNS admission tuple;
it does not mint another support owner. Legacy case `immersed-mac-wall-stress`
measures the normal-only IMEX wall-on/off tuple and explicitly does not claim SBDF2.
Separate case `immersed-mac-sbdf2-restart` covers full-vector constraint, serialized
startup/advanced history, restart, and both balance-ledger phases. Evidence from
either tuple cannot be transferred to the other.

See [Immersed-boundary coupling](guides_immersed_boundary.md#fixed-immersed-mac-les).

## Closure-data filtering and targets

Offline filtering does not reuse a runtime or dealiasing identity implicitly.
`FilterSpec` prepares shape-bound identity, box, Gaussian, or periodic spectral-cutoff
filters. `ReynoldsFilter` applies that filter directly. `FavreFilter` computes
`F(rho*phi)/F(rho)` and requires both raw and filtered density strictly above its
explicit floor. `LESFilterPair` records distinct primary/test semantic filters with
`primary-resolved` test input, but performs no filtering.

`filter_commutation` reports `F(D phi) - D(F phi)` for the declared discrete central
difference. `filter_refinement_commutation` compares restrict-then-filter with
filter-then-restrict. These are measured discrete defects, not declarations of
commutation.

For exact periodic analysis, `prepare_periodic_les_analysis` binds one source Fourier
discretization, one no-finer resolved Fourier discretization, the exact source-to-
resolved modal transfer, the runtime `ResolvedLESFilter`, and a reference manifest ID.
It produces exact finite-grid Reynolds stress, stress-divergence, positive-forward
energy-transfer, and named scalar-flux targets. `ClosureAnalysisDAG` and
`LESAnalysisReference` preserve source, resolved, transfer, filter, target, and
reference lineage. Full targets preserve trace; deviatoric targets remove it explicitly.

## Learned stress versus model-error assimilation

`LearnedStressBindingPlan` is an artifact-bound constant-density specific-deviatoric
stress contract. Feature layout/dtype/units/flow schema, output, target, filter,
discretization, regime, provenance, model artifact, and train-only normalizer IDs
must agree.

`PeriodicLearnedStressPlan` builds the fixed nine-component velocity-gradient ABI,
evaluates the bound stress, and owns Fourier divergence, Leray projection, momentum,
reality, energy-policy, and work evidence on a single-device 3-D periodic grid.
`MACLearnedStressPlan` owns conservative stress divergence and pressure-rate
projection on a single-device periodic-uniform 3-D MAC grid with a certified
transform projection. Both retain the selected signed/dissipative/bounded-backscatter
policy and exact feature schema; neither silently falls back to another boundary or
grid route.

`PeriodicModelErrorParameterization` instead produces a piecewise-constant additive
modal momentum-rate correction in an exact solenoidal Hermitian basis. Sparse
velocity observations cannot identify that correction as SGS stress: forcing,
filtering, discretization, observation, and resolved-model error are confounded. The
assimilation identity separately binds problem, compiler, filter, base forcing, and
observation IDs; training misfit drives the objective while holdout misfit is reported,
not optimized. Do not relabel model-error evidence as closure validation.

## Production, restart, and statistics

Periodic statistics are equation-bound:

```python
statistics = phx.applications.incompressible_flow.PeriodicModalTurbulenceStatisticsPlan(
    dynamics,
    bin_edges,
    tail_start_wavenumber=tail_start,
)
instantaneous = statistics.evaluate(
    time,
    modal_velocity,
    args,
    stage=stage,
    step_restriction=restriction,
)
```

The signature is `(dynamics, bin_edges, /, *, tail_start_wavenumber=None,
reality_tolerance=..., solenoidal_tolerance=...)`. It reports energy, molecular
dissipation, advective/SGS/forcing shells, resolved flux, scales/tails, work and
step-limit evidence. For dynamic LES it additionally binds provenance, averaging,
regularization, and backscatter IDs and reports coefficient extrema plus
regularization/accepted/rejected update counts. `evaluate` accepts explicit
continuation and accepted-update mask and rejects foreign stages/restrictions.

Periodic production requires an explicit case identity and the route-compatible method:

```python
case = phx.applications.incompressible_flow.PeriodicSpectralProductionCase(
    dynamics,
    initial_modal_velocity,
    case_id="decaying-isotropic-les",
)
plan = phx.applications.incompressible_flow.PeriodicSpectralProductionPlan(
    dynamics,
    method,
    statistics,
    case,
    start_time=0.0,
    end_time=end_time,
    step_size=step_size,
    checkpoint_interval=checkpoint_interval,
)
prepared = plan.prepare(checkpoint_root)
state = prepared.initialize(initial_modal_velocity)
result = prepared.run(state)
```

The constructor is `(dynamics, method, statistics, case, /, *, ...)`, not the removed
`(method, statistics, problem_id=...)` form. Static LES requires the matching
`PreparedLESStabilityGuardedETDRKMethod`. Dynamic LES takes ordinary prepared ETDRK
for the matching drift, then the plan installs `PreparedPeriodicDynamicETDRKMethod`;
Lagrangian state is initialized and committed transactionally. Static guard and
dynamic wrapper are mutually exclusive. Constant-power identities remain explicit;
OU forcing is unavailable with dynamic continuation.

`PreparedPeriodicSpectralProduction.initialize` accepts only the modal value bound
into `PeriodicSpectralProductionCase`. Accepted velocity, optional dynamic/OU
continuation, controller/RNG state, moments, triggers, output cursors, and checkpoint
generation move atomically. Restore requires matching manifest, plan, runtime,
encoding, content, and resolved-run identities.

Distributed production first binds the exact initial field:

```python
distributed_dynamics = (
    phx.applications.incompressible_flow.compile_distributed_periodic_les(
        problem,
        source_plan,
    )
)
distributed_case = (
    phx.applications.incompressible_flow.DistributedPeriodicLESProductionCase(
        distributed_dynamics,
        initial_modal_velocity,
        case_id="forced-hit",
    )
)
distributed_plan = (
    phx.applications.incompressible_flow.DistributedPeriodicLESProductionPlan(
        problem,
        source_plan,
        method,
        distributed_case,
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
        checkpoint_interval=checkpoint_interval,
    )
)
```

The constructor is `(problem, source_plan, method, case, /, *, ...)`. Its
slab/pencil runtime remains device-resident and its profile remains
backend-specific. Channel
production uses the exact-step SBDF2 lattice and may instead use the prepared mixed
wall-traction owner. Wall-bounded structured-MAC production retains plane statistics;
periodic-uniform dynamic MAC uses `PreparedMACDynamicExplicitMethod` directly rather
than claiming incompatible wall-plane production statistics.

## Diagnostics before claims

At minimum, retain the route's typed evidence:

- filter, provenance, formula, prepared action, discretization, projection, and
  compilation IDs;
- finite, symmetry/trace, divergence, reality/Nyquist, boundary, and pressure-gauge
  evidence applicable to the route;
- pointwise and integrated SGS transfer, modal/variational work, projection defect, and
  energy-identity defect;
- advective, molecular, SGS, and combined timestep restrictions with the selected
  temporal policy;
- scalar mass closure, heat/species fluxes, KSGS contribution/continuation, or Favre
  admissibility where active;
- resource, sharding, restart, and parity evidence only from the backend actually run;
- exact case, reference artifact, configuration, environment, and run identities.

A finite rate is not conservation evidence, a one-device parity result is not scaling
evidence, and a diagnostic success flag is not release evidence.

## Qualification and release boundary

`tools/large_eddy_simulation_qualification.py` is the LES campaign producer:

```text
python tools/large_eddy_simulation_qualification.py --output <directory>
```

The optional canonical inputs are
`--campaign benchmarks/large_eddy_simulation_qualification_campaign.json` and
`--matrix benchmarks/large_eddy_simulation_qualification_matrix.json`. The output
root contains `candidate.json`, `campaign.json`, `matrix.json`, `coverage.json`, and
the `raw/`, `evidence/`, `support/`, `run-specs/`, `profiles/`,
`base-profiles/`, and `references/` inventories. Each admitted generic
`ReferenceArtifactManifest` is written as `references/<manifest_id>.json`, and
`candidate.json` lists the bound `reference_manifest_ids`.

Artifacts and the candidate JSON on stdout are emitted before process termination.
Exit status is `0` only for `qualification_outcome="passed"`, `1` for `"failed"`,
and `2` for `"inconclusive"`; none of those codes changes release status.

The campaign has 28 exact case labels: `periodic-smagorinsky`, `periodic-amd`,
`periodic-vreman`, `periodic-wale`, `periodic-apriori-filter`,
`periodic-dynamic-smagorinsky`, `periodic-dynamic-production`,
`mac-momentum-scalar-boussinesq`, `mac-prognostic-ksgs`, `mac-dynamic-ksgs`,
`mac-frozen-imex`, `mac-frozen-sbdf2`, `ocean-low-re-ksgs`,
`spectral-channel-wale`, `channel-mixed-wall-stress`,
`channel-complete-restriction`, `stochastic-mac-inflow-owner`,
`distributed-periodic-slab`, `distributed-full-flow-production`,
`learned-stress-periodic`, `learned-stress-mac`, `favre-compressible-smoke`,
`favre-transported-sgs-dg`, `unstructured-low-mach-smoke`,
`unstructured-pressure-continuation`, `lbm-smagorinsky-smoke`,
`immersed-mac-wall-stress`, and `immersed-mac-sbdf2-restart`. Each label denotes
only its exact support and run spec.

The learned-stress cases exercise analytic bound predictors and backend contracts;
they do not claim trained-model accuracy or generalization.

Static Smagorinsky/AMD/Vreman/WALE cases gate formula and prepared-backend activity.
Every nonzero model route has a preregistered lower-bound activity gate, so a
zero-branch-only execution cannot satisfy its evidence.

Outputs use the existing generic contracts rather than another gate schema:

1. `ReferenceArtifactManifest` binds reference files, checksums, rights, provenance,
   units, nondimensionalization, uncertainty, and permitted use.
2. `ResolvedRunSpec` binds the exact configuration, build, backend, precision,
   topology, scheduler, repository, authentication policy, and support dependencies
   before execution.
3. `QualificationEvidence` records the reviewed result for one exact execution and
   evidence scope.
4. `QualificationMatrix` evaluates the named predicates and distinguishes failed proof
   from missing or inconclusive proof.
5. `SupportTuple` describes only the coordinates actually exercised.
6. `CapabilityProfile` assembles the exact tuples and dependencies.

`candidate.json` keeps status domains separate: `status` is
`unreleased-candidate`, `qualification_outcome` is `passed`, `failed`, or
`inconclusive`, and both `released` and `signed` are false. The completed frozen
campaign emitted `qualification_outcome="passed"` for the 28 exact cases above.
Predicate counts and mutable content IDs are read from emitted artifacts rather
than copied into documentation. This measured pass remains blocked by
`base-candidate-release-admission`, `independent-review`,
`release-gate-binding`, and `trusted-release-index-signature`, and cannot waive
its base incompressible dependency.

See [Solver evidence gates](guides_solver_evidence.md) for generic publication and
resource rules and [Incompressible flow API](api/solver/incompressible.md) for the base
candidate routes.
